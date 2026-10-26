"""IctalX detection pipeline (docs/product-spec.md, sections 7.1 to 7.4).

Every signal-processing and model step calls the verbatim training copies in
`ictalx.training` (DL_WITH_SSL_GA, commit e463396). This module only reads the
signal, feeds blocks and segments through those functions in the training
order, and collects the results.
"""
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import torch

from ictalx.training import eval_utils as ev
from ictalx.training import preprocessing_binary as pp
from ictalx.training import tcn_utils as tu

WEIGHTS_FILE   = "multiscale_tcn_final_weights.pt"
PARAMS_FILE    = "best_multiscale_params.json"
WEIGHTS_SHA256 = "f36105d61f49ca13ef710a6f1ea6344f93996aeded19164a2efed489bb3e99da"


class WeightsChecksumError(RuntimeError):
    """The weights file is not the one the pipeline was verified with."""


class Cancelled(Exception):
    """Raised from a progress callback to stop a running detection."""


# ---------------------------------------------------------------------------
# Signal sources: one channel, read block by block
# ---------------------------------------------------------------------------
class EdfSource:
    """One channel of an EDF file, read on demand (header-only open)."""

    unit_to_uv = 1e6                     # MNE returns EDF data in volts

    def __init__(self, path, channel):
        # include= reads only this channel, so MNE does not upsample it to the
        # rate of a faster channel in the same file.
        self.raw = mne.io.read_raw_edf(str(path), include=[channel], preload=False, verbose="ERROR")
        self.raw.pick([channel])
        self.channel = channel
        self.fs = float(self.raw.info["sfreq"])
        self.n_samples = int(self.raw.n_times)
        meas_date = self.raw.info["meas_date"]
        self.start = meas_date.replace(tzinfo=None) if meas_date is not None else None

    def read(self, start, stop):
        """Calibrated samples [start, stop) as float64.

        Same values as raw.get_data(start=start, stop=stop). get_data is not used
        because it first builds a time index for the whole recording (3.65 GB for
        m427).
        """
        return self.raw._read_segment(start=start, stop=stop).squeeze().astype(np.float64)


class ArraySource:
    """A signal already in memory, e.g. one CSV column."""

    def __init__(self, samples, fs, start=None, unit_to_uv=1.0, channel="signal"):
        self.samples = np.asarray(samples)
        self.fs = float(fs)
        self.n_samples = len(self.samples)
        self.start = start
        self.unit_to_uv = float(unit_to_uv)
        self.channel = channel

    def read(self, start, stop):
        return self.samples[start:stop].astype(np.float64)


@dataclass
class DetectionResult:
    events: list              # dicts from eval_utils.detect_events_in_chunk, in time order
    t_start_sec: np.ndarray   # start time of every segment on the grid
    y_prob: np.ndarray        # model probability per segment; NaN for removed segments
    kept: np.ndarray          # False for segments removed as extreme (spec 7.1, step 5)
    n_chunks: int             # contiguous runs of kept segments
    fs: float
    n_samples: int
    recording_start: datetime  # None for a CSV without a start time


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model(model_dir, device="cpu"):
    """Build M3 from the tuned hyperparameters and load the verified weights, frozen."""
    model_dir = Path(model_dir)
    weights = model_dir / WEIGHTS_FILE
    if sha256_file(weights) != WEIGHTS_SHA256:
        raise WeightsChecksumError(f"{weights} does not match SHA-256 {WEIGHTS_SHA256}")

    cfg = json.loads((model_dir / PARAMS_FILE).read_text(encoding="utf-8"))
    hp, bd = cfg["hyperparameters"], cfg["branch_dilations"]
    model = tu.MultiScaleTCN(
        num_filters=int(hp["num_filters"]),
        kernel_size=int(hp["kernel_size"]),
        dropout=float(hp["dropout"]),
        branch1_dilations=bd["branch1"],
        branch2_dilations=bd["branch2"],
        branch3_dilations=bd["branch3"],
        fusion=str(hp["fusion"]),
    )
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Grid and blocks
# ---------------------------------------------------------------------------
def segment_grid(n_samples):
    return pp.compute_segment_grid(int(n_samples), pp.win_len, pp.step)


def block_range(starts, first):
    """Sample range [c0, c1) of the block whose first segment is starts[first]."""
    group = starts[first:first + pp.segments_per_chunk]
    return int(group[0]), int(group[-1]) + pp.win_len


def preprocess_block_steps(raw_block, fs):
    """Steps 1-2 and 1-3 of preprocess_chunk for one block.

    Returns (notch + DC removed, robust z-score). The z-score array is exactly
    what preprocess_chunk returns: it is the same three calls in the same order.
    """
    step2 = pp.remove_dc(pp.apply_notch_filter(raw_block, fs, pp.notch_freq, pp.notch_Q))
    return step2, pp.robust_zscore(step2)


def is_extreme(segment, threshold=1000.0):
    """Removal rule of tcn_utils.filter_extreme_segments, applied to one segment.

    filter_extreme_segments reads .npy files from disk, so it cannot be called on
    an in-memory segment; this repeats its two checks: any NaN/Inf, or
    max(|x|) > threshold.
    """
    if not np.isfinite(segment).all():
        return True
    return float(np.abs(segment).max()) > threshold


def iter_preprocessed_blocks(source):
    """Yield (first_segment_index, segment_starts, block_start, preprocessed_block).

    Same grid, 40-segment blocks and preprocessing order as preprocessing_binary.py.
    """
    starts = segment_grid(source.n_samples)
    for i in range(0, len(starts), pp.segments_per_chunk):
        c0, c1 = block_range(starts, i)
        block = pp.preprocess_chunk(source.read(c0, c1), source.fs, pp.notch_freq, pp.notch_Q)
        yield i, starts[i:i + pp.segments_per_chunk], c0, block


# ---------------------------------------------------------------------------
# Inference and post-processing
# ---------------------------------------------------------------------------
def predict(model, source, batch_size=256, device="cpu", progress=None):
    """Segment probabilities for the whole recording.

    Returns (t_start_sec, y_prob, kept). Removed segments get NaN and kept=False.
    progress(processed, total) is called after every batch, if given; total counts
    all grid segments. The callback may raise Cancelled to stop.
    """
    n_segs = len(segment_grid(source.n_samples))
    t_start = np.empty(n_segs, dtype=np.float64)
    y_prob = np.full(n_segs, np.nan, dtype=np.float64)
    kept = np.zeros(n_segs, dtype=bool)
    batch, batch_idx = [], []
    seen = 0

    def flush():
        nonlocal batch, batch_idx
        if batch:
            x = torch.from_numpy(np.stack(batch)).unsqueeze(1).to(device)    # (B, 1, win_len)
            with torch.no_grad():
                y_prob[batch_idx] = torch.sigmoid(model(x)).cpu().numpy()
            batch, batch_idx = [], []
        if progress:
            progress(seen, n_segs)

    for i, group, c0, block in iter_preprocessed_blocks(source):
        for j, s in enumerate(group):
            k = i + j
            seen = k + 1
            t_start[k] = int(s) / source.fs
            segment = block[int(s) - c0:int(s) - c0 + pp.win_len].astype(np.float32)
            if is_extreme(segment):
                continue
            kept[k] = True
            batch.append(segment)
            batch_idx.append(k)
            if len(batch) >= batch_size:
                flush()
    flush()
    return t_start, y_prob, kept


def detect_events(t_start, y_prob, kept, label="recording"):
    """Post-processing per contiguous chunk of kept segments, as in eval_utils.evaluate_event_level."""
    idx = np.nonzero(kept)[0]
    chunks = ev.split_into_chunks(idx)
    events = []
    for chunk_id, (s, e) in enumerate(chunks):
        found, _, _ = ev.detect_events_in_chunk(t_start[idx][s:e], y_prob[idx][s:e], label, chunk_id)
        events.extend(found)
    return events, len(chunks)


def run_detection(source, model, batch_size=256, device="cpu", progress=None, label="recording"):
    """Full detection on one channel: preprocessing, inference, post-processing."""
    if source.fs != pp.fs:
        raise ValueError(f"Sampling rate is {source.fs:g} Hz. IctalX currently accepts {pp.fs} Hz recordings only.")
    t_start, y_prob, kept = predict(model, source, batch_size, device, progress)
    events, n_chunks = detect_events(t_start, y_prob, kept, label)
    return DetectionResult(events, t_start, y_prob, kept, n_chunks, source.fs,
                           source.n_samples, source.start)
