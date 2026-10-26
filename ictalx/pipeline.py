"""IctalX detection pipeline (docs/product-spec.md, sections 7.1 to 7.4).

Every signal-processing and model step calls the verbatim training copies in
`ictalx.training` (DL_WITH_SSL_GA, commit e463396). This module only reads the
EDF, feeds blocks and segments through those functions in the training order,
and collects the results.
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


@dataclass
class DetectionResult:
    events: list              # dicts from eval_utils.detect_events_in_chunk
    t_start_sec: np.ndarray   # start time of every segment on the grid
    y_prob: np.ndarray        # model probability per segment; NaN for removed segments
    kept: np.ndarray          # False for segments removed as extreme (spec 7.1, step 5)
    n_chunks: int             # contiguous runs of kept segments
    fs: float
    n_samples: int
    recording_start: datetime


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


def open_edf(path, channel):
    """Open an EDF header-only and keep the selected channel."""
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    raw.pick([channel])
    return raw


def read_samples(raw, start, stop):
    """Calibrated samples [start, stop) of the picked channel, as float64.

    Same values as raw.get_data(start=start, stop=stop). get_data is not used
    because it first builds a time index for the whole recording (3.65 GB for
    m427).
    """
    return raw._read_segment(start=start, stop=stop).squeeze().astype(np.float64)


def is_extreme(segment, threshold=1000.0):
    """Removal rule of tcn_utils.filter_extreme_segments, applied to one segment.

    filter_extreme_segments reads .npy files from disk, so it cannot be called on
    an in-memory segment; this repeats its two checks: any NaN/Inf, or
    max(|x|) > threshold.
    """
    if not np.isfinite(segment).all():
        return True
    return float(np.abs(segment).max()) > threshold


def iter_preprocessed_blocks(raw, fs):
    """Yield (first_segment_index, segment_starts, block_start, preprocessed_block).

    Same grid, 40-segment blocks and preprocessing order as preprocessing_binary.py.
    """
    starts = pp.compute_segment_grid(int(raw.n_times), pp.win_len, pp.step)
    for i in range(0, len(starts), pp.segments_per_chunk):
        group = starts[i:i + pp.segments_per_chunk]
        c0 = int(group[0])
        c1 = int(group[-1]) + pp.win_len
        block = pp.preprocess_chunk(read_samples(raw, c0, c1), fs, pp.notch_freq, pp.notch_Q)
        yield i, group, c0, block


def predict(model, raw, fs, batch_size=256, device="cpu", progress=None):
    """Segment probabilities for the whole recording.

    Returns (t_start_sec, y_prob, kept). Removed segments get NaN and kept=False.
    progress(scored, total) is called after every batch, if given; total counts all
    grid segments, including any that end up removed.
    """
    n_segs = len(pp.compute_segment_grid(int(raw.n_times), pp.win_len, pp.step))
    t_start = np.empty(n_segs, dtype=np.float64)
    y_prob = np.full(n_segs, np.nan, dtype=np.float64)
    kept = np.zeros(n_segs, dtype=bool)
    batch, batch_idx = [], []
    done = 0

    def flush():
        nonlocal batch, batch_idx, done
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).unsqueeze(1).to(device)    # (B, 1, win_len)
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy()
        y_prob[batch_idx] = probs
        done += len(batch)
        batch, batch_idx = [], []
        if progress:
            progress(done, n_segs)

    for i, group, c0, block in iter_preprocessed_blocks(raw, fs):
        for j, s in enumerate(group):
            k = i + j
            t_start[k] = int(s) / fs
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


def run_detection(edf_path, channel, model, batch_size=256, device="cpu", progress=None):
    """Full detection on one EDF channel: preprocessing, inference, post-processing."""
    raw = open_edf(edf_path, channel)
    fs = float(raw.info["sfreq"])
    if fs != pp.fs:
        raise ValueError(f"Sampling rate is {fs:g} Hz. IctalX currently accepts {pp.fs} Hz recordings only.")
    meas_date = raw.info["meas_date"]
    if meas_date is None:
        raise ValueError(f"{edf_path} has no recording start time in its header.")
    t_start, y_prob, kept = predict(model, raw, fs, batch_size, device, progress)
    events, n_chunks = detect_events(t_start, y_prob, kept, Path(edf_path).stem)
    return DetectionResult(events, t_start, y_prob, kept, n_chunks, fs,
                           int(raw.n_times), meas_date.replace(tzinfo=None))
