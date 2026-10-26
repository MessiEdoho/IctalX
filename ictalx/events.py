"""Event descriptors and display windows (docs/product-spec.md, sections 7.5 and 7.6).

Both are computed after detection, while the recording is still available. The
signal is rebuilt from the same 40-segment blocks used for detection; where two
blocks share samples, the earlier block's value is used.
"""
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from matplotlib import mlab
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

from ictalx import pipeline
from ictalx.training import preprocessing_binary as pp

CONTEXT_SEC = 120.0          # seconds shown before onset and after offset

# Sample status inside a display window
NO_DATA = 0                  # outside the recording, or after the last full segment
REMOVED = 1                  # belongs only to removed segments (spec 7.1, step 5)
VALID   = 2


@dataclass
class EventWindow:
    start_sec: float         # window start, may be negative (padding before the recording)
    fs: float
    zscore: np.ndarray       # float32, preprocessed signal; NaN where status != VALID
    status: np.ndarray       # int8, NO_DATA / REMOVED / VALID per sample
    onset_sec: float
    offset_sec: float


def to_hms(sec):
    """Elapsed time as HHhMMmSS.s, as in eval_utils.evaluate_event_level (lines 928-933, commit e463396)."""
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h:02d}h{m:02d}m{s:04.1f}"


def clock_time(recording_start, sec):
    if recording_start is None:
        return None
    return (recording_start + timedelta(seconds=float(sec))).isoformat(sep=" ", timespec="milliseconds")


def rebuild_signal(source, kept, a, b):
    """Steps 1-2 (µV), z-score and status for samples [a, b) of the recording.

    a may be negative and b may exceed the recording; those samples are NO_DATA.
    """
    n = b - a
    step2_uv = np.full(n, np.nan)
    zscore = np.full(n, np.nan)
    filled = np.zeros(n, dtype=bool)
    status = np.full(n, NO_DATA, dtype=np.int8)

    starts = pipeline.segment_grid(source.n_samples)
    for first in range(0, len(starts), pp.segments_per_chunk):
        c0, c1 = pipeline.block_range(starts, first)
        if c1 <= a or c0 >= b:
            continue
        step2, z = pipeline.preprocess_block_steps(source.read(c0, c1), source.fs)
        lo, hi = max(a, c0), min(b, c1)
        dst = np.arange(lo - a, hi - a)
        take = ~filled[dst]                         # earlier block wins on shared samples
        step2_uv[dst[take]] = step2[lo - c0:hi - c0][take] * source.unit_to_uv
        zscore[dst[take]] = z[lo - c0:hi - c0][take]
        filled[dst] = True

        group = starts[first:first + pp.segments_per_chunk]
        for j, s in enumerate(group):
            s0, s1 = max(a, int(s)), min(b, int(s) + pp.win_len)
            if s1 <= s0:
                continue
            seg = status[s0 - a:s1 - a]
            if kept[first + j]:
                seg[:] = VALID
            else:
                seg[seg == NO_DATA] = REMOVED
    return step2_uv, zscore, status


def describe_and_window(source, kept, event):
    """Descriptors (7.5) and display window (7.6) for one detected event."""
    fs = source.fs
    onset, offset = float(event["start_sec"]), float(event["end_sec"])
    w0 = int(round((onset - CONTEXT_SEC) * fs))
    w1 = int(round((offset + CONTEXT_SEC) * fs))
    step2_uv, zscore, status = rebuild_signal(source, kept, w0, w1)

    i0, i1 = int(round(onset * fs)) - w0, int(round(offset * fs)) - w0
    ok = status[i0:i1] == VALID
    x_uv = step2_uv[i0:i1][ok]
    x_z = zscore[i0:i1][ok]
    descriptors = {
        "peak_uV":     float(np.max(np.abs(x_uv))),
        "kurtosis":    float(kurtosis(x_z)),
        "skewness":    float(skew(x_z)),
        "envelope_uV": float(np.mean(np.abs(hilbert(x_uv)))),
    }
    zscore[status != VALID] = np.nan
    window = EventWindow(w0 / fs, fs, zscore.astype(np.float32), status, onset, offset)
    return descriptors, window


def waveform_columns(window, n_columns=1200):
    """Min/max per display column, so peaks survive downsampling.

    Returns dict with t (column start times), lo, hi (None where no valid data)
    and status per column (VALID if any valid sample, else REMOVED or NO_DATA).
    """
    n = len(window.zscore)
    edges = np.linspace(0, n, n_columns + 1).astype(int)
    t, lo, hi, st = [], [], [], []
    for c in range(n_columns):
        a, b = edges[c], max(edges[c + 1], edges[c] + 1)
        seg = window.zscore[a:b]
        s = window.status[a:b]
        t.append(round(window.start_sec + a / window.fs, 4))
        if (s == VALID).any():
            v = seg[s == VALID]
            lo.append(round(float(v.min()), 4))
            hi.append(round(float(v.max()), 4))
            st.append(VALID)
        else:
            lo.append(None)
            hi.append(None)
            st.append(int(s.max()))
    return {"t": t, "lo": lo, "hi": hi, "status": st}


def spectrogram(window):
    """Spectrogram as in missed_seizure_audit.py: mean removed, NFFT = fs (1 s), 75% overlap,
    PSD shown in dB. Frequencies 0 to fs/2. Columns touching non-valid samples are None."""
    fs = window.fs
    nfft, nover = int(fs), int(fs * 0.75)
    x = window.zscore.astype(np.float64)
    valid = window.status == VALID
    x = np.where(valid, x - np.nanmean(x[valid]), 0.0)
    pxx, freqs, times = mlab.specgram(x, NFFT=nfft, Fs=fs, noverlap=nover)
    db = 10.0 * np.log10(np.maximum(pxx, 1e-20))
    hop = nfft - nover
    columns = []
    for k in range(db.shape[1]):
        a = k * hop
        if valid[a:a + nfft].all():
            columns.append([round(float(v), 1) for v in db[:, k]])
        else:
            columns.append(None)
    return {
        "t": [round(window.start_sec + float(tt), 3) for tt in times],
        "f": [float(f) for f in freqs],
        "db": columns,
    }
