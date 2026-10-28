"""events.csv and the summary report zip (docs/product-spec.md, section 9)."""
import csv
import io
import json
import zipfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ictalx import __version__
from ictalx import events as ev_mod
from ictalx.training import eval_utils as eu

TRAINING_COMMIT = "e463396"

CSV_COLUMNS = [
    "analysis_id", "event_id",
    "onset_s", "offset_s", "duration_s",
    "onset_elapsed", "offset_elapsed", "onset_clock", "offset_clock",
    "peak_uV", "kurtosis", "skewness", "envelope_uV",
    "model_confidence", "status",
]

README = """IctalX results
==============

events.csv          one row per detected event
summary.json        recording, pipeline and model details, counts by review status
events/<id>_waveform.png      preprocessed signal, 120 s before onset to 120 s after offset
events/<id>_spectrogram.png   spectrogram of the same window, seizure outlined in red

events.csv columns
------------------
analysis_id       ID of this analysis
event_id          EV-001, EV-002, ... in time order
onset_s           event start, seconds from recording start
offset_s          event end, seconds from recording start
duration_s        offset_s - onset_s
onset_elapsed     onset as HHhMMmSS.s from recording start
offset_elapsed    offset as HHhMMmSS.s from recording start
onset_clock       onset as date and time (blank if the recording has no start time)
offset_clock      offset as date and time (blank if the recording has no start time)
peak_uV           largest absolute value during the event, notch-filtered and mean-removed signal, in microvolts
kurtosis          excess kurtosis of the preprocessed signal during the event (0 for Gaussian noise)
skewness          skewness of the preprocessed signal during the event
envelope_uV       mean Hilbert amplitude envelope during the event, in microvolts
model_confidence  mean of the smoothed model probabilities of the event's segments
status            review decision: pending, confirmed or rejected

IctalX is a research aid for mouse EEG. It does not diagnose and is not a clinical device.

Citation: [citation to be added]
"""


def event_rows(a):
    rows = []
    for e in a.events:
        rows.append({
            "analysis_id": a.short_id,
            "event_id": e["event_id"],
            "onset_s": e["onset_s"],
            "offset_s": e["offset_s"],
            "duration_s": e["duration_s"],
            "onset_elapsed": ev_mod.to_hms(e["onset_s"]),
            "offset_elapsed": ev_mod.to_hms(e["offset_s"]),
            "onset_clock": ev_mod.clock_time(a.recording_start, e["onset_s"]) or "",
            "offset_clock": ev_mod.clock_time(a.recording_start, e["offset_s"]) or "",
            "peak_uV": round(e["peak_uV"], 3),
            "kurtosis": round(e["kurtosis"], 4),
            "skewness": round(e["skewness"], 4),
            "envelope_uV": round(e["envelope_uV"], 3),
            "model_confidence": round(e["mean_prob"], 6),
            "status": a.decisions[e["event_id"]],
        })
    return rows


def events_csv(a):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(event_rows(a))
    return buf.getvalue().encode("utf-8")


def summary(a, model_sha256):
    counts = {s: sum(1 for d in a.decisions.values() if d == s) for s in ("pending", "confirmed", "rejected")}
    return {
        "analysis_id": a.short_id,
        "created": a.created.isoformat(timespec="seconds"),
        "ictalx_version": __version__,
        "recording": {
            "file_name": a.file_name,
            "format": a.file_format,
            "channel": a.channel,
            "sampling_rate_hz": a.fs,
            "n_samples": a.n_samples,
            "duration_s": round(a.n_samples / a.fs, 3) if a.fs else None,
            "start": a.recording_start.isoformat(sep=" ") if a.recording_start else None,
        },
        "analysis": {
            "n_segments": a.n_segments,
            "n_segments_removed": a.n_removed,
            "n_contiguous_chunks": a.n_chunks,
            "n_events": len(a.events),
            "review_counts": counts,
        },
        "pipeline": {
            "training_code": f"DL_WITH_SSL_GA commit {TRAINING_COMMIT}",
            "model": "MultiScaleTCN (M3)",
            "model_weights_sha256": model_sha256,
            "post_processing": {
                "order": "min_then_refractory",
                "threshold": eu.THRESHOLD,
                "smoothing_window": eu.SMOOTHING_WIN,
                "min_event_duration_sec": eu.MIN_EVENT_SEC,
                "refractory_period_sec": eu.REFRACTORY_SEC,
                "max_event_duration_sec": eu.MAX_EVENT_SEC,
            },
        },
    }


def _shade_gaps(ax, window, t):
    """Grey hatch where the window has no valid data (padding or removed segments)."""
    bad = window.status != ev_mod.VALID
    if not bad.any():
        return
    edges = np.flatnonzero(np.diff(np.concatenate(([0], bad.astype(int), [0]))))
    for a, b in zip(edges[::2], edges[1::2]):
        label = "removed" if (window.status[a:b] == ev_mod.REMOVED).any() else "no data"
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color="0.85", hatch="//", lw=0)
        ax.text((t[a] + t[min(b, len(t) - 1)]) / 2, 0.5, label, transform=ax.get_xaxis_transform(),
                ha="center", va="center", fontsize=8, color="0.4")


def waveform_png(window, event_id):
    t = window.start_sec + np.arange(len(window.zscore)) / window.fs
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, window.zscore, lw=0.4, color="k")
    ax.axvspan(window.onset_sec, window.offset_sec, color="#F0A81E", alpha=0.25, label="seizure")
    _shade_gaps(ax, window, t)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time from recording start (s)")
    ax.set_ylabel("z-score")
    ax.set_title(f"{event_id}: preprocessed signal, 120 s before onset to 120 s after offset")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def spectrogram_png(window, event_id):
    spec = ev_mod.spectrogram(window)
    t, f = np.array(spec["t"]), np.array(spec["f"])
    db = np.array([c if c is not None else [np.nan] * len(f) for c in spec["db"]]).T
    fig, ax = plt.subplots(figsize=(12, 3))
    mesh = ax.pcolormesh(t, f, np.ma.masked_invalid(db), shading="nearest", cmap="viridis")
    ax.add_patch(Rectangle((window.onset_sec, f[0]), window.offset_sec - window.onset_sec, f[-1] - f[0],
                           fill=False, edgecolor="red", lw=2))
    ax.set_xlim(window.start_sec, window.start_sec + len(window.zscore) / window.fs)
    ax.set_ylim(0, window.fs / 2)
    ax.set_xlabel("Time from recording start (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{event_id}: spectrogram, same window")
    bar = fig.colorbar(mesh, ax=ax, label="Power")
    bar.set_ticks(mesh.get_clim(), labels=["Low", "High"])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def report_zip(a, model_sha256):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("events.csv", events_csv(a))
        z.writestr("summary.json", json.dumps(summary(a, model_sha256), indent=2))
        z.writestr("README.txt", README)
        for e in a.events:
            w = a.windows[e["event_id"]]
            z.writestr(f"events/{e['event_id']}_waveform.png", waveform_png(w, e["event_id"]))
            z.writestr(f"events/{e['event_id']}_spectrogram.png", spectrogram_png(w, e["event_id"]))
    return buf.getvalue()


def file_stem(a):
    return f"ictalx_{a.short_id}_{datetime.now():%Y%m%d}"
