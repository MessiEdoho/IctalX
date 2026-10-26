"""Verbatim copies from DL_WITH_SSL_GA/eval_utils.py at commit e463396.

Source file SHA-256 at that commit: 32f37a654caf82eddbff4a90fbfc0a42f34554e9cd70f6e09bcaaf6c1c5d1522
Do not edit. Each block below is preceded by its source line numbers.
"""
import numpy as np

# eval_utils.py lines 91-91
AMPLITUDE_THRESHOLD = 1000.0     # matches train-side filter (apply_val_test_filter.py)
# eval_utils.py lines 355-355
FS              = 500       # Hz
# eval_utils.py lines 356-356
WIN_LEN         = 2500      # samples (= 5 s)
# eval_utils.py lines 357-357
STEP            = 1250      # samples (= 2.5 s)
# eval_utils.py lines 358-358
SEGMENT_SEC     = WIN_LEN / FS    # 5.0
# eval_utils.py lines 359-359
STEP_SEC        = STEP / FS       # 2.5
# eval_utils.py lines 361-361
THRESHOLD       = 0.5
# eval_utils.py lines 362-362
SMOOTHING_WIN   = 3
# eval_utils.py lines 363-363
REFRACTORY_SEC  = 30.0
# eval_utils.py lines 364-364
MIN_EVENT_SEC   = 25.0
# eval_utils.py lines 365-365
MAX_EVENT_SEC   = 120.0     # drop events longer than 2 min (artefact / stuck-model cleanup;


# eval_utils.py lines 507-519
def split_into_chunks(chrono_idx_arr):
    """Split a sorted chrono_idx array into contiguous chunks. Two
    consecutive chrono_idx values that differ by > 1 are separated by
    at least one filtered-out segment, so they break the chunk.
    Returns a list of (start_pos, end_pos) slice indices into the array.
    """
    if len(chrono_idx_arr) == 0:
        return []
    deltas = np.diff(chrono_idx_arr)
    breaks = np.where(deltas != 1)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends   = np.concatenate((breaks, [len(chrono_idx_arr)]))
    return list(zip(starts.tolist(), ends.tolist()))


# eval_utils.py lines 525-538
def _refractory_merge(events):
    """Merge consecutive events whose inter-event gap is < REFRACTORY_SEC."""
    merged = []
    for evt in events:
        if merged and (evt["start_sec"] - merged[-1]["end_sec"]) < REFRACTORY_SEC:
            merged[-1]["end_sec"] = evt["end_sec"]
            merged[-1]["seg_indices"].extend(evt["seg_indices"])
        else:
            merged.append({
                "start_sec":   evt["start_sec"],
                "end_sec":     evt["end_sec"],
                "seg_indices": list(evt["seg_indices"]),
            })
    return merged


# eval_utils.py lines 541-546
def _min_duration_filter(events, min_sec=None):
    """Drop events shorter than min_sec (default: MIN_EVENT_SEC)."""
    if min_sec is None:
        min_sec = MIN_EVENT_SEC
    return [evt for evt in events
            if (evt["end_sec"] - evt["start_sec"]) >= min_sec]


# eval_utils.py lines 549-556
def _max_duration_filter(events, max_sec=None):
    """Drop events longer than max_sec (default: MAX_EVENT_SEC). Catches
    artefact-driven super-long events and refractory-merged super-events
    that should not be considered single seizures."""
    if max_sec is None:
        max_sec = MAX_EVENT_SEC
    return [evt for evt in events
            if (evt["end_sec"] - evt["start_sec"]) <= max_sec]


# eval_utils.py lines 559-654
def detect_events_in_chunk(t_start, y_prob, mouse_id, chunk_id,
                           order="min_then_refractory",
                           min_event_duration_sec=None,
                           max_event_duration_sec=None):
    """Run smoothing + threshold + run-detection, then apply min-duration,
    refractory-merge (in the order specified), and finally max-duration.

    Parameters
    ----------
    order : {"refractory_then_min", "min_then_refractory"}, default
        "min_then_refractory" (selected by the 2 x 5 grid sweep on val;
        see STUDY_REPORT.txt Section 7.6.10).
        "min_then_refractory" drops short candidate events first, then
        merges surviving ones; matches the standard order in the clinical
        seizure-detection literature and Pareto-dominates the alternative
        on this dataset.
        "refractory_then_min" (legacy) merges first, then drops surviving
        short events; can rescue fragmented true detections at the cost
        of also rescuing fragmented false alarms.
    min_event_duration_sec : float or None
        Minimum event length. None -> use module-level MIN_EVENT_SEC.
    max_event_duration_sec : float or None
        Maximum event length. Applied as the FINAL step in both orderings.
        None -> use module-level MAX_EVENT_SEC. Events exceeding this are
        dropped (artefact / model-stuck cleanup).

    Returns
    -------
    (events, smoothed_probs, smoothed_preds)
        Chunks shorter than SMOOTHING_WIN bypass smoothing so the smoothed
        array length always equals the input length.
    """
    n = len(y_prob)
    if n == 0:
        return [], np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    if n < SMOOTHING_WIN:
        smoothed = y_prob.astype(np.float64).copy()
    else:
        kernel = np.ones(SMOOTHING_WIN) / SMOOTHING_WIN
        smoothed = np.convolve(y_prob, kernel, mode="same")
    preds = (smoothed >= THRESHOLD).astype(np.int64)

    raw_events = []
    in_event = False
    seg_idx_in_event = []
    for i in range(n):
        if preds[i] == 1 and not in_event:
            in_event = True
            seg_idx_in_event = [i]
        elif preds[i] == 1 and in_event:
            seg_idx_in_event.append(i)
        elif preds[i] == 0 and in_event:
            in_event = False
            raw_events.append({
                "start_sec":   float(t_start[seg_idx_in_event[0]]),
                "end_sec":     float(t_start[seg_idx_in_event[-1]]) + SEGMENT_SEC,
                "seg_indices": list(seg_idx_in_event),
            })
    if in_event:
        raw_events.append({
            "start_sec":   float(t_start[seg_idx_in_event[0]]),
            "end_sec":     float(t_start[seg_idx_in_event[-1]]) + SEGMENT_SEC,
            "seg_indices": list(seg_idx_in_event),
        })

    if order == "refractory_then_min":
        processed = _min_duration_filter(
            _refractory_merge(raw_events), min_event_duration_sec)
    elif order == "min_then_refractory":
        processed = _refractory_merge(
            _min_duration_filter(raw_events, min_event_duration_sec))
    else:
        raise ValueError(
            "Unknown post-processing order %r; expected 'refractory_then_min' "
            "or 'min_then_refractory'." % order)

    # Max-duration filter: applied as the FINAL step in both orderings so
    # it catches both raw long runs and refractory-merged super-events.
    # Events with duration > max_event_duration_sec are dropped (treated as
    # artefact / model-stuck cleanup; see STUDY_REPORT 7.6.9).
    processed = _max_duration_filter(processed, max_event_duration_sec)

    final = []
    for evt in processed:
        seg_idx = evt["seg_indices"]
        final.append({
            "mouse_id":     mouse_id,
            "chunk_id":     chunk_id,
            "start_sec":    round(evt["start_sec"], 4),
            "end_sec":      round(evt["end_sec"], 4),
            "duration_sec": round(evt["end_sec"] - evt["start_sec"], 4),
            "mean_prob":    round(float(np.mean(smoothed[seg_idx])), 6),
            "max_prob":     round(float(np.max(smoothed[seg_idx])), 6),
        })
    return final, smoothed, preds
