"""Analyses held in memory, and the background detection queue (spec sections 8.3 and 10.3).

There is no database. An analysis lives until it has been idle for the session
length or the server restarts. The uploaded file is deleted as soon as its
analysis finishes, fails or is cancelled.
"""
import secrets
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ictalx import events as ev_mod
from ictalx import pipeline

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
REVIEW_STATES = ("pending", "confirmed", "rejected")


@dataclass
class Analysis:
    id: str
    short_id: str
    file_name: str
    file_format: str
    channel: str
    created: datetime
    upload_path: Path = None
    source: object = None
    status: str = QUEUED
    stage: str = "queued"
    processed: int = 0
    total: int = 0
    error: str = None
    fs: float = None
    n_samples: int = None
    recording_start: datetime = None
    n_segments: int = None
    n_removed: int = None
    n_chunks: int = None
    started: float = None
    finished: float = None
    events: list = field(default_factory=list)       # dicts: event_id, times, descriptors, probabilities
    windows: dict = field(default_factory=dict)      # event_id -> events.EventWindow
    decisions: dict = field(default_factory=dict)    # event_id -> pending / confirmed / rejected
    last_access: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)


class AnalysisStore:
    def __init__(self, session_seconds):
        self.session_seconds = session_seconds
        self._items = {}
        self._lock = threading.Lock()

    def create(self, file_name, file_format, channel):
        aid = uuid.uuid4().hex
        a = Analysis(aid, "AX-" + secrets.token_hex(3).upper(), file_name, file_format, channel, datetime.now())
        with self._lock:
            self._items[aid] = a
        return a

    def get(self, aid):
        with self._lock:
            a = self._items.get(aid)
        if a is not None:
            a.last_access = time.time()
        return a

    def queue_position(self, a):
        """1-based place among queued analyses, oldest first; 0 when not queued."""
        if a.status != QUEUED:
            return 0
        with self._lock:
            queued = sorted((x for x in self._items.values() if x.status == QUEUED), key=lambda x: x.created)
        return 1 + queued.index(a) if a in queued else 0

    def expire(self):
        """Drop analyses idle for longer than the session; cancel them if still running."""
        cutoff = time.time() - self.session_seconds
        with self._lock:
            old = [aid for aid, a in self._items.items() if a.last_access < cutoff]
            for aid in old:
                self._items[aid].cancel.set()
                del self._items[aid]
        return len(old)


def delete_upload(a):
    a.source = None                                  # release the file handle first
    if a.upload_path is not None:
        Path(a.upload_path).unlink(missing_ok=True)
        a.upload_path = None


class JobRunner:
    """Runs detections on a fixed number of worker threads; extra jobs wait in order."""

    def __init__(self, model, device="cpu", workers=1, batch_size=256):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="detect")

    def submit(self, a):
        self.pool.submit(self._run, a)

    def _run(self, a):
        try:
            if a.cancel.is_set():
                a.status, a.stage = CANCELLED, "cancelled"
                return
            a.status, a.stage, a.started = RUNNING, "detecting", time.time()   # preprocessing + inference, block by block

            def progress(processed, total):
                a.processed, a.total = processed, total
                if a.cancel.is_set():
                    raise pipeline.Cancelled()

            result = pipeline.run_detection(a.source, self.model, self.batch_size, self.device,
                                            progress, label=a.file_name)
            a.fs, a.n_samples, a.recording_start = result.fs, result.n_samples, result.recording_start
            a.n_segments = len(result.kept)
            a.n_removed = int((~result.kept).sum())
            a.n_chunks = result.n_chunks

            a.stage = "event windows"
            for n, e in enumerate(result.events, start=1):
                if a.cancel.is_set():
                    raise pipeline.Cancelled()
                eid = f"EV-{n:03d}"
                descriptors, window = ev_mod.describe_and_window(a.source, result.kept, e)
                a.events.append({
                    "event_id": eid,
                    "onset_s": e["start_sec"],
                    "offset_s": e["end_sec"],
                    "duration_s": e["duration_sec"],
                    "mean_prob": e["mean_prob"],
                    "max_prob": e["max_prob"],
                    **descriptors,
                })
                a.windows[eid] = window
                a.decisions[eid] = "pending"
            a.status, a.stage = DONE, "done"
        except pipeline.Cancelled:
            a.status, a.stage = CANCELLED, "cancelled"
        except Exception as exc:                       # reported to the user, details kept in the log
            a.status, a.stage, a.error = FAILED, "failed", str(exc)
            traceback.print_exc()
        finally:
            a.finished = time.time()
            delete_upload(a)
