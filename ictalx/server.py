"""IctalX web server (docs/product-spec.md, section 12).

Run locally:  uvicorn ictalx.server:app --port 8000
"""
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ictalx import events as ev_mod
from ictalx import exports, jobs, pipeline, validation
from ictalx.config import load_settings

CHUNK = 1 << 20                                         # upload copy size: 1 MB


class State:
    settings = None
    model = None
    model_sha256 = None
    store = None
    runner = None


def _sweeper(stop):
    while not stop.wait(300):                           # every 5 minutes
        State.store.expire()


@asynccontextmanager
async def lifespan(app):
    s = load_settings()
    State.settings = s
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    State.model = pipeline.load_model(s.model_dir, s.device)     # refuses to start on a checksum mismatch
    State.model_sha256 = pipeline.WEIGHTS_SHA256
    State.store = jobs.AnalysisStore(session_seconds=s.session_hours * 3600)
    State.runner = jobs.JobRunner(State.model, s.device, s.workers, s.batch_size)
    stop = threading.Event()
    threading.Thread(target=_sweeper, args=(stop,), daemon=True).start()
    yield
    stop.set()
    State.runner.pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="IctalX", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=2048)             # spectrograms are large JSON
app.mount("/static", StaticFiles(directory=load_settings().web_dir, check_dir=False), name="static")


def _analysis(aid):
    a = State.store.get(aid)
    if a is None:
        raise HTTPException(404, "Analysis not found. It may have expired; results are kept for 3 hours of inactivity.")
    return a


def _event(a, event_id):
    if event_id not in a.windows:
        raise HTTPException(404, "Event not found.")
    return a.windows[event_id]


def _save_upload(file):
    """Stream the upload to a temporary file, stopping at the 2 GB limit (V07)."""
    suffix = Path(file.filename or "").suffix.lower()
    path = State.settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    written = 0
    with open(path, "wb") as out:
        while chunk := file.file.read(CHUNK):
            written += len(chunk)
            if written > validation.MAX_BYTES:
                out.close()
                path.unlink(missing_ok=True)
                raise validation.ValidationError("V07", "Files larger than 2 GB are not supported.")
            out.write(chunk)
    return path


def _validation_response(err):
    return JSONResponse(status_code=422, content={"code": err.code, "message": err.message})


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    page = State.settings.web_dir / "index.html"
    if not page.exists():
        return Response("IctalX server is running. The interface is not installed yet.", media_type="text/plain")
    return FileResponse(page)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
@app.post("/api/detect", status_code=202)
def detect(
    file: UploadFile = File(...),
    file_format: str = Form(...),               # "EDF" or "CSV"
    channel: str = Form(None),                  # EDF
    signal_column: str = Form(None),            # CSV
    sampling_rate: str = Form(None),            # CSV
    unit: str = Form(None),                     # CSV
    time_column: str = Form(None),              # CSV, optional
    start_time: str = Form(None),               # CSV, optional
):
    try:
        path = _save_upload(file)
    except validation.ValidationError as err:
        return _validation_response(err)
    try:
        if file_format == "EDF":
            source = validation.validate_edf(path, channel)
        elif file_format == "CSV":
            source = validation.validate_csv(path, signal_column, sampling_rate, unit, time_column, start_time)
        else:
            raise validation.ValidationError("V01", "This file is not a valid EDF or CSV file.")
    except validation.ValidationError as err:
        path.unlink(missing_ok=True)
        return _validation_response(err)

    a = State.store.create(file.filename, file_format, source.channel)
    a.upload_path, a.source = path, source
    if file_format == "CSV":                    # the signal is already in memory
        source_path, a.upload_path = path, None
        source_path.unlink(missing_ok=True)
    State.runner.submit(a)
    return {"analysis_id": a.id, "short_id": a.short_id, "status": a.status}


@app.get("/api/analyses/{aid}")
def analysis_status(aid: str):
    a = _analysis(aid)
    body = {
        "analysis_id": a.id,
        "short_id": a.short_id,
        "file_name": a.file_name,
        "channel": a.channel,
        "status": a.status,
        "stage": a.stage,
        "queue_position": State.store.queue_position(a),
        "processed_segments": a.processed,
        "total_segments": a.total,
        "error": a.error,
    }
    if a.status == jobs.DONE:
        body["summary"] = exports.summary(a, State.model_sha256)
        body["events"] = [
            {**row, "max_prob": e["max_prob"]}
            for row, e in zip(exports.event_rows(a), a.events)
        ]
    return body


@app.post("/api/analyses/{aid}/cancel")
def cancel(aid: str):
    a = _analysis(aid)
    a.cancel.set()
    return {"status": "cancelling" if a.status in (jobs.QUEUED, jobs.RUNNING) else a.status}


@app.get("/api/events/{aid}/{event_id}")
def event_window(aid: str, event_id: str):
    a = _analysis(aid)
    w = _event(a, event_id)
    return {
        "event_id": event_id,
        "onset_s": w.onset_sec,
        "offset_s": w.offset_sec,
        "window_start_s": w.start_sec,
        "window_end_s": w.start_sec + len(w.zscore) / w.fs,
        "waveform": ev_mod.waveform_columns(w),
        "spectrogram": ev_mod.spectrogram(w),
        "status_codes": {"no_data": ev_mod.NO_DATA, "removed": ev_mod.REMOVED, "valid": ev_mod.VALID},
    }


# ---------------------------------------------------------------------------
# Review and reports
# ---------------------------------------------------------------------------
class Review(BaseModel):
    analysis_id: str
    event_id: str
    status: str


@app.post("/api/review")
def review(r: Review):
    a = _analysis(r.analysis_id)
    _event(a, r.event_id)
    if r.status not in jobs.REVIEW_STATES:
        raise HTTPException(400, "Status must be pending, confirmed or rejected.")
    a.decisions[r.event_id] = r.status
    return {"event_id": r.event_id, "status": r.status}


@app.get("/api/report")
def report(analysis_id: str, format: str = "csv"):
    a = _analysis(analysis_id)
    if a.status != jobs.DONE:
        raise HTTPException(409, "Results are not ready yet.")
    stem = exports.file_stem(a)
    if format == "csv":
        return Response(exports.events_csv(a), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{stem}_events.csv"'})
    if format == "zip":
        return Response(exports.report_zip(a, State.model_sha256), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{stem}_report.zip"'})
    raise HTTPException(400, "Format must be csv or zip.")


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------
@app.get("/api/samples")
def samples():
    d = State.settings.samples_dir
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".edf", ".csv")) if d.exists() else []
    return {"samples": [{"name": p.name, "bytes": p.stat().st_size} for p in files]}


@app.get("/api/samples/{name}")
def sample_file(name: str):
    d = State.settings.samples_dir.resolve()
    p = (d / name).resolve()
    if p.parent != d or not p.is_file():
        raise HTTPException(404, "Sample not found.")
    return FileResponse(p, filename=p.name)
