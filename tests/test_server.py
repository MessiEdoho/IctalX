"""End-to-end API test on a 20-minute slice of m427 that contains reference seizure 1 (243450-243485 s).

The slice starts at 242600 s, so the seizure is at about 850-885 s in the slice. Block boundaries
differ from the full recording, so times are checked to within one segment step, not exactly.
"""
import csv
import io
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from edf_writer import write_edf
from ictalx import pipeline
from test_m427_parity import EDF, MODEL_DIR

SLICE_START, SLICE_END = 242600, 243800       # seconds in m427
SEIZURE = (243450 - SLICE_START, 243485 - SLICE_START)

pytestmark = pytest.mark.skipif(not (EDF.exists() and MODEL_DIR.exists()),
                                reason="m427 or model files not available")


@pytest.fixture(scope="module")
def slice_edf(tmp_path_factory):
    src = pipeline.EdfSource(EDF, "EEG")
    x_uv = src.read(SLICE_START * 500, SLICE_END * 500) * 1e6
    path = tmp_path_factory.mktemp("edf") / "m427_slice.edf"
    # recording start 2022-12-10 16:58:03 + 242600 s = 2022-12-13 12:21:23
    write_edf(path, {"EEG": x_uv}, 500, start="13.12.22", start_time="12.21.23")
    return path


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("ICTALX_MODEL_DIR", str(MODEL_DIR))
    mp.setenv("ICTALX_UPLOAD_DIR", str(tmp_path_factory.mktemp("uploads")))
    from ictalx.server import app
    with TestClient(app) as c:
        yield c
    mp.undo()


def wait_done(client, aid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        body = client.get(f"/api/analyses/{aid}").json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(1)
    raise TimeoutError(aid)


def test_full_flow(client, slice_edf):
    with open(slice_edf, "rb") as f:
        r = client.post("/api/detect", files={"file": ("m427_slice.edf", f)},
                        data={"file_format": "EDF", "channel": "EEG"})
    assert r.status_code == 202, r.text
    aid = r.json()["analysis_id"]

    body = wait_done(client, aid)
    assert body["status"] == "done", body.get("error")

    events = body["events"]
    assert len(events) == 1
    e = events[0]
    assert abs(e["onset_s"] - SEIZURE[0]) <= 2.5 and abs(e["offset_s"] - SEIZURE[1]) <= 2.5
    assert e["onset_clock"].startswith("2022-12-13 12:3")
    assert e["status"] == "pending" and e["peak_uV"] > 0

    w = client.get(f"/api/events/{aid}/{e['event_id']}").json()
    assert w["window_start_s"] == pytest.approx(e["onset_s"] - 120)
    assert len(w["waveform"]["lo"]) == 1200
    assert w["spectrogram"]["f"][-1] == 250.0

    r = client.post("/api/review", json={"analysis_id": aid, "event_id": e["event_id"], "status": "confirmed"})
    assert r.status_code == 200

    rows = list(csv.DictReader(io.StringIO(client.get("/api/report", params={"analysis_id": aid}).text)))
    assert rows[0]["status"] == "confirmed" and rows[0]["event_id"] == "EV-001"

    z = zipfile.ZipFile(io.BytesIO(client.get("/api/report", params={"analysis_id": aid, "format": "zip"}).content))
    assert sorted(z.namelist()) == sorted(["events.csv", "summary.json", "README.txt",
                                           "events/EV-001_waveform.png", "events/EV-001_spectrogram.png"])


def test_upload_deleted(client, slice_edf):
    from ictalx.server import State
    with open(slice_edf, "rb") as f:
        aid = client.post("/api/detect", files={"file": ("m427_slice.edf", f)},
                          data={"file_format": "EDF", "channel": "EEG"}).json()["analysis_id"]
    wait_done(client, aid)
    assert list(State.settings.upload_dir.iterdir()) == []


def test_validation_error_returned(client, slice_edf):
    with open(slice_edf, "rb") as f:
        r = client.post("/api/detect", files={"file": ("m427_slice.edf", f)},
                        data={"file_format": "EDF", "channel": "EMG"})
    assert r.status_code == 422 and r.json()["code"] == "V04"


def test_unknown_analysis(client):
    assert client.get("/api/analyses/nope").status_code == 404
