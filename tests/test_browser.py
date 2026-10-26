"""Interface test in Microsoft Edge (headless) against a real server, on a 3.4 h slice of m427
holding reference seizures 1 and 2 (243450 s and 255312.5 s). About 2 minutes.
Run with:  pytest -m browser
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from edf_writer import write_edf
from ictalx import pipeline
from test_m427_parity import EDF, MODEL_DIR

pw = pytest.importorskip("playwright.sync_api")
REPO = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.browser,
              pytest.mark.skipif(not (EDF.exists() and MODEL_DIR.exists()), reason="m427 or model files not available")]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("browser")
    two = d / "m427_two_seizures.edf"
    x = pipeline.EdfSource(EDF, "EEG").read(243200 * 500, 255600 * 500) * 1e6
    write_edf(two, {"EEG": x}, 500, start="13.12.22", start_time="12.31.23")     # 16:58:03 + 243200 s
    fast = d / "fast_1000hz.edf"
    write_edf(fast, {"EEG": np.random.default_rng(0).normal(0, 50, 60000)}, 1000)
    return two, fast


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    port = free_port()
    log = open(tmp_path_factory.mktemp("log") / "server.log", "wb")     # never a pipe: a full pipe stalls uvicorn
    env = dict(os.environ, ICTALX_MODEL_DIR=str(MODEL_DIR), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "ictalx.server:app", "--port", str(port)],
                            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}/"
    for _ in range(60):
        try:
            urllib.request.urlopen(url)
            break
        except OSError:
            time.sleep(1)
    yield url
    proc.terminate()
    log.close()


def test_interface(server, files):
    two, fast = files
    errors = []
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server)

        # V05 is caught in the browser before any upload
        page.set_input_files("#fileInput", str(fast))
        page.wait_for_selector("#errorBox:not([hidden])")
        assert "V05" in page.inner_text("#errorBox") and page.is_disabled("#detectBtn")

        # header read in the browser
        page.click("#fileRemove")
        page.set_input_files("#fileInput", str(two))
        page.wait_for_selector("#edfFields:not([hidden])")
        assert page.input_value("#edfChan") == "EEG"
        assert page.inner_text("#edfRate") == "500 Hz"
        assert page.inner_text("#edfStart") == "2022-12-13 12:31:23"

        # detection with progress
        page.click("#detectBtn")
        page.wait_for_selector("#steps li.now[data-step='detect']", timeout=120000)
        page.wait_for_selector("#review:not([hidden])", timeout=300000)
        page.wait_for_function("document.querySelectorAll('#evBody tr').length === 2")
        rows = [page.locator("#evBody tr").nth(i).inner_text() for i in range(2)]
        assert "00h04m10.0" in rows[0] and "2022-12-13 12:35:33.000" in rows[0]      # m427 243450 s
        assert "03h21m52.5" in rows[1] and "2022-12-13 15:53:15.500" in rows[1]      # m427 255312.5 s

        # draft, unsaved prompt, Stay (Escape) and Discard
        page.locator("#evBody tr").first.focus()
        page.keyboard.press("r")
        assert page.is_visible("#unsavedFlag") and page.text_content("#rvStatus") == "Pending review"
        page.keyboard.press("ArrowDown")
        page.wait_for_selector("#unsavedModal:not([hidden])")
        page.keyboard.press("Escape")
        assert page.is_hidden("#unsavedModal") and page.inner_text("#dEvId") == "1"
        page.locator("#evBody tr").first.focus()
        page.keyboard.press("ArrowDown")
        page.click("#umDiscard")
        assert page.inner_text("#dEvId") == "2"

        # save with the keyboard
        page.keyboard.press("c")
        page.keyboard.press("s")
        page.wait_for_function("document.querySelector('#stReviewed').textContent.startsWith('1')")
        assert page.text_content("#rvStatus") == "Confirmed"

        # downloads carry the saved decision
        with page.expect_download() as dl:
            page.click("#dlCsv")
        text = Path(dl.value.path()).read_text()
        assert "EV-001" in text and text.strip().splitlines()[2].endswith(",confirmed")
        with page.expect_download() as dl2:
            page.click("#dlZip")
        assert dl2.value.suggested_filename.endswith("_report.zip")
        browser.close()
    assert errors == []
