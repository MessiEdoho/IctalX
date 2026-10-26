"""End-to-end parity on m427: the app must give the 12 events of the published test evaluation.

Reference: MultiScaleTCN/evaluation/post_process_varing_sec/test/min_then_refractory/
MIN_EVENT_SEC_25s/test_event_details.csv (rows for m427). About 90 minutes on a laptop CPU.
Run with:  pytest -m slow
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ictalx import pipeline

PROJECT = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT\MultiScaleTCN")
EDF = Path(os.environ.get("ICTALX_M427_EDF", PROJECT / "m427.edf"))
REFERENCE = Path(os.environ.get(
    "ICTALX_M427_REFERENCE",
    PROJECT / "evaluation" / "post_process_varing_sec" / "test" / "min_then_refractory"
    / "MIN_EVENT_SEC_25s" / "test_event_details.csv"))
MODEL_DIR = Path(os.environ.get("ICTALX_MODEL_DIR", r"C:\Users\messi\ictalx_models"))


@pytest.mark.slow
@pytest.mark.skipif(not (EDF.exists() and REFERENCE.exists() and MODEL_DIR.exists()),
                    reason="m427, reference events or model files not available")
def test_m427_matches_reference_events():
    model = pipeline.load_model(MODEL_DIR)
    result = pipeline.run_detection(pipeline.EdfSource(EDF, "EEG"), model, label="m427")

    assert len(result.t_start_sec) == 392311
    assert int((~result.kept).sum()) == 61
    assert result.n_chunks == 4

    app = pd.DataFrame(result.events)[["start_sec", "end_sec", "duration_sec", "mean_prob", "max_prob"]]
    ref = pd.read_csv(REFERENCE)
    ref = ref[ref.mouse_id == "m427"][["start_sec", "end_sec", "duration_sec", "mean_prob", "max_prob"]]
    app, ref = app.reset_index(drop=True), ref.reset_index(drop=True)

    assert len(app) == len(ref) == 12
    for col in ["start_sec", "end_sec", "duration_sec"]:
        np.testing.assert_array_equal(app[col].to_numpy(), ref[col].to_numpy(), err_msg=col)
    for col in ["mean_prob", "max_prob"]:
        np.testing.assert_allclose(app[col].to_numpy(), ref[col].to_numpy(), rtol=0, atol=1e-6, err_msg=col)

    # The model must still hold the verified weights after inference.
    assert pipeline.sha256_file(MODEL_DIR / pipeline.WEIGHTS_FILE) == pipeline.WEIGHTS_SHA256
