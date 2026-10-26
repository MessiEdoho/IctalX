"""Quick regression check: first 4,000 m427 segments against probabilities saved from the
pipeline version that passed the full m427 parity test (commit e775ecb). About 1 minute."""
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from ictalx import pipeline
from test_m427_parity import EDF, MODEL_DIR

SAVED = Path(__file__).parent / "data" / "m427_first4000_probs.npy"


@pytest.mark.skipif(not (EDF.exists() and MODEL_DIR.exists()), reason="m427 or model files not available")
def test_first_4000_segments_unchanged():
    model = pipeline.load_model(MODEL_DIR)
    source = pipeline.EdfSource(EDF, "EEG")
    expected = np.load(SAVED)

    probs = np.full(4000, np.nan)
    segs, idx = [], []
    for i, group, c0, block in itertools.islice(pipeline.iter_preprocessed_blocks(source), 100):
        for j, s in enumerate(group):
            seg = block[int(s) - c0:int(s) - c0 + 2500].astype(np.float32)
            if not pipeline.is_extreme(seg):
                segs.append(seg)
                idx.append(i + j)
    x = np.stack(segs)
    with torch.no_grad():
        out = [torch.sigmoid(model(torch.from_numpy(x[k:k + 256]).unsqueeze(1))).numpy()
               for k in range(0, len(x), 256)]
    probs[idx] = np.concatenate(out)

    np.testing.assert_array_equal(np.isnan(probs), np.isnan(expected))
    np.testing.assert_allclose(probs[idx], expected[idx], rtol=0, atol=1e-7)
