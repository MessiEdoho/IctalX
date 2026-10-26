# IctalX

Web application for seizure detection in single-channel mouse EEG (pilocarpine model). It runs the
MultiScaleTCN (M3) and the pre- and post-processing of the training codebase `DL_WITH_SSL_GA`
(commit `e463396`), and lets a researcher review and export the detected events.

IctalX is a research aid. It does not diagnose and is not a clinical device.

The full specification is in [docs/product-spec.md](docs/product-spec.md); the wireframes are in
[docs/wireframes.html](docs/wireframes.html).

## Layout

| Path | Contents |
| --- | --- |
| `ictalx/training/` | Verbatim copies of the training functions; see `PROVENANCE.md` |
| `ictalx/pipeline.py` | Detection: reading, preprocessing, inference, post-processing |
| `ictalx/events.py` | Event descriptors, display windows, spectrograms |
| `ictalx/validation.py` | Upload checks V01 to V12 |
| `ictalx/jobs.py` | In-memory analyses and the background detection queue |
| `ictalx/exports.py` | `events.csv` and the report zip |
| `ictalx/server.py` | Web API |
| `tests/` | Tests; `test_m427_parity.py` is the full reference check |

## Setup

```
conda env create -f environment.yml
```

The model weights are not in git. Put `multiscale_tcn_final_weights.pt` and
`best_multiscale_params.json` in one folder and point `ICTALX_MODEL_DIR` to it. The server
refuses to start if the weights do not match the verified SHA-256.

## Run

```
set ICTALX_MODEL_DIR=C:\Users\messi\ictalx_models
conda run -n ictalx uvicorn ictalx.server:app --port 8000
```

Then open http://localhost:8000. Other settings are listed in section 12.1 of the spec.

## Test

```
conda run -n ictalx pytest            # fast tests, about 1 minute
conda run -n ictalx pytest -m slow    # full m427 parity test, about 1 hour on a laptop CPU
```

Tests that need `m427.edf`, the reference results or the weights are skipped when those files
are not available.
