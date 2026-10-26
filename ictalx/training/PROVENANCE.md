# Provenance of `ictalx/training/`

Every function and constant in this folder is copied character for character from the training codebase `DL_WITH_SSL_GA` at commit `e463396`. Only the import lines at the top of each file and the header comments were added. `tests/test_training_copies.py` re-checks the copies against the commit.

## `preprocessing_binary.py`

Source SHA-256 at `e463396`: `e41f9727d2e168c86bf5d5462dce81cd77ac3a942f43fb13f27358ab48570b1a`

| Name | Source lines |
| --- | --- |
| `fs` | 22–22 |
| `win_len` | 23–23 |
| `step` | 24–24 |
| `segments_per_chunk` | 27–27 |
| `notch_freq` | 30–30 |
| `notch_Q` | 31–31 |
| `compute_segment_grid` | 115–146 |
| `apply_notch_filter` | 151–176 |
| `remove_dc` | 179–197 |
| `robust_zscore` | 200–227 |
| `preprocess_chunk` | 230–257 |

## `eval_utils.py`

Source SHA-256 at `e463396`: `32f37a654caf82eddbff4a90fbfc0a42f34554e9cd70f6e09bcaaf6c1c5d1522`

| Name | Source lines |
| --- | --- |
| `AMPLITUDE_THRESHOLD` | 91–91 |
| `FS` | 355–355 |
| `WIN_LEN` | 356–356 |
| `STEP` | 357–357 |
| `SEGMENT_SEC` | 358–358 |
| `STEP_SEC` | 359–359 |
| `THRESHOLD` | 361–361 |
| `SMOOTHING_WIN` | 362–362 |
| `REFRACTORY_SEC` | 363–363 |
| `MIN_EVENT_SEC` | 364–364 |
| `MAX_EVENT_SEC` | 365–365 |
| `split_into_chunks` | 507–519 |
| `_refractory_merge` | 525–538 |
| `_min_duration_filter` | 541–546 |
| `_max_duration_filter` | 549–556 |
| `detect_events_in_chunk` | 559–654 |

## `tcn_utils.py`

Source SHA-256 at `e463396`: `67bb220683f85fc2a7b6a853eebd6d80c4705d7d2ebdf584c13c61d3dfa42422`

| Name | Source lines |
| --- | --- |
| `filter_extreme_segments` | 470–574 |
| `CausalConvBlock` | 746–821 |
| `MultiScaleTCN` | 2146–2294 |
