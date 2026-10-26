# IctalX Product Specification

| Field | Value |
| --- | --- |
| Status | Draft 0.4, for review |
| Date | 2026-10-26 |
| Owner | Mercy Edoho, UCD |
| Source inputs | `IctalX_architecture_last_version.svg`, `IctalX_UI_FINAL.html` |
| Training codebase | `DL_WITH_SSL_GA` (preprocessing_binary.py, eval_utils.py, tcn_utils.py). `deploy_inference.py` is not a reference. |
| Model files | `multiscale_tcn_final_weights.pt` (1.8 MB), `best_multiscale_params.json` |
| Reference recording | `m427.edf` |
| Training environment | Python 3.10.20, PyTorch 2.5.1+cu118 |

Values marked **TBC** are open and will be confirmed by the owner. They are not to be guessed.

---

## 1. Purpose

IctalX is a web application that runs a trained MultiScale Temporal Convolutional Network (TCN) on a single channel of mouse EEG and returns candidate seizure events. A researcher reviews each event, marks it confirmed or rejected, and exports the results.

The target users work with the pilocarpine mouse model of epilepsy. IctalX is a research aid. It does not diagnose and is not a clinical device.

## 2. Goals and non-goals

### 2.1 Goals

1. Reproduce the published detection pipeline exactly, so that the web tool and the paper give the same events for the same recording.
2. Let a researcher go from file to reviewed event table in one session, in a web browser.
3. Run locally first, then move to UCD hosting with no change to the pipeline.
4. Keep no copy of the recording. The uploaded file is processed and then deleted.

### 2.2 Non-goals

1. Multi-channel analysis. One channel is analysed per run.
2. Human EEG or any clinical use.
3. Model training or fine-tuning inside the app.
4. User accounts, shared projects or long-term storage.
5. A database. Results live in server memory for the session only (section 8.3).
6. Manual drawing of new events. The reviewer can confirm or reject detected events only.

## 3. Users

| User | Need |
| --- | --- |
| Researcher / EEG reviewer | Upload a recording, check each detected event against the trace, export a clean event list. |
| Lab lead | Trust that the tool matches the published method; cite it. |
| External lab | Try the tool on sample data, then run it on their own machine. |

## 4. Deployment

One application, run in three places. The code is the same in all of them. A CPU is not enough for real use (section 14.1), so hosted stages run on a GPU.

### 4.1 Stage 1: local

| Part | Technology | Role |
| --- | --- | --- |
| Backend | Python, FastAPI, served by Uvicorn | Validation, detection pipeline, review state, exports |
| Pipeline | Functions copied verbatim from the training codebase (`DL_WITH_SSL_GA`, commit `e463396`) into this repo; each copy names its source file and commit. The m427 parity check runs as a test | Same functions that produced the paper |
| Model | PyTorch, original weights. The weights file is not stored in git; it is supplied at deployment. The app refuses to start unless its SHA-256 is `f36105d61f49ca13ef710a6f1ea6344f93996aeded19164a2efed489bb3e99da` | Inference, CPU by default, GPU if present |
| Interface | Static HTML, CSS and JavaScript, based on the prototype, served by FastAPI | Upload, review, export |
| Address | `http://localhost:8000` | Opened in any current browser |

The recording travels from the browser to the app on the same computer. It does not leave the machine.

### 4.2 Stage 2: temporary AWS hosting, for the focus group

| Part | Technology | Role |
| --- | --- | --- |
| Package | Same Docker image as stage 1 | No code differences from local |
| Compute | AWS GPU instance; type **TBC** after a timing test | Model inference on GPU |
| Users | 20 focus-group participants, possibly running detection at the same time | Jobs are queued and run one or a few at a time on the GPU; each user sees their own progress |
| Data | Sample recordings supplied by the owner | Files **TBC** |
| Region | Ireland (eu-west-1) | Where uploaded recordings are processed |
| Transport | HTTPS / TLS | Encrypts uploads and results |
| Upload limit | 2 GB | Matches validation rule V07 |
| Access | **TBC** (open link, or restricted to focus-group participants) | |
| Lifetime | Until the focus group ends; then shut down | |

Speed on the chosen GPU instance is measured with `m427.edf` before the focus group.

### 4.3 Stage 3: UCD hosting

| Part | Technology | Role |
| --- | --- | --- |
| Package | Same Docker image | Runs the same way on any server |
| Transport | HTTPS / TLS, set up by UCD IT | Encrypts uploads and results |
| Upload limit | 2 GB, set on the UCD web server | Matches validation rule V07 |
| Access | **TBC** with UCD (open, UCD network only, or sign-in) | |

The same Docker image is the "Local / offline version" in the architecture diagram, given to other labs.

### 4.4 Why this approach

1. The app uses the training code directly, so there is no second version of the pipeline to keep in step.
2. No model conversion is needed.
3. Moving between local, AWS and UCD means packaging and configuration, not new code.

## 5. System overview

```
Researcher
   |
   v
Browser interface
   |  HTTP (local) / HTTPS (UCD)
   v
FastAPI backend
   |
   v
Input and metadata handler   <- file + channel, sampling rate, unit, start time
   |
   v
Validation and input checks  -- fail --> Rejected, error shown, file deleted
   |
   pass
   v
Analysis ID assigned
   |
   v
Detection pipeline
   1. Preprocessing
   2. MultiScale TCN inference
   3. Event post-processing
   4. Detected seizure events
   5. Event windows cut out (±120 s per event)
   |
   v
Uploaded file and full signal deleted
   |
   v
Review manager (in memory)  <-->  Review workspace (event table, detail view)
   |
   v
Reporting and export  -->  events.csv, report .zip
```

## 6. Input specification

### 6.1 Accepted formats

| Format | Required from user | Read from file |
| --- | --- | --- |
| EDF / EDF+ | Channel to analyse | Sampling rate, physical unit, start date and time, channel labels |
| CSV | Signal column, sampling rate, unit; optional time column and start time | Signal values; time values if a time column is named |

CSV layout, taken from the prototype's CSV fields: comma-separated, UTF-8, first row is a header with column names, one sample per row, decimal point `.`. The user names the signal column (default `eeg`) and, optionally, a time column in seconds. Other columns are ignored.

Multi-channel EDF files are accepted. The user picks one channel and only that channel is read. Channel names differ between recordings, so the app does not assume a name such as `EEG`.

Reference recording `m427.edf`: one channel labelled `EEG`, unit V, physical range ±0.005 V, 16-bit, 500 Hz, 98,078 records of 10 s (272.4 h), start 10.12.22 16:58:03, file size 980.8 MB.

### 6.2 Metadata fields

| Field | EDF | CSV | Rule |
| --- | --- | --- | --- |
| Channel | Dropdown, filled from the file header | Signal column name | Must exist in the file. |
| Sampling rate | From header, shown read-only | Number entry, default 500 Hz | Must be exactly 500 Hz during the pilot, or the file is rejected (V05). No resampling. For CSV with a time column, the rate must agree with the time values. |
| Unit | From header | µV or mV | Robust z-scoring removes scale, so the unit does not change detection. |
| Time column | Not used | Optional | If given, must be monotonic and agree with the sampling rate. |
| Start time | From header, required | Optional | EDF files without a start date and time are rejected (V09). For CSV without a start time, clock times are left blank. |

For EDF files, the channel list needs the file header. The interface reads the header in the browser to fill the dropdown, so the full file is uploaded only once, when the user presses "Detect seizures".

### 6.3 Validation rules

A file is rejected, with a clear message, if any check fails. A rejected file is not analysed and is deleted at once.

| Code | Check | Message to user |
| --- | --- | --- |
| V01 | Extension and content are EDF or CSV | "This file is not a valid EDF or CSV file." |
| V02 | File parses without error | "The file appears to be damaged and could not be read." |
| V03 | File contains samples | "The file contains no signal data." |
| V04 | The chosen channel or column exists | "The selected channel was not found in this file." |
| V05 | Sampling rate is exactly 500 Hz (the training rate). Pilot rule; other rates may be supported later | "Sampling rate is X Hz. IctalX currently accepts 500 Hz recordings only." |
| V06 | Removed. Segments with NaN, infinite or extreme values are dropped silently (section 7.1, step 5); the file is not rejected. | |
| V07 | File size is 2 GB or less | "Files larger than 2 GB are not supported." |
| V08 | Recording is at least one segment long (5 s) | "The recording is shorter than 5 s." |
| V09 | EDF header has a recording start date and time | "This EDF file has no recording start time." |

### 6.4 Analysis ID

Each accepted recording gets a unique ID (UUID v4, shown in short form, for example `AX-7F3C2A`). The ID appears in the summary, in every export file name and inside the exported files.

## 7. Detection pipeline

The app reproduces the training codebase (`preprocessing_binary.py`, `eval_utils.py`, `tcn_utils.py`). `deploy_inference.py` is not used: it differs from training in block length and flat-signal handling.

### 7.1 Preprocessing

Order as in the revised architecture diagram (`IctalX_architecture_last_version.svg`).

Steps 1 to 3 run on processing blocks, not on the whole recording and not on single segments.

| Step | Setting | Source |
| --- | --- | --- |
| 0. Segment grid | Start indices 0, 1250, 2500, ... up to n_samples − 2500. Samples after the last full window are not analysed. | `compute_segment_grid` |
| 0. Blocks | Grid split into groups of 40 segments. A block runs from its first segment start to its last segment start + 2,500 samples (102.5 s for a full block; consecutive blocks overlap by 1,250 samples). The last block may be shorter. | `segments_per_chunk = 40` |
| 1. Notch filter | `iirnotch(50 / (fs / 2), 30)`, applied with `filtfilt` (default padding) on each block, float64 | `apply_notch_filter` |
| 2. DC removal | Subtract the block mean | `remove_dc` |
| 3. Robust z-score | (x − median) / (1.4826 × MAD), MAD with `scale=1`. If MAD = 0 the block is returned unchanged. | `robust_zscore` |
| 4. Segmentation | Each segment cut from its own block, cast to float32, shape (1, 2500) | main loop |
| 5. Removal of extreme segments | A segment (as float32) containing NaN, ±inf, or any value with abs(x) > 1000 is removed, not clipped. Done silently. | `tcn_utils.filter_extreme_segments`, via `apply_val_test_filter.py` |

### 7.2 Inference

| Item | Value |
| --- | --- |
| Model | M3, MultiScaleTCN without attention (`tcn_utils.MultiScaleTCN`) |
| Hyperparameters | num_filters 64, kernel_size 7, dropout 0.1, fusion concat |
| Branch dilations | [1, 2, 4], [8, 16, 32], [32, 64, 128]; two causal convolutions per block |
| Input | Batch of segments, shape (batch, 1, 2500) |
| Output | Seizure probability in [0, 1] per segment, after sigmoid |
| Batch size | Chosen for memory, does not change results |
| Precision | float32 |

### 7.3 Post-processing

Order: minimum duration, then refractory merge, as in the paper (`order="min_then_refractory"`). Source: `eval_utils.detect_events_in_chunk`.

Removed segments leave gaps. The surviving segments are split into contiguous chunks (`eval_utils.split_into_chunks`) and steps 1 to 6 run on each chunk separately, as in `evaluate_event_level`. Events never span a gap.

| Step | Setting |
| --- | --- |
| 1. Probability smoothing | `np.convolve(p, ones(3)/3, mode="same")`; skipped if fewer than 3 segments |
| 2. Thresholding | Smoothed probability ≥ 0.5 |
| 3. Grouping | Consecutive positive segments form one event. Onset = start of first segment; offset = start of last segment + 5 s |
| 4. Minimum-duration filter | Keep events with duration ≥ 25 s |
| 5. Refractory merging | Merge an event into the previous one if the gap between them is < 30 s |
| 6. Maximum-duration filter | Keep events with duration ≤ 120 s |

### 7.4 Event confidence

The event probability is the mean of the smoothed probabilities of the segments in the event. For merged events, segments in the gap are not included.

### 7.5 Event descriptors

Shown in the detail view and exported in `events.csv`. The four signal descriptors are not in the training codebase; they are new to the app. Their formulas were approved by the owner on 2026-10-26. Peak amplitude and envelope are in µV (7.5.3), confirmed on 2026-10-26.

#### 7.5.1 Event samples

| Item | Rule |
| --- | --- |
| Interval | From onset to offset of the final event (after post-processing, section 7.3) |
| Sample indices | i0 = round(onset × fs) to i1 = round(offset × fs), end excluded; N = i1 − i0 |
| Gaps | Events never span removed segments, because post-processing runs per contiguous chunk (7.3) |
| Block borders | An event of 25 to 120 s can cross one or two 40-segment block borders. Samples are taken as in 7.6 (shared samples from the earlier block) |
| Precision | Calculated in float64 |

#### 7.5.2 Definitions (approved)

x is the event signal (7.5.3), m its mean.

| Descriptor | Unit | Formula | Implementation | Reading |
| --- | --- | --- | --- | --- |
| Onset | s | Start of the event | Section 7.3 | Elapsed time and clock time |
| Offset | s | End of the event | Section 7.3 | Elapsed time and clock time |
| Duration | s | Offset − onset | Section 7.3 | 25 to 120 s by construction |
| Peak amplitude | See 7.5.3 | max(abs(x)) | `np.max(np.abs(x))` | Largest swing during the event |
| Kurtosis | none | mean((x − m)^4) / mean((x − m)^2)^2 − 3 | `scipy.stats.kurtosis(x)` (Fisher, biased; defaults) | 0 for Gaussian background; high for spiky discharges |
| Skewness | none | mean((x − m)^3) / mean((x − m)^2)^1.5 | `scipy.stats.skew(x)` (biased; defaults) | Positive if spikes point mainly up, negative if down |
| Envelope | See 7.5.3 | mean(abs(hilbert(x))) | `np.mean(np.abs(scipy.signal.hilbert(x)))` | Average oscillation strength during the event |
| Probability | 0 to 1 | Mean of smoothed segment probabilities | Section 7.4 | Model confidence |

Kurtosis and skewness do not depend on scale, so they are calculated on the preprocessed signal (the waveform samples).

#### 7.5.3 Signal for peak amplitude and envelope

| Option | Signal | Unit | Status |
| --- | --- | --- | --- |
| (a) | Preprocessed signal, steps 1 to 3 (z-score) | Multiples of the block's robust spread | Not adopted |
| (b) | Steps 1 and 2 only (notch filter, block mean removed), converted to µV | µV | Adopted (confirmed 2026-10-26) |

Reasons for adopting (b):

1. Robust z-scoring divides by the spread of the whole 102.5 s block. A seizure of 25 to 120 s fills a large part of its block, so it inflates that spread and shrinks its own z-score. The same seizure would score lower in a block it dominates than in a block of quiet background.
2. An event that crosses a block border would mix two different scales.
3. µV is the unit researchers read in other EEG software and in the literature, so values can be checked and compared.

Unit conversion for (b): EDF data are read in volts (MNE converts using the header unit) and multiplied by 10^6. CSV data are multiplied by 1 (µV) or 10^3 (mV).

### 7.6 Event windows

After detection, the backend cuts out a window for each event, then deletes the uploaded file and the full signal.

| Item | Rule |
| --- | --- |
| Window | From 120 s before onset to 120 s after offset |
| Removed segments | Samples belonging only to removed segments (7.1, step 5) are not drawn. They are shown as a gap labelled "removed", in the same style as edge padding |
| Recording edges | If the event is less than 120 s from the start or end, the missing part is padded with blank space and shown greyed out, so every window has the same layout |
| Signal shown | Preprocessed signal (section 7.1, steps 1 to 3), unitless robust z-score. Blocks are the same 40-segment blocks used for detection; nothing is re-blocked for display. |
| Block overlap | Blocks are produced exactly as in `preprocessing_binary.py`: same grid, same 40-segment groups, same order. Neighbouring blocks share 1,250 samples (2.5 s); those samples are shown from the block the code processes first (the earlier block), which is the value in that block's last segment. Every displayed sample therefore equals a value the model received. |
| Seizure marking | The event selected in the event table is highlighted on the waveform |
| Waveform data | Downsampled for display with min/max decimation, so peaks are kept |
| Spectrogram | Same window as the waveform, preprocessed signal. A red rectangle outlines the seizure (onset to offset, full frequency range). |
| Spectrogram settings | As in `missed_seizure_audit.py`: mean removed, 1 s windows (NFFT = fs), 75% overlap. Frequency axis 0 Hz to the Nyquist frequency (fs / 2; 250 Hz at 500 Hz) |

Only these windows, the event table and the review decisions stay in memory.

## 8. Review workflow

### 8.1 States

Each event has one of three states: **Pending**, **Confirmed**, **Rejected**. All events start as Pending.

### 8.2 Rules

1. Selecting an event in the table opens it in the detail view.
2. The reviewer picks Confirm or Reject. The choice is a draft until "Save review" is pressed.
3. An unsaved draft shows the "Unsaved decision" marker.
4. Moving to another event, or leaving the page, with an unsaved draft opens a prompt with three choices: **Save**, **Discard**, **Stay**.
5. A saved decision can be changed at any time.
6. Arrow keys move through the table. Keyboard shortcuts are listed in section 10.4.

### 8.3 Where results live

There is no database.

| Item | Where | How long |
| --- | --- | --- |
| Uploaded file | Temporary folder on the server | Deleted as soon as detection ends, or at once if rejected |
| Event table, event windows, decisions | Server memory, keyed by analysis ID | Until the session expires (3 hours with no activity) or the server restarts |
| Long-term record | `events.csv` and the report zip, downloaded by the researcher | Kept by the researcher |

The interface tells the user, once, that results are not saved on the server and must be downloaded. If the user tries to close the page with unexported decisions, the browser asks them to confirm.

## 9. Outputs

### 9.1 events.csv

One row per event.

| Column | Example |
| --- | --- |
| analysis_id | AX-7F3C2A |
| event_id | EV-001 |
| onset_s | 1812.40 |
| offset_s | 1864.40 |
| duration_s | 52.00 |
| onset_elapsed | 00h30m12.4 |
| onset_clock | 2022-12-10 17:28:15.400 |
| peak_uV | 412 |
| kurtosis | 6.8 |
| skewness | 0.42 |
| envelope | 0.71 |
| probability (mean) | 0.97 |
| status | confirmed |

### 9.2 Summary report (.zip)

Built by the backend.

| File | Content |
| --- | --- |
| `events.csv` | As above |
| `summary.json` | Analysis ID, file name, channel, sampling rate, duration, pipeline version, model version, counts by status |
| `events/EV-001_waveform.png` | Event window (section 7.6) with the seizure highlighted |
| `events/EV-001_spectrogram.png` | 0 Hz to fs / 2 spectrogram of the same window, seizure outlined in red |
| `README.txt` | Column definitions and citation |

### 9.3 Versioning

Every export records the IctalX version, the pipeline version and a checksum of the model file, so a result can be traced to the exact method that produced it.

## 10. Interface specification

The page keeps the visual language of the prototype: navy hero, indigo accents, amber for seizure events, IBM Plex Sans, IBM Plex Mono and Bricolage Grotesque.

### 10.1 Page structure

| Order | Section | Content |
| --- | --- | --- |
| 1 | Top bar | Wordmark, links to sections 01 to 03, Contact |
| 2 | Hero | One-line purpose, short description, animated trace |
| 3 | 01 Upload and review | Recording card, summary card, review workspace |
| 4 | 02 Pipeline | Four steps, link to the paper |
| 5 | 03 Test data and offline use | Sample files, how to run locally |
| 6 | Cite | Citation, BibTeX, contact, licence |
| 7 | Footer | Research-use disclaimer |

The prototype's "Edit mode" toolbar is a design tool and is removed from the product.

### 10.2 Recording card

1. Subtitle: "Single channel of mouse EEG. Your file is processed and then deleted."
2. Format switch: EDF or CSV.
3. Drop zone with click-to-browse.
4. File chip with name, size and remove button.
5. Fields for the chosen format (section 6.2). For EDF, the channel list is filled once the file header is read.
6. "Detect seizures" button, disabled until a file and all required fields are present.
7. Validation errors appear under the file chip, in plain words, with the check that failed.

### 10.3 Progress

While the pipeline runs, the button area shows:

1. Upload progress, as a percentage.
2. The current stage: validating, preprocessing, inference, post-processing.
3. A progress bar for inference, based on segments done out of total.
4. A Cancel button.

### 10.4 Review workspace

Master list, then detail, stacked vertically as in the prototype.

**Event table:** number, onset, offset, duration, mean probability, status. Onset and offset show both elapsed time and clock time. Sortable by onset and probability. Filter by status. Scrolls inside its own panel.

**Detail view:**

1. Header with event number and status pill.
2. Descriptor grid (section 7.5).
3. Waveform: preprocessed signal (z-score) from 120 s before onset to 120 s after offset. The selected seizure is highlighted, with its start and end marked. Time labels at both ends of the window.
4. Spectrogram: same window, preprocessed signal, 0 Hz to fs / 2 (250 Hz at 500 Hz), with the seizure outlined by a red rectangle.
5. Review controls: Confirm, Reject, Save review, unsaved marker.

**Keyboard:**

| Key | Action |
| --- | --- |
| Up / Down | Previous / next event |
| C | Draft Confirm |
| R | Draft Reject |
| S or Enter | Save review |

### 10.5 Summary card

Recording length, number of events, total ictal time, average confidence, reviewed count, confirmed count, rejected count, analysis ID.

## 11. Data handling and security

| Topic | Stage 1, local | Stage 2, UCD |
| --- | --- | --- |
| Transport | HTTP on `localhost` | HTTPS / TLS only |
| Uploaded file | Deleted after detection | Deleted after detection |
| Rejected file | Deleted at once | Deleted at once |
| Results and decisions | Server memory, session only | Server memory, session only |
| Logs | No signal data or file contents in logs | Same; retention as per UCD policy |
| Third-party scripts | Fonts only; to be self-hosted | Same |

## 12. API

| Method | Path | Purpose | Request | Response |
| --- | --- | --- | --- | --- |
| GET | `/` | Serve the interface | | HTML |
| POST | `/api/detect` | Validate and run detection | Multipart: file + metadata | Analysis ID, events, descriptors, summary; or validation error with code |
| GET | `/api/events/{analysis_id}/{event_id}` | Waveform and spectrogram data for one event | | JSON |
| POST | `/api/review` | Save a decision | Analysis ID, event ID, status | Updated event |
| GET | `/api/report` | Download results | Analysis ID, format (`csv` or `zip`) | File |
| GET | `/api/samples` | List or fetch sample files | Optional file name | JSON list or file |

The `/api/events` route is new compared with the diagram. It lets the detail view load one event at a time instead of sending every window at once.

Error responses use the validation codes in section 6.3.

## 13. Non-functional requirements

| Area | Requirement |
| --- | --- |
| Accuracy | Same events as the training codebase on the reference recordings (section 14). |
| Performance | Measured on CPU: about 75 segments per second (19 minutes per recorded day; 87 minutes for m427). Detection must run as a background job with progress, not inside one web request. |
| Memory | Read only the chosen channel, block by block. Do not call MNE `get_data` on a long recording: it builds a time index for the whole file (3.65 GB for m427). Free the signal once event windows are cut. |
| Concurrent use (UCD) | Several users at once; analyses are queued if the server is busy. Limits **TBC** with UCD. |
| Browsers | Current Chrome, Edge, Firefox and Safari. |
| Accessibility | Keyboard use throughout, visible focus, WCAG 2.1 AA colour contrast, reduced-motion support. |

## 14. Testing

| Level | Test | Pass rule |
| --- | --- | --- |
| Pipeline | Run reference recordings through the app and through the training scripts | Identical event lists and probabilities |
| Validation | One test file per rejection code | Correct code and message |
| Deletion | Check the temporary folder after detection and after rejection | No uploaded file remains |
| Event windows | Events near the start, middle and end of a recording | Correct window bounds and highlighted span |
| Interface | Review flow, Save / Discard / Stay prompt, keyboard use, export contents | Manual checklist plus automated browser tests |
| Docker | Build the image and run the reference test inside it | Same results as local |

Reference recordings and their expected outputs come from the training codebase.

### 14.1 First parity run, m427 (2026-10-25)

| Item | Result |
| --- | --- |
| Setup | CPU only, Intel Core Ultra 9 185H, 16 threads, PyTorch 2.10 (CPU), checksum-verified copy of the weights, model frozen |
| Segments | 392,311 (272.4 h) |
| Time | 87.4 min in total: 5,176 s inference, 47 s preprocessing, 18 s reading. About 75 segments per second, roughly 19 minutes per recorded day |
| Reference events | All 12 reproduced exactly: onset, offset, duration, mean and max probability to 6 decimals |
| Extra events, first pass | 3, at 4,400 s, 530,205 s and 531,252.5 s, when the 61 extreme segments were clipped instead of removed |
| With the production filter | 61 segments removed, 392,250 kept, 4 chunks. 12 events, identical to the reference; difference in mean and max probability 0.0 |
| Weights after run | Unchanged (state and file checksums verified) |

## 15. Open questions

Answered questions are kept for the record; the answer is in the section named.

| No. | Question | Status |
| --- | --- | --- |
| Q1 | Required sampling rate; reject or resample other rates? | Answered: 500 Hz only during the pilot; other rates rejected (V05) |
| Q2 | Normalisation per recording or per segment? | Answered: per 40-segment block (7.1) |
| Q3 | Threshold, minimum duration and refractory gap? | Answered: 0.5, 25 s, 30 s; maximum 120 s (7.3) |
| Q4 | Event probability: mean or maximum? | Answered: mean (7.4) |
| Q5 | Waveform: raw or preprocessed signal? | Answered: preprocessed (7.6) |
| Q6 | Descriptor definitions? | Answered (7.5) |
| Q7 | What counts as an impossible value? | Answered: NaN, inf, or abs(x) > 1000 after preprocessing; the segment is removed (7.1) |
| Q8 | Session length? | Answered: 3 hours with no activity (8.3) |
| Q9 | Elapsed time, clock time, or both? | Answered: both (10.4) |
| Q10 | Licence for code and model? | Open |
| Q11 | Sample recordings for the focus group? | Open: owner to supply files |
| Q12 | UCD access: open, UCD network only, or sign-in? | Open |
| Q13 | AWS focus group: access method, budget, dates | Open |
| Q14 | Supporting sampling rates other than 500 Hz | Deferred until after the pilot |

## 16. Decision log

| Date | Decision |
| --- | --- |
| 2026-10-24 | Cloudflare dropped. Run locally first, then host at UCD with the same app. |
| 2026-10-24 | Multi-channel EDF files are accepted; the user picks one channel. |
| 2026-10-24 | Preprocessing order follows the architecture diagram: DC removal, notch filter, normalisation, segmentation. Superseded on 2026-10-25. |
| 2026-10-25 | Preprocessing order follows the revised architecture diagram: notch filter, DC removal, median / MAD normalisation, segmentation. |
| 2026-10-25 | `deploy_inference.py` is not a reference. The app follows the training code. |
| 2026-10-25 | Preprocessing runs on blocks of 40 segments. A flat block (MAD = 0) is left unchanged, as in training. |
| 2026-10-25 | The app analyses the channel the user selects; channel names are not assumed. |
| 2026-10-25 | Sampling rate and unit are read from the recording. |
| 2026-10-25 | NaN, infinite and extreme values are cleaned silently. Superseded the same day. |
| 2026-10-25 | EDF files without a recording start time are rejected. |
| 2026-10-25 | CSV upload stays; format to be supplied. |
| 2026-10-25 | The prototype's four descriptors stay; definitions to be supplied. |
| 2026-10-25 | Event probability is the mean. |
| 2026-10-25 | Post-processing order is minimum duration (25 s), then refractory merge (30 s), then maximum duration (120 s). |
| 2026-10-25 | The waveform shows the preprocessed signal. Superseded the same day. |
| 2026-10-25 | Times are shown as both elapsed time and clock time. |
| 2026-10-25 | Model is M3, MultiScaleTCN without attention. |
| 2026-10-25 | Sampling rate is taken from the file and used as it is. Superseded on 2026-10-26. |
| 2026-10-25 | Waveform shows the original signal; the selected event is highlighted. Superseded the same day. |
| 2026-10-25 | Spectrogram uses the waveform window, with a red rectangle around the seizure. Settings from `missed_seizure_audit.py`. |
| 2026-10-25 | Spectrogram frequency axis runs to the Nyquist frequency (fs / 2), not 70 Hz. |
| 2026-10-25 | Waveform and spectrogram both use the preprocessed signal, built from the same blocks as detection. Replaces the raw-signal decision. |
| 2026-10-25 | Display signal follows the preprocessing code's blocks and processing order; shared block samples come from the earlier block. |
| 2026-10-25 | Display details follow the codebase, not the prototype's sample values. |
| 2026-10-25 | File size limit raised to 2 GB. |
| 2026-10-25 | CPU is not enough. The app is hosted temporarily on AWS for the focus group, then moved to UCD. |
| 2026-10-25 | Extreme segments are removed as in the production code (`filter_extreme_segments`), and post-processing runs per contiguous chunk. |
| 2026-10-26 | AWS focus group: 20 users, owner's sample data, possibly concurrent, Ireland region. Ethics not in scope. |
| 2026-10-26 | CSV format taken from the prototype's CSV fields. |
| 2026-10-26 | Descriptor formulas approved: peak amplitude, kurtosis, skewness, envelope (section 7.5.2). |
| 2026-10-26 | Peak amplitude and envelope in µV, from the notch-filtered, mean-removed signal (option b). |
| 2026-10-26 | Training functions are copied into this repo with source file and commit `e463396` noted. |
| 2026-10-26 | Weights stay out of git and are checked by SHA-256 at start-up. |
| 2026-10-26 | Removed segments inside a display window are shown as a labelled gap. |
| 2026-10-26 | Session length 3 hours with no activity. |
| 2026-10-26 | Pilot accepts 500 Hz recordings only; other rates are rejected. Support for other rates to be considered later. |
| 2026-10-25 | Reference for testing: m427 rows of `test_event_details.csv` (MIN_EVENT_SEC_25s, min_then_refractory). |
| 2026-10-24 | No database at this stage. Results live in server memory for the session. |
| 2026-10-24 | Waveform and spectrogram come from the real signal, 120 s before onset to 120 s after offset, with the seizure highlighted. |
| 2026-10-24 | Windows near the recording edges are padded with blank space, shown greyed out. |
| 2026-10-24 | Unsaved decisions trigger a Save / Discard / Stay prompt. |
| 2026-10-24 | Edit mode toolbar removed from the product. |
| 2026-10-24 | Privacy line: "Your file is processed and then deleted." |

## 17. Glossary

| Term | Meaning |
| --- | --- |
| EDF | European Data Format, a standard file format for biosignals |
| TCN | Temporal Convolutional Network |
| FastAPI | Python framework for building web services |
| Docker | Tool that packages an app so it runs the same way on any server |
| MAD | Median absolute deviation |
| Ictal | Relating to a seizure |
| Refractory merging | Joining two events separated by a gap shorter than a set limit |
