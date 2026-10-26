"""Upload validation (docs/product-spec.md, section 6.3).

Each check raises ValidationError with the spec's code and a plain message.
A file that passes is returned as a signal source ready for detection.
"""
from datetime import date, datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ictalx import pipeline
from ictalx.training import preprocessing_binary as pp

MAX_BYTES = 2 * 1024 ** 3                     # V07: 2 GB
UNIT_TO_UV = {"µV": 1.0, "uV": 1.0, "mV": 1e3}
PLACEHOLDER_DATE = date(1985, 1, 1)           # EDF convention for an unknown start date


class ValidationError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def _check_size(path):
    if Path(path).stat().st_size > MAX_BYTES:
        raise ValidationError("V07", "Files larger than 2 GB are not supported.")


def _check_length_and_rate(n_samples, fs):
    if n_samples == 0:
        raise ValidationError("V03", "The file contains no signal data.")
    if fs != pp.fs:
        raise ValidationError("V05", f"Sampling rate is {fs:g} Hz. IctalX currently accepts "
                                     f"{pp.fs} Hz recordings only.")
    if n_samples < pp.win_len:
        raise ValidationError("V08", "The recording is shorter than 5 s.")


def edf_channels(path):
    """Channel labels of an EDF file (header only)."""
    try:
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    except Exception:
        raise ValidationError("V02", "The file appears to be damaged and could not be read.")
    return list(raw.ch_names)


def validate_edf(path, channel):
    path = Path(path)
    if path.suffix.lower() != ".edf":
        raise ValidationError("V01", "This file is not a valid EDF or CSV file.")
    _check_size(path)
    if channel not in edf_channels(path):
        raise ValidationError("V04", "The selected channel was not found in this file.")
    try:
        source = pipeline.EdfSource(path, channel)
    except Exception:
        raise ValidationError("V02", "The file appears to be damaged and could not be read.")
    # A blank date reads as None. 01.01.85 is the EDF placeholder for an unknown or
    # anonymised date; it is treated as missing, whatever the time field says.
    if source.start is None or source.start.date() == PLACEHOLDER_DATE:
        raise ValidationError("V09", "This EDF file has no recording start time.")
    _check_length_and_rate(source.n_samples, source.fs)
    return source


def validate_csv(path, signal_column, sampling_rate, unit, time_column=None, start_time=None):
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValidationError("V01", "This file is not a valid EDF or CSV file.")
    _check_size(path)
    if unit not in UNIT_TO_UV:
        raise ValidationError("V12", "Unit must be µV or mV.")
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        raise ValidationError("V02", "The file appears to be damaged and could not be read.")
    wanted = [signal_column] + ([time_column] if time_column else [])
    if any(c not in header for c in wanted):
        raise ValidationError("V04", "The selected column was not found in this file.")
    try:
        df = pd.read_csv(path, usecols=wanted)
        signal = pd.to_numeric(df[signal_column], errors="raise").to_numpy(dtype=np.float64)
    except Exception:
        raise ValidationError("V02", "The file appears to be damaged and could not be read.")

    try:
        fs = float(sampling_rate)
    except (TypeError, ValueError):
        raise ValidationError("V05", "Enter the sampling rate in Hz.")
    _check_length_and_rate(len(signal), fs)

    if time_column:
        try:
            t = pd.to_numeric(df[time_column], errors="raise").to_numpy(dtype=np.float64)
        except Exception:
            raise ValidationError("V10", "The time column must contain numbers in seconds.")
        dt = np.diff(t)
        if (dt <= 0).any() or abs(float(np.median(dt)) * fs - 1.0) > 0.01:
            raise ValidationError("V10", "The time column must increase steadily and match the sampling rate.")

    start = None
    if start_time:
        try:
            start = datetime.fromisoformat(str(start_time).strip())
        except ValueError:
            raise ValidationError("V11", "Start time not understood. Use YYYY-MM-DD hh:mm:ss.")
    return pipeline.ArraySource(signal, fs, start=start, unit_to_uv=UNIT_TO_UV[unit], channel=signal_column)
