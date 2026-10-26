"""Validation rules V01-V12 on small synthetic files."""
import numpy as np
import pytest

from edf_writer import write_edf
from ictalx import validation
from ictalx.validation import ValidationError


def noise(seconds, fs=500, seed=0):
    return np.random.default_rng(seed).normal(0, 50, int(seconds * fs))


def code_of(call):
    with pytest.raises(ValidationError) as exc:
        call()
    return exc.value.code


def test_valid_edf(tmp_path):
    p = tmp_path / "ok.edf"
    write_edf(p, {"EEG": noise(60), "EMG": noise(60, seed=1)}, 500)
    source = validation.validate_edf(p, "EEG")
    assert source.fs == 500 and source.n_samples == 30000 and source.start is not None


def test_v01_wrong_extension(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    assert code_of(lambda: validation.validate_edf(p, "EEG")) == "V01"


def test_v02_damaged(tmp_path):
    p = tmp_path / "bad.edf"
    p.write_bytes(b"not an edf file" * 10)
    assert code_of(lambda: validation.validate_edf(p, "EEG")) == "V02"


def test_v04_channel_missing(tmp_path):
    p = tmp_path / "ok.edf"
    write_edf(p, {"EEG": noise(60)}, 500)
    assert code_of(lambda: validation.validate_edf(p, "EEG2")) == "V04"


def test_v05_wrong_rate(tmp_path):
    p = tmp_path / "fast.edf"
    write_edf(p, {"EEG": noise(60, fs=1000)}, 1000)
    assert code_of(lambda: validation.validate_edf(p, "EEG")) == "V05"


def test_v08_too_short(tmp_path):
    p = tmp_path / "short.edf"
    write_edf(p, {"EEG": noise(3)}, 500)
    assert code_of(lambda: validation.validate_edf(p, "EEG")) == "V08"


def test_valid_csv_and_units(tmp_path):
    p = tmp_path / "ok.csv"
    x = noise(30)
    t = np.arange(len(x)) / 500
    p.write_text("t,eeg\n" + "\n".join(f"{a:.4f},{b:.3f}" for a, b in zip(t, x)))
    src = validation.validate_csv(p, "eeg", "500", "mV", "t", "2022-12-10 16:58:03")
    assert src.n_samples == len(x) and src.unit_to_uv == 1000.0 and src.start.year == 2022


def test_csv_codes(tmp_path):
    p = tmp_path / "ok.csv"
    x = noise(30)
    p.write_text("eeg\n" + "\n".join(f"{b:.3f}" for b in x))
    assert code_of(lambda: validation.validate_csv(p, "signal", "500", "µV")) == "V04"
    assert code_of(lambda: validation.validate_csv(p, "eeg", "250", "µV")) == "V05"
    assert code_of(lambda: validation.validate_csv(p, "eeg", "500", "V")) == "V12"
    assert code_of(lambda: validation.validate_csv(p, "eeg", "500", "µV", start_time="yesterday")) == "V11"


def test_v10_time_column_mismatch(tmp_path):
    p = tmp_path / "t.csv"
    x = noise(30)
    t = np.arange(len(x)) / 250                          # says 250 Hz, user enters 500
    p.write_text("t,eeg\n" + "\n".join(f"{a:.4f},{b:.3f}" for a, b in zip(t, x)))
    assert code_of(lambda: validation.validate_csv(p, "eeg", "500", "µV", "t")) == "V10"


def test_v09_no_start_time(tmp_path):
    p = tmp_path / "nodate.edf"
    write_edf(p, {"EEG": noise(60)}, 500, start="", start_time="")
    assert code_of(lambda: validation.validate_edf(p, "EEG")) == "V09"


def test_v09_placeholder_date(tmp_path):
    p = tmp_path / "anon.edf"
    write_edf(p, {"EEG": noise(60)}, 500, start="01.01.85", start_time="10.00.00")
    assert code_of(lambda: validation.validate_edf(p, "EEG")) == "V09"
