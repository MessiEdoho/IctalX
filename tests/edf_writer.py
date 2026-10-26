"""Minimal single-file EDF writer for tests (EDF spec: 256-byte header + 256 bytes per signal + int16 records)."""
import numpy as np


def _field(value, width):
    text = str(value)[:width]
    return text.ljust(width).encode("ascii")


def write_edf(path, signals, fs, start="10.12.22", start_time="16.58.03",
              unit="uV", phys_range=(-5000.0, 5000.0)):
    """signals: dict label -> 1-D array in `unit`. All signals share fs; one record per second."""
    labels = list(signals)
    n_sig = len(labels)
    spr = int(fs)                                        # samples per record (1 s records)
    n_rec = len(next(iter(signals.values()))) // spr
    pmin, pmax = phys_range
    dmin, dmax = -32768, 32767

    header = b"".join([
        _field("0", 8), _field("test", 80), _field("test", 80),
        _field(start, 8), _field(start_time, 8),
        _field(256 + 256 * n_sig, 8), _field("", 44),
        _field(n_rec, 8), _field(1, 8), _field(n_sig, 4),
    ])
    for width, values in [
        (16, labels), (80, [""] * n_sig), (8, [unit] * n_sig),
        (8, [pmin] * n_sig), (8, [pmax] * n_sig), (8, [dmin] * n_sig), (8, [dmax] * n_sig),
        (80, [""] * n_sig), (8, [spr] * n_sig), (32, [""] * n_sig),
    ]:
        header += b"".join(_field(v, width) for v in values)

    digital = {}
    for lab in labels:
        x = np.asarray(signals[lab][:n_rec * spr], dtype=np.float64)
        d = np.round((x - pmin) / (pmax - pmin) * (dmax - dmin) + dmin)
        digital[lab] = np.clip(d, dmin, dmax).astype("<i2").reshape(n_rec, spr)

    with open(path, "wb") as f:
        f.write(header)
        for r in range(n_rec):
            for lab in labels:
                f.write(digital[lab][r].tobytes())
