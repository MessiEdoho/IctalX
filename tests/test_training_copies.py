"""The files in ictalx/training/ must match DL_WITH_SSL_GA at commit e463396, block by block."""
import os
import re
import subprocess
from pathlib import Path

import pytest

COMMIT = "e463396"
TRAINING_DIR = Path(__file__).resolve().parents[1] / "ictalx" / "training"
SOURCE_REPO = Path(os.environ.get(
    "ICTALX_TRAINING_REPO",
    r"C:\Users\messi\OneDrive\Documents\UCD PhD\UNIQURE\DL_WITH_SSL_GA"))
MARKER = re.compile(r"^# (\S+\.py) lines (\d+)-(\d+)\n", re.MULTILINE)


def copied_blocks(path):
    text = path.read_text(encoding="utf-8")
    marks = list(MARKER.finditer(text))
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        yield m.group(1), int(m.group(2)), int(m.group(3)), text[m.end():end]


@pytest.mark.skipif(not (SOURCE_REPO / ".git").exists(), reason="training repository not available")
@pytest.mark.parametrize("name", ["preprocessing_binary.py", "eval_utils.py", "tcn_utils.py"])
def test_copies_match_commit(name):
    source = subprocess.run(["git", "-C", str(SOURCE_REPO), "show", f"{COMMIT}:{name}"],
                            capture_output=True, check=True).stdout.decode("utf-8").splitlines()
    blocks = list(copied_blocks(TRAINING_DIR / name))
    assert blocks, f"no copied blocks found in {name}"
    for src_name, a, b, copied in blocks:
        assert src_name == name
        expected = "\n".join(source[a - 1:b]).rstrip()
        assert copied.rstrip() == expected, f"{name} lines {a}-{b} differ from {COMMIT}"
