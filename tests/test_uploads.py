"""Start-up clean-up of leftover uploads (spec section 8.3)."""
import uuid

from ictalx import jobs


def test_leftover_uploads_removed_other_files_kept(tmp_path):
    leftovers = [f"{uuid.uuid4().hex}.edf", f"{uuid.uuid4().hex}.csv", uuid.uuid4().hex]
    keep = ["notes.txt", ".hidden", f"{uuid.uuid4().hex[:31]}.edf", f"{uuid.uuid4().hex.upper()}.edf"]
    for name in leftovers + keep:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / uuid.uuid4().hex).mkdir()                 # a folder with an upload-like name is kept

    removed = jobs.remove_leftover_uploads(tmp_path)

    assert sorted(removed) == sorted(leftovers)
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == sorted(keep)
    assert sum(p.is_dir() for p in tmp_path.iterdir()) == 1
