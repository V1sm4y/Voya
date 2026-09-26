from pathlib import Path

import pytest

from ber import paths
from ber.submission import SubmissionError, run_validator, write_id_lists, write_submission

S1 = ["S1-1", "S1-2", "S1-3"]


def _write_test_dir(d: Path):
    d.mkdir()
    hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    (d / "test_source1.tsv").write_text(hdr + "".join(f"{i}\tX\tY\tUS\n" for i in S1))
    (d / "test_source2.tsv").write_text(hdr + "S2-10\tX\tY\tUS\nS2-11\tX\tY\tUS\n")
    (d / "test_source3.tsv").write_text(hdr + "S3-20\tX\tY\tUS\n")


def test_exact_bytes(tmp_path):
    p = tmp_path / "m.tsv"
    stats = write_id_lists(p, {"S1-2": ["S2-10", "S3-20", "S2-10"]}, S1)
    assert p.read_bytes() == b"source1_entity_id\tmatched_entity_ids\nS1-1\t\nS1-2\tS2-10,S3-20\nS1-3\t\n"
    assert stats == {"path": str(p), "rows": 3, "empty": 2, "links": 2}


def test_candidate_header(tmp_path):
    p = tmp_path / "c.tsv"
    write_id_lists(p, {}, S1, kind="candidate")
    assert p.read_text().splitlines()[0] == "source1_entity_id\tcandidate_entity_ids"


@pytest.mark.parametrize("bad", ["S1-9", "S2-1 ", " S2-1", "S2-1,S3-2", "X-1", "", None])
def test_rejects_bad_ids(tmp_path, bad):
    with pytest.raises(SubmissionError):
        write_id_lists(tmp_path / "m.tsv", {"S1-1": [bad]}, S1)


def test_rejects_unknown_s1(tmp_path):
    with pytest.raises(SubmissionError):
        write_id_lists(tmp_path / "m.tsv", {"S1-999": ["S2-1"]}, S1)


def test_rejects_duplicate_required(tmp_path):
    with pytest.raises(SubmissionError):
        write_id_lists(tmp_path / "m.tsv", {}, ["S1-1", "S1-1"])


def test_matches_must_be_candidates(tmp_path):
    with pytest.raises(SubmissionError):
        write_submission(tmp_path, {"S1-1": ["S2-10"]}, {"S1-1": ["S2-11"]}, S1)


@pytest.mark.skipif(not paths.VALIDATOR.exists(), reason="organisers' validator not available")
def test_official_validator_passes(tmp_path):
    test_dir = tmp_path / "test"
    _write_test_dir(test_dir)
    out = tmp_path / "out"
    write_submission(out, {"S1-1": ["S2-10", "S3-20"], "S1-3": ["S2-11"]},
                     {"S1-1": ["S2-10", "S3-20", "S2-11"], "S1-2": ["S2-11"], "S1-3": ["S2-11"]}, S1)
    code, log = run_validator(out, test_dir, paths.VALIDATOR, check_ids=True)
    assert code == 0, log
    assert "PASS" in log and "WARNING: 1" not in log


def test_write_from_pairs_equals_write_id_lists(tmp_path):
    import numpy as np
    from ber.submission import write_from_pairs
    s1 = np.array(["S1-1", "S1-2", "S1-3", "S1-4"], dtype=object)
    t_ids = np.array(["S2-10", "S3-20", "S2-11"], dtype=object)
    s_idx = np.array([2, 0, 0, 0, 2])
    t_idx = np.array([2, 1, 0, 1, 2])            # includes a duplicate pair
    write_from_pairs(tmp_path / "a.tsv", s1, s_idx, t_ids, t_idx, chunk=2)
    write_id_lists(tmp_path / "b.tsv", {"S1-1": ["S2-10", "S3-20"], "S1-3": ["S2-11"]}, list(s1))
    assert (tmp_path / "a.tsv").read_bytes() == (tmp_path / "b.tsv").read_bytes()
