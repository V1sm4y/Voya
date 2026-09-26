"""Format-safe writer for matching_results.tsv and candidate_pairs.tsv (README "Output Format").

Guarantees:
- the header is exactly `source1_entity_id<TAB>matched_entity_ids`, or `...<TAB>candidate_entity_ids`;
- there is one row per required S1 id, in the given order;
- S1 ids with no prediction get an empty list, and unknown S1 ids raise;
- ids are joined with "," with no spaces, contain only `S2-`/`S3-` ids, are de-duplicated with the
  first occurrence kept, and the file is written as UTF-8 with "\n" line endings;
- when both files are written, every matched id must also be a candidate (raises otherwise).
"""
import re
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

HEADERS = {
    "matching": ("source1_entity_id", "matched_entity_ids"),
    "candidate": ("source1_entity_id", "candidate_entity_ids"),
}
S1_RE = re.compile(r"S1-[0-9A-Za-z_]+")
TARGET_RE = re.compile(r"S[23]-[0-9A-Za-z_]+")


class SubmissionError(ValueError):
    pass


def _clean_ids(s1: str, ids: Iterable[str]) -> list:
    out, seen = [], set()
    for i in ids:
        if not isinstance(i, str) or not TARGET_RE.fullmatch(i):
            raise SubmissionError(f"{s1}: invalid target id {i!r} (must be S2-/S3- id, no spaces/commas)")
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def write_id_lists(path, mapping: Mapping[str, Iterable[str]], required_s1: Sequence[str],
                   kind: str = "matching") -> dict:
    """Write one submission-style TSV. Returns stats. `required_s1` = all ids of test_source1, in order."""
    if kind not in HEADERS:
        raise ValueError(kind)
    required = list(required_s1)
    req_set = set(required)
    if len(req_set) != len(required):
        raise SubmissionError("required_s1 contains duplicates")
    bad = [i for i in required if not S1_RE.fullmatch(i)]
    if bad:
        raise SubmissionError(f"required_s1 has non-S1 ids, e.g. {bad[:3]}")
    unknown = set(mapping) - req_set
    if unknown:
        raise SubmissionError(f"{len(unknown)} S1 ids not in required set, e.g. {sorted(unknown)[:3]}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_empty = n_links = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(HEADERS[kind]) + "\n")
        for s1 in required:
            ids = _clean_ids(s1, mapping.get(s1, ()))
            n_links += len(ids)
            n_empty += not ids
            f.write(f"{s1}\t{','.join(ids)}\n")
    stats = {"path": str(path), "rows": len(required), "empty": n_empty, "links": n_links}
    _self_check(path, kind, len(required))
    return stats


def _self_check(path: Path, kind: str, n_rows: int) -> None:
    raw = path.read_bytes()
    if b"\r" in raw:
        raise SubmissionError("CR found in output")
    lines = raw.decode("utf-8").split("\n")
    if lines[0] != "\t".join(HEADERS[kind]) or lines[-1] != "" or len(lines) != n_rows + 2:
        raise SubmissionError(f"self-check failed for {path}")


def write_submission(out_dir, matches: Mapping[str, Iterable[str]], candidates: Mapping[str, Iterable[str]],
                     required_s1: Sequence[str]) -> dict:
    """Write both output files; matched ids must be a subset of that S1's candidates."""
    for s1, ids in matches.items():
        extra = set(ids) - set(candidates.get(s1, ()))
        if extra:
            raise SubmissionError(f"{s1}: matched ids not in candidates: {sorted(extra)[:3]}")
    out_dir = Path(out_dir)
    return {
        "matching": write_id_lists(out_dir / "matching_results.tsv", matches, required_s1, "matching"),
        "candidate": write_id_lists(out_dir / "candidate_pairs.tsv", candidates, required_s1, "candidate"),
    }


def run_validator(out_dir, test_dir, validator, check_ids: bool = False, python: Optional[str] = None):
    """Run the organisers' validate_submission.py; returns (exit_code, output)."""
    import subprocess
    import sys
    out_dir = Path(out_dir)
    cmd = [python or sys.executable, str(validator), "--matching", str(out_dir / "matching_results.tsv"),
           "--candidate", str(out_dir / "candidate_pairs.tsv"), "--test-dir", str(test_dir)]
    if check_ids:
        cmd.append("--check-ids")
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def write_from_pairs(path, s1_ids: "np.ndarray", s_idx: "np.ndarray", t_ids: "np.ndarray", t_idx: "np.ndarray",
                     kind: str = "matching", chunk: int = 200_000) -> dict:
    """Streaming writer for large outputs: one row per entry of `s1_ids` (the required S1 ids, in order),
    listing t_ids[t_idx] for every pair with s_idx == row. Pairs may come in any order; duplicates are
    dropped. Writes the same exact format as write_id_lists (UTF-8, '\\n', ',' joins with no spaces) without
    ever holding all id strings in memory."""
    import numpy as np
    import pandas as pd
    if kind not in HEADERS:
        raise ValueError(kind)
    s1_ids = np.asarray(s1_ids, dtype=object)
    if len(set(s1_ids)) != len(s1_ids) or not all(S1_RE.fullmatch(x) for x in s1_ids[:1000]):
        raise SubmissionError("s1_ids must be unique S1 ids")
    key = np.unique(np.asarray(s_idx, np.int64) << 32 | np.asarray(t_idx, np.int64))
    s_sorted, t_sorted = (key >> 32).astype(np.int64), (key & 0xFFFFFFFF).astype(np.int64)
    bounds = np.searchsorted(s_sorted, np.arange(0, len(s1_ids) + chunk, chunk))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_links = n_empty = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(HEADERS[kind]) + "\n")
        for c, start in enumerate(range(0, len(s1_ids), chunk)):
            lo, hi = bounds[c], bounds[c + 1]
            ids = pd.Series(np.asarray(t_ids, dtype=object)[t_sorted[lo:hi]])
            bad = ~ids.str.match(r"^S[23]-[0-9A-Za-z_]+$")
            if bad.any():
                raise SubmissionError(f"invalid target ids, e.g. {ids[bad].head(3).tolist()}")
            joined = ids.groupby(s_sorted[lo:hi]).agg(",".join)
            rows = pd.Series("", index=np.arange(start, min(start + chunk, len(s1_ids))))
            rows.loc[joined.index] = joined.values
            n_links += hi - lo
            n_empty += int((rows == "").sum())
            f.write("".join(f"{a}\t{b}\n" for a, b in zip(s1_ids[rows.index], rows.values)))
    _self_check(path, kind, len(s1_ids))
    return {"path": str(path), "rows": len(s1_ids), "empty": n_empty, "links": int(n_links)}
