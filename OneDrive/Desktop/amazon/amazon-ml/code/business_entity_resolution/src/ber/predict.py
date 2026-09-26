"""Test inference: features -> averaged fold models -> decision layer -> submission files.

Streams features per partition, averages the 5 fold models' probabilities, keeps pairs with p >= KEEP_P
(a small table) and applies decide.decide with the thresholds saved by decide.py. Writes
output/matching_results.tsv and output/candidate_pairs.tsv. The candidate file holds exactly the pairs the
model scored: every blocking candidate. Both files go through the streaming writer, then the official validator.

Usage (from src/):  python -m ber.predict --sample full --split test
"""
import json
import time

import numpy as np
import pandas as pd

from . import paths
from .cli import base_parser


def score(split: str, tag: str, verbose=True) -> pd.DataFrame:
    from .blocking import load_candidates
    from .features import FEATURES, FeatureContext, iter_features
    from .train import KEEP_P, load_models
    models, _ = load_models()
    ctx = FeatureContext(split)
    cands = load_candidates(tag)
    parts = []
    t0 = time.time()
    for _, df in iter_features(ctx, cands, verbose=verbose):
        X = df[FEATURES].to_numpy(np.float32)
        p = np.mean([m.predict(X) for m in models], axis=0).astype(np.float32)
        keep = p >= KEEP_P
        parts.append(pd.DataFrame({"s": df.s.to_numpy()[keep], "t": df.t.to_numpy()[keep], "p": p[keep]}))
    res = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"  scored {len(cands):,} pairs, kept {len(res):,} with p >= {KEEP_P} ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
    return res, cands[["s", "t"]]


def write_outputs(split: str, scored: pd.DataFrame, cands: pd.DataFrame, cfg: dict, out_dir) -> dict:
    from .data import _read_cached
    from .decide import decide
    from .submission import run_validator, write_from_pairs
    s1_ids = _read_cached(f"{split}_source1", columns=["entity_id"]).entity_id.to_numpy(dtype=object)
    t_ids = pd.concat([_read_cached(f"{split}_source{n}", columns=["entity_id"]).entity_id for n in (2, 3)],
                      ignore_index=True).to_numpy(dtype=object)
    s, t, p = scored.s.to_numpy(np.int64), scored.t.to_numpy(np.int64), scored.p.to_numpy()
    keep = decide(s, t, p, cfg["t_keep"], cfg["ratio"], cfg["t_null"], cfg["assign"], cfg.get("cap", 11))
    stats = {
        "matching": write_from_pairs(out_dir / "matching_results.tsv", s1_ids, s[keep], t_ids, t[keep], "matching"),
        "candidate": write_from_pairs(out_dir / "candidate_pairs.tsv", s1_ids, cands.s.to_numpy(), t_ids,
                                      cands.t.to_numpy(), "candidate"),
    }
    # predicted match-set size distribution by country (France sanity check)
    ctry = _read_cached(f"{split}_source1", columns=["country"]).country.to_numpy()
    size = np.bincount(s[keep], minlength=len(s1_ids))
    dist = pd.crosstab(pd.Series(ctry, name="country"), pd.Series(np.minimum(size, 8), name="pred_size"),
                       normalize="index").round(4)
    stats["size_dist"] = dist
    stats["mean_size"] = pd.Series(size).groupby(ctry).mean().round(3).to_dict()
    code, log = run_validator(out_dir, paths.DATA_DIR / split, paths.VALIDATOR, check_ids=False)
    stats["validator_exit"], stats["validator_log"] = code, log
    return stats


def main():
    p = base_parser(__doc__, split_default="test")
    a = p.parse_args()
    cfg = json.loads((paths.WORK_DIR / "models" / "decision.json").read_text())
    tag = f"{a.split}_{a.sample}"
    scored, cands = score(a.split, tag)
    scored.to_parquet(paths.CACHE_DIR / "train" / f"scored_{tag}.parquet", index=False)
    out_dir = paths.OUTPUT_DIR if (a.split, a.sample) == ("test", "full") else paths.WORK_DIR / f"out_{tag}"
    st = write_outputs(a.split, scored, cands, cfg, out_dir)
    rep = [f"# Prediction — {tag}\n", f"decision config: {cfg}\n", f"matching: {st['matching']}\n",
           f"candidate: {st['candidate']}\n", f"mean predicted match-set size by country: {st['mean_size']}\n",
           "## Predicted match-set size distribution (share of S1; 8 = 8+)\n", st["size_dist"].to_markdown(), "",
           f"validator exit code: {st['validator_exit']}\n```\n{st['validator_log']}\n```\n"]
    (paths.REPORTS_DIR / f"predict_{tag}.md").write_text("\n".join(rep))
    print("\n".join(rep))
    raise SystemExit(st["validator_exit"])


if __name__ == "__main__":
    main()
