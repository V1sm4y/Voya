"""Offline tuning of blocking K / blocker subsets from one `blocking --sweep` run.

Reads data/cache/candidates/raw_<tag>.parquet (every generated pair with its blocker bits and per-blocker
rank) and truth_<tag>.parquet. Any configuration with K <= the sweep K and any subset of the blockers can
then be scored exactly, in about 0.2 s each (probe S1s excluded). Writes reports/blocking_tune_<tag>.md with
the recall-vs-candidates Pareto front and the ablation of each blocker at the chosen point.

Usage (from src/):  python -m ber.blocking_tune --sample dev
"""
import itertools

import numpy as np
import pandas as pd

from . import paths
from .blocking import BIT, BLOCKERS, RANKED
from .cli import base_parser


def _load(tag):
    d = paths.CACHE_DIR / "candidates"
    raw = pd.read_parquet(d / f"raw_{tag}.parquet")
    raw = raw[~raw.probe].reset_index(drop=True)
    tp = pd.read_parquet(d / f"truth_{tag}.parquet")
    tp = tp[~tp.probe & (tp.t >= 0)].reset_index(drop=True)
    n_total = int((pd.read_parquet(d / f"truth_{tag}.parquet").probe == False).sum())  # noqa: E712
    return raw, tp, n_total


def _attach(raw, tp):
    """Blocker bits + ranks of each true pair (0 / 255 when never generated)."""
    rk = raw.s.to_numpy(np.int64) << 32 | raw.t.to_numpy(np.int64)
    tk = tp.s.to_numpy(np.int64) << 32 | tp.t.to_numpy(np.int64)
    pos = np.clip(np.searchsorted(rk, tk), 0, len(rk) - 1)
    hit = rk[pos] == tk
    out = {"bits": np.where(hit, raw.bits.to_numpy()[pos], 0).astype(np.uint16)}
    for b in RANKED:
        out[b] = np.where(hit, raw[f"rank_{b}"].to_numpy()[pos], 255).astype(np.uint8)
    return out


def _mask(bits, ranks, k: dict, unranked: list):
    m = (bits & np.uint16(sum(int(BIT[b]) for b in unranked))) > 0
    for b in RANKED:
        if k.get(b, 0) > 0:
            m |= ranks[b] < k[b]
    return m


def evaluate(raw_cols, tp_cols, tp_country, n_true, n_s1, k, unranked):
    keep_c = _mask(raw_cols["bits"], raw_cols, k, unranked)
    keep_t = _mask(tp_cols["bits"], tp_cols, k, unranked)
    r = {"cands_per_s1": keep_c.sum() / n_s1, "recall": keep_t.sum() / n_true}
    for c in ("India", "US"):
        m = tp_country == c
        r[f"recall {c}"] = keep_t[m].mean() if m.any() else np.nan
    return r


def main():
    p = base_parser(__doc__)
    p.add_argument("--target", type=float, default=0.98)
    a = p.parse_args()
    tag = f"{a.split}_{a.sample}"
    raw, tp, n_true = _load(tag)
    n_s1 = raw.s.nunique()
    raw_cols = {"bits": raw.bits.to_numpy()}
    raw_cols.update({b: raw[f"rank_{b}"].to_numpy() for b in RANKED})
    tp_cols = _attach(raw, tp)
    ctry = tp.country.to_numpy()
    unranked_all = [b for b in BLOCKERS if b not in RANKED]
    grid = {"name_tfidf": [5, 10, 15, 20, 30], "addr_tfidf": [5, 10, 15, 20, 30],
            "name_phon_tfidf": [0, 3, 5, 10], "placeless_name": [0, 2, 5]}
    rows = []
    for combo in itertools.product(*grid.values()):
        k = dict(zip(grid, combo))
        rows.append({**k, **evaluate(raw_cols, tp_cols, ctry, len(tp), n_s1, k, unranked_all)})
    res = pd.DataFrame(rows).sort_values("cands_per_s1")
    res["pareto"] = res.recall >= res.recall.cummax()
    front = res[res.pareto].drop(columns="pareto")
    # recall is measured over pairs whose target is in the pool; pool coverage is 100% on DEV
    best = front[front.recall >= a.target].head(1)
    best_in = front[(front.recall >= a.target) & (front["recall India"] >= a.target)].head(1)
    out = [f"# Blocking tuning — {tag}\n", f"true pairs {len(tp):,} | S1 {n_s1:,}\n",
           "## Pareto front (all unranked blockers on)\n", front.round(4).to_markdown(index=False), ""]
    for label, row in (("cheapest config with overall recall >= target", best),
                       ("cheapest config with overall AND India recall >= target", best_in)):
        if row.empty:
            out.append(f"**{label}: none in grid**\n")
            continue
        k = {b: int(row.iloc[0][b]) for b in grid}
        out.append(f"## {label}: {k}\n")
        base = evaluate(raw_cols, tp_cols, ctry, len(tp), n_s1, k, unranked_all)
        abl = [{"dropped": "(none)", **base}]
        for b in unranked_all:
            abl.append({"dropped": b, **evaluate(raw_cols, tp_cols, ctry, len(tp), n_s1, k,
                                                 [u for u in unranked_all if u != b])})
        abl = pd.DataFrame(abl).set_index("dropped")
        abl["d_recall"] = abl.recall - base["recall"]
        abl["d_cands"] = abl.cands_per_s1 - base["cands_per_s1"]
        out += ["ablation (drop one unranked blocker):\n", abl.round(4).to_markdown(), ""]
    txt = "\n".join(out)
    print(txt)
    (paths.REPORTS_DIR / f"blocking_tune_{tag}.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
