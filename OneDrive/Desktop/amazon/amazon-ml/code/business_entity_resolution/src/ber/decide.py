"""Decision layer: pair probabilities -> per-S1 match lists.

  1. assignment (optional): each S2/S3 record goes to at most one S1, the one with the highest probability
     (ties: lowest S1 index). This is the per-record argmax, which equals greedy global assignment
     because S1s have no capacity limit before the cap below.
  2. per S1, over the records it still holds: keep p >= t_keep AND p >= ratio * best_p; at most CAP (11),
     highest p first; several per source are allowed.
  3. if the S1's best p (after assignment) < t_null, predict an empty list.

`fast_macro_f05` is an exact, vectorised re-implementation of metrics.f05 averaged over S1s. It is used to
grid-search on OOF and is checked against metrics.report on the chosen setting.

Usage (from src/):  python -m ber.decide --sample full
"""
import itertools
import json

import numpy as np
import pandas as pd

from . import paths
from .cli import base_parser

CAP = 11


def assignment_mask(s: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """True for the pair that owns each target: highest p, ties to the lowest S1 index."""
    n = len(s)
    o = np.lexsort((s, -p, t))
    first = np.ones(n, bool)
    first[1:] = t[o][1:] != t[o][:-1]
    alive = np.zeros(n, bool)
    alive[o[first]] = True
    return alive


def decide(s: np.ndarray, t: np.ndarray, p: np.ndarray, t_keep: float, ratio: float, t_null: float,
           assign: bool = True, cap: int = CAP, alive: np.ndarray = None) -> np.ndarray:
    """Boolean mask over pairs: which (s, t) are predicted matches. Pass a precomputed `alive`
    (assignment_mask) to avoid re-sorting inside grid searches."""
    n = len(s)
    if n == 0:
        return np.zeros(0, bool)
    if alive is None:
        alive = assignment_mask(s, t, p) if assign else np.ones(n, bool)
    elif not assign:
        alive = np.ones(n, bool)
    # best p per S1 among alive pairs
    best = np.zeros(int(s.max()) + 1, np.float32)
    np.maximum.at(best, s[alive], p[alive])
    b = best[s]
    keep = alive & (p >= t_keep) & (p >= ratio * b) & (b >= t_null)
    # cap per S1
    if cap and keep.any():
        idx = np.flatnonzero(keep)
        o = idx[np.lexsort((-p[idx], s[idx]))]
        ss = s[o]
        start = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
        rank = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)]))
        keep[o[rank >= cap]] = False
    return keep


def fast_macro_f05(n_s1: int, n_true: np.ndarray, s: np.ndarray, y: np.ndarray, pred: np.ndarray,
                   s_subset: np.ndarray = None) -> float:
    """Exact macro F0.5 over S1 rows 0..n_s1-1 (or s_subset). n_true[i] = size of i's true match set.
    (s, y, pred) cover the scored pairs; true pairs not scored count as missed (they are in n_true)."""
    npred = np.bincount(s[pred], minlength=n_s1).astype(np.float64)
    tp = np.bincount(s[pred & (y == 1)], minlength=n_s1).astype(np.float64)
    nt = n_true.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0.0)
        rec = np.where(nt > 0, tp / nt, 0.0)
        f = np.where(tp > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
    f = np.where(nt == 0, (npred == 0).astype(np.float64), f)
    return float(f.mean() if s_subset is None else f[s_subset].mean())


def per_s1_f05(n_s1, n_true, s, y, pred) -> np.ndarray:
    npred = np.bincount(s[pred], minlength=n_s1).astype(np.float64)
    tp = np.bincount(s[pred & (y == 1)], minlength=n_s1).astype(np.float64)
    nt = n_true.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0.0)
        rec = np.where(nt > 0, tp / nt, 0.0)
        f = np.where(tp > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
    return np.where(nt == 0, (npred == 0).astype(np.float64), f)


class FastDecider:
    """Grid-search helper, equal to decide() but O(n) per setting after one O(n log n) setup.

    Inside an S1, kept pairs are always its top alive pairs by p (every rule is a threshold on p), so the
    (s, -p) order, the per-S1 best alive p and each alive pair's rank among alive pairs are computed once
    per assign flag."""

    def __init__(self, s, t, p, cap=CAP):
        self.s, self.p, self.cap = s, p, cap
        self.n_s = int(s.max()) + 1 if len(s) else 0
        order = np.lexsort((-p, s))
        self.prep = {}
        for asg in (True, False):
            alive = assignment_mask(s, t, p) if asg else np.ones(len(s), bool)
            so, ao = s[order], alive[order]
            start = np.r_[0, np.flatnonzero(np.diff(so)) + 1] if len(so) else np.array([], int)
            cum = np.cumsum(ao)
            base = np.repeat(np.r_[0, cum[start[1:] - 1]] if len(start) else [], np.diff(np.r_[start, len(so)]))
            arank = np.empty(len(s), np.int64)
            arank[order] = cum - base - 1            # rank among alive pairs of the same S1 (valid where alive)
            pa = np.where(alive, p, -1.0)
            best = np.full(self.n_s, -1.0)
            if len(so):
                best[so[start]] = np.maximum.reduceat(pa[order], start)
            self.prep[asg] = (alive, arank, best[s])

    def __call__(self, t_keep, ratio, t_null, assign=True):
        alive, arank, b = self.prep[bool(assign)]
        p = self.p
        return alive & (p >= t_keep) & (p >= ratio * b) & (b >= t_null) & (arank < self.cap)


GRID = {"t_keep": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        "ratio": [0.0, 0.2, 0.4, 0.6, 0.8],
        "t_null": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        "assign": [True, False]}


def grid_search(n_s1, n_true, s, t, p, y, s_mask=None) -> pd.DataFrame:
    rows = []
    fd = FastDecider(s, t, p)
    for tk, r, tn, asg in itertools.product(*GRID.values()):
        if tn < tk:
            continue
        pred = fd(tk, r, tn, asg)
        rows.append({"t_keep": tk, "ratio": r, "t_null": tn, "assign": asg,
                     "f05": fast_macro_f05(n_s1, n_true, s, y, pred, s_mask)})
    return pd.DataFrame(rows)


def stable_choice(g: pd.DataFrame) -> pd.Series:
    """Best setting by the mean F0.5 of itself and its grid neighbours (one step in any threshold, same
    assign flag). A point in a flat region beats an isolated peak."""
    keys = ["t_keep", "ratio", "t_null"]
    pos = {k: {v: i for i, v in enumerate(GRID[k])} for k in keys}
    g = g.copy()
    for k in keys:
        g[f"_{k}"] = g[k].map(pos[k])
    lookup = {(r["assign"],) + tuple(r[f"_{k}"] for k in keys): r["f05"] for _, r in g.iterrows()}
    sm = []
    for _, r in g.iterrows():
        vals = []
        base = [r[f"_{k}"] for k in keys]
        for d in itertools.product((-1, 0, 1), repeat=3):
            key = (r["assign"],) + tuple(b + x for b, x in zip(base, d))
            if key in lookup:
                vals.append(lookup[key])
        sm.append(np.mean(vals))
    g["f05_smooth"] = sm
    return g.sort_values(["f05_smooth", "f05"], ascending=False).iloc[0]


def main():
    from .data import load_truth, _read_cached
    from .splits import load_folds
    from .train import work_dir
    p = base_parser(__doc__)
    a = p.parse_args()
    oof = pd.read_parquet(work_dir() / "oof.parquet", columns=["s", "t", "p", "p_cross", "y"])
    f = load_folds()
    n_s1 = len(f)
    n_true = f.n_true.to_numpy()
    country = f.country.to_numpy()
    s, t, pr, y = oof.s.to_numpy(np.int64), oof.t.to_numpy(np.int64), oof.p.to_numpy(), oof.y.to_numpy()
    out = ["# Decision layer (OOF, full train, exact macro F0.5)\n"]

    g = grid_search(n_s1, n_true, s, t, pr, y)
    best_peak = g.sort_values("f05", ascending=False).iloc[0]
    choice = stable_choice(g)
    flat = decide(s, t, pr, 0.5, 0.0, 0.5, assign=False)
    out += [f"flat 0.5 cutoff (no assignment): **{fast_macro_f05(n_s1, n_true, s, y, flat):.4f}**\n",
            f"single peak: {best_peak.to_dict()}\n",
            f"chosen (stable, neighbourhood-smoothed): {choice.to_dict()}\n",
            "## top 15 settings\n", g.sort_values("f05", ascending=False).head(15).round(4).to_markdown(index=False), ""]

    # transfer check: tune on one country, apply to the other (with the cross-country model scores too)
    rows = []
    for tune_c, eval_c in (("US", "India"), ("India", "US")):
        m_t = country == tune_c
        m_e = country == eval_c
        gt = grid_search(n_s1, n_true, s, t, pr, y, s_mask=m_t)
        ct = stable_choice(gt)
        pred = decide(s, t, pr, ct.t_keep, ct.ratio, ct.t_null, bool(ct.assign))
        pc_ = oof.p_cross.fillna(0).to_numpy()
        pred_x = decide(s, t, pc_, ct.t_keep, ct.ratio, ct.t_null, bool(ct.assign))
        own = decide(s, t, pr, choice.t_keep, choice.ratio, choice.t_null, bool(choice.assign))
        rows.append({"thresholds tuned on": tune_c, "evaluated on": eval_c,
                     "tuned": f"keep {ct.t_keep} ratio {ct.ratio} null {ct.t_null} assign {ct.assign}",
                     "F0.5 (in-country model, transferred thresholds)": fast_macro_f05(n_s1, n_true, s, y, pred, m_e),
                     "F0.5 (other-country model + thresholds = France proxy)":
                         fast_macro_f05(n_s1, n_true, s, y, pred_x, m_e),
                     "F0.5 (global chosen thresholds)": fast_macro_f05(n_s1, n_true, s, y, own, m_e)})
    out += ["## Threshold transfer / cross-country (France proxy)\n", pd.DataFrame(rows).round(4)
            .to_markdown(index=False), ""]

    # breakdowns at the chosen setting vs flat 0.5
    pred = decide(s, t, pr, choice.t_keep, choice.ratio, choice.t_null, bool(choice.assign))
    fc = per_s1_f05(n_s1, n_true, s, y, pred)
    ff = per_s1_f05(n_s1, n_true, s, y, flat)
    bd = pd.DataFrame({"country": country, "size": np.minimum(n_true, 8), "chosen": fc, "flat_0.5": ff})
    out += ["## By country\n", bd.groupby("country")[["chosen", "flat_0.5"]].mean().round(4).to_markdown(), "",
            "## By true match-set size (8 = 8+)\n",
            bd.groupby("size")[["chosen", "flat_0.5"]].agg(["mean", "size"]).round(4).to_markdown(), "",
            "## Singletons vs non-singletons\n",
            bd.assign(kind=np.where(n_true == 0, "singleton", "non-singleton")).groupby("kind")[["chosen", "flat_0.5"]]
            .mean().round(4).to_markdown(), ""]

    # exactness check against metrics.report on the chosen setting
    from .metrics import report
    ids = _read_cached("train_source1", columns=["entity_id"]).entity_id.to_numpy()
    t_ids = pd.concat([_read_cached(f"train_source{n}", columns=["entity_id"]).entity_id for n in (2, 3)],
                      ignore_index=True).to_numpy()
    pred_df = pd.DataFrame({"s": ids[s[pred]], "t": t_ids[t[pred]]})
    pmap = pred_df.groupby("s").t.agg(list).to_dict()
    rep = report(load_truth(), pmap)
    exact = float(rep["overall"].loc["all", "macro_f05"])
    fast = fast_macro_f05(n_s1, n_true, s, y, pred)
    out.append(f"\nexactness check: metrics.report {exact:.6f} vs fast {fast:.6f}\n")
    assert abs(exact - fast) < 1e-9, (exact, fast)

    cfg = {"t_keep": float(choice.t_keep), "ratio": float(choice.ratio), "t_null": float(choice.t_null),
           "assign": bool(choice.assign), "cap": CAP, "cv_macro_f05": fast,
           "cv_by_country": bd.groupby("country").chosen.mean().round(4).to_dict()}
    (paths.WORK_DIR / "models" / "decision.json").write_text(json.dumps(cfg, indent=1))
    txt = "\n".join(out) + f"\nsaved decision config: {cfg}\n"
    (paths.REPORTS_DIR / "decision_report.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
