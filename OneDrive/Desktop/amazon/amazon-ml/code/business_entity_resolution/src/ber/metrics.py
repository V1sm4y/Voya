"""Exact challenge metric: macro-averaged F0.5 over Source-1 entities.

Per S1 entity (truth T, prediction P, both treated as sets):
  T empty, P empty      -> 1.0
  T empty, P non-empty  -> 0.0
  T non-empty, P empty  -> 0.0
  otherwise             -> F0.5 = 1.25*prec*rec / (0.25*prec + rec), 0 if no overlap
The score is the plain mean over the evaluated S1 ids. An S1 missing from `pred` counts as an
empty prediction (the real scorer would reject the file; the writer guarantees it can't happen).
"""
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

BETA2 = 0.25  # beta = 0.5


def f05(true_ids: Iterable[str], pred_ids: Iterable[str]) -> float:
    t, p = set(true_ids), set(pred_ids)
    if not t:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    tp = len(t & p)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(t)
    return (1 + BETA2) * prec * rec / (BETA2 * prec + rec)


def per_entity(truth: Mapping[str, Iterable[str]], pred: Mapping[str, Iterable[str]],
               ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """One row per evaluated S1: n_true, n_pred, tp, f05. `ids` defaults to all truth keys."""
    ids = list(truth) if ids is None else list(ids)
    missing = [i for i in ids if i not in truth]
    if missing:
        raise KeyError(f"{len(missing)} evaluated ids have no truth, e.g. {missing[:3]}")
    n_true = np.empty(len(ids), np.int32); n_pred = np.empty(len(ids), np.int32)
    tp = np.empty(len(ids), np.int32); f = np.empty(len(ids), np.float64)
    empty = ()
    for k, i in enumerate(ids):
        t, p = set(truth[i]), set(pred.get(i, empty))
        n_true[k], n_pred[k], tp[k] = len(t), len(p), len(t & p)
        f[k] = f05(t, p)
    return pd.DataFrame({"s1": ids, "n_true": n_true, "n_pred": n_pred, "tp": tp, "f05": f})


def _summ(g: pd.DataFrame) -> pd.Series:
    sp, st = g.n_pred.sum(), g.n_true.sum()
    extra = {}
    if "w" in g:
        extra["macro_f05_w"] = np.average(g.f05, weights=g.w) if g.w.sum() > 0 else np.nan
    return pd.Series({
        "n_s1": len(g),
        "macro_f05": g.f05.mean(),
        **extra,
        "link_precision": g.tp.sum() / sp if sp else np.nan,   # micro, diagnostic only
        "link_recall": g.tp.sum() / st if st else np.nan,
        "pred_empty_rate": (g.n_pred == 0).mean(),
        "mean_n_pred": g.n_pred.mean(),
    })


def report(truth: Mapping[str, Iterable[str]], pred: Mapping[str, Iterable[str]],
           country: Optional[Mapping[str, str]] = None, ids: Optional[Iterable[str]] = None,
           max_size: int = 8, weights: Optional[Mapping[str, float]] = None) -> Dict[str, pd.DataFrame]:
    """Overall, singleton vs non-singleton, by country, by true match-set size (clipped at max_size+).

    `weights` (e.g. splits.dev_weights()) adds `macro_f05_w`, a weighted macro F0.5 that estimates the
    full-data score from a stratified sample. S1s missing from `weights` get weight 0."""
    e = per_entity(truth, pred, ids)
    if weights is not None:
        e["w"] = e.s1.map(weights).fillna(0.0)
    extra = len(set(pred) - set(e.s1))
    out = {"overall": _summ(e).to_frame("all").T}
    out["overall"]["pred_s1_not_evaluated"] = extra
    e["kind"] = np.where(e.n_true == 0, "singleton", "non-singleton")
    out["by_kind"] = e.groupby("kind").apply(_summ, include_groups=False)
    if country is not None:
        e["country"] = e.s1.map(country)
        out["by_country"] = e.groupby("country").apply(_summ, include_groups=False)
    e["true_size"] = e.n_true.clip(upper=max_size).astype(str).where(e.n_true < max_size, f"{max_size}+")
    order = [str(i) for i in range(max_size)] + [f"{max_size}+"]
    out["by_true_size"] = (e.groupby("true_size").apply(_summ, include_groups=False)
                           .reindex([o for o in order if o in set(e.true_size)]))
    out["per_entity"] = e
    return out


def format_report(rep: Dict[str, pd.DataFrame], title: str = "") -> str:
    fmt = {"n_s1": "{:,.0f}", "macro_f05": "{:.4f}", "macro_f05_w": "{:.4f}", "link_precision": "{:.4f}", "link_recall": "{:.4f}",
           "pred_empty_rate": "{:.3f}", "mean_n_pred": "{:.2f}", "pred_s1_not_evaluated": "{:,.0f}"}
    lines = [f"## {title}"] if title else []
    for key in ("overall", "by_kind", "by_country", "by_true_size"):
        if key not in rep:
            continue
        df = rep[key].copy()
        for c in df.columns:
            df[c] = [fmt.get(c, "{}").format(v) if pd.notna(v) else "–" for v in df[c]]
        lines += [f"\n**{key}**\n", df.to_markdown()]
    return "\n".join(lines) + "\n"
