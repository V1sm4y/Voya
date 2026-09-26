"""OOF error analysis at the chosen decision setting. Writes reports/error_analysis.md.

Every error is put into one category, and the category's oracle gain is computed: the exact macro F0.5 if
exactly those errors were fixed (false merges removed / missed pairs added). This ranks fixes by upper bound.

False merges (predicted, not true):
  decoy false merge     the target belongs to no S1 (planted near-duplicate)
  generic-name merge    the target belongs to another S1, and the names are near-identical and short (<= 2 tokens)
  wrong-S1 merge        the target belongs to another S1 (other cases)
Missed true pairs:
  blocking miss         never a candidate
  lost to assignment    p passed the thresholds, but another S1 had a higher p for this target
  lost to null          the S1 was predicted empty (best p < t_null)
  Indic-script miss     low p, the target name is in an Indic script
  address-only miss     low p, the name is unrelated (n_tset_best < 50) but the address agrees (a_tset >= 70)
  empty-address miss    low p, the target address is empty
  other low-p miss

Usage (from src/):  python -m ber.errors --sample full
"""
import json

import numpy as np
import pandas as pd

from . import paths
from .cli import base_parser
from .decide import FastDecider, assignment_mask, fast_macro_f05


def analyse(oof: pd.DataFrame, n_true: np.ndarray, owner_of_t: np.ndarray, cfg: dict, country: np.ndarray):
    s, t, p, y = oof.s.to_numpy(np.int64), oof.t.to_numpy(np.int64), oof.p.to_numpy(), oof.y.to_numpy()
    n_s1 = len(n_true)
    fd = FastDecider(s, t, p)
    pred = fd(cfg["t_keep"], cfg["ratio"], cfg["t_null"], cfg["assign"])
    base_f = fast_macro_f05(n_s1, n_true, s, y, pred)
    alive, _, best = fd.prep[bool(cfg["assign"])]
    cat = np.full(len(s), "", object)
    fp = pred & (y == 0)
    own = owner_of_t[t]
    generic = (oof.n_tset_best.to_numpy() >= 90) & (oof.tok_t.to_numpy() <= 2)
    cat[fp & (own < 0)] = "decoy false merge"
    cat[fp & (own >= 0) & generic] = "generic-name merge"
    cat[fp & (own >= 0) & ~generic] = "wrong-S1 merge"
    fn = (~pred) & (y == 1)
    passes = (p >= cfg["t_keep"]) & (p >= cfg["ratio"] * best)
    lost_assign = fn & ~alive & passes
    lost_null = fn & ~lost_assign & (best < cfg["t_null"]) & (p >= cfg["t_keep"])
    low = fn & ~lost_assign & ~lost_null
    indic = oof.t_indic.to_numpy() > 0
    addr_only = (oof.n_tset_best.to_numpy() < 50) & (oof.a_tset.to_numpy() >= 70)
    empty = oof.t_addr_empty.to_numpy() > 0
    cat[lost_assign] = "lost to assignment"
    cat[lost_null] = "lost to null"
    cat[low & indic] = "Indic-script miss"
    cat[low & ~indic & addr_only] = "address-only miss"
    cat[low & ~indic & ~addr_only & empty] = "empty-address miss"
    cat[low & ~indic & ~addr_only & ~empty] = "other low-p miss"
    rows = []
    for c in pd.unique(cat[cat != ""]):
        m = cat == c
        if c.endswith("merge"):
            f = fast_macro_f05(n_s1, n_true, s, y, pred & ~m)
        else:
            f = fast_macro_f05(n_s1, n_true, s, y, pred | m)
        ctry = pd.Series(country[s[m]]).value_counts().to_dict()
        rows.append({"category": c, "pairs": int(m.sum()), "India": ctry.get("India", 0), "US": ctry.get("US", 0),
                     "oracle_gain": f - base_f})
    # blocking misses: true pairs never scored
    scored_true = np.bincount(s[y == 1], minlength=n_s1)
    missing = n_true - scored_true
    # oracle: pretend the missing pairs were predicted -> tp and npred both +missing
    npred = np.bincount(s[pred], minlength=n_s1) + missing
    tp = np.bincount(s[pred & (y == 1)], minlength=n_s1) + missing
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(npred > 0, tp / npred, 0)
        rc = np.where(n_true > 0, tp / n_true, 0)
        ff = np.where(tp > 0, 1.25 * pr * rc / (0.25 * pr + rc), 0)
    ff = np.where(n_true == 0, (npred == 0).astype(float), ff)
    rows.append({"category": "blocking miss", "pairs": int(missing.sum()),
                 "India": int(missing[country == "India"].sum()), "US": int(missing[country == "US"].sum()),
                 "oracle_gain": float(ff.mean()) - base_f})
    tab = pd.DataFrame(rows).sort_values("oracle_gain", ascending=False)
    return base_f, tab, cat


def main():
    from .splits import load_folds
    from .train import owner_array, work_dir
    p = base_parser(__doc__)
    a = p.parse_args()
    cfg = json.loads((paths.WORK_DIR / "models" / "decision.json").read_text())
    oof = pd.read_parquet(work_dir() / "oof.parquet")
    f = load_folds()
    from .data import _read_cached
    n_t = sum(len(_read_cached(f"train_source{n}", columns=["entity_id"])) for n in (2, 3))
    owner = owner_array(n_t, None)
    base_f, tab, cat = analyse(oof, f.n_true.to_numpy(), owner, cfg, f.country.to_numpy())
    out = [f"# OOF error analysis (decision {cfg})\n", f"CV macro F0.5 at this setting: **{base_f:.4f}**\n",
           "oracle_gain = macro F0.5 if exactly this category's errors were fixed, minus the current score.\n",
           tab.round(5).to_markdown(index=False), ""]
    # a few examples per top category
    from .data import _read_cached as rc
    s1 = rc("train_source1", columns=["business_name", "business_address"])
    tg = pd.concat([rc(f"train_source{n}", columns=["business_name", "business_address"]) for n in (2, 3)],
                   ignore_index=True)
    for c in tab.category.head(4):
        if c == "blocking miss":
            continue
        ex = oof[cat == c].sample(min(8, int((cat == c).sum())), random_state=0)
        d = pd.DataFrame({"p": ex.p.round(3).values, "s1_name": s1.business_name.values[ex.s],
                          "s1_addr": s1.business_address.values[ex.s], "t_name": tg.business_name.values[ex.t],
                          "t_addr": tg.business_address.values[ex.t]})
        out += [f"### examples: {c}\n", d.to_markdown(index=False), ""]
    txt = "\n".join(out)
    (paths.REPORTS_DIR / "error_analysis.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
