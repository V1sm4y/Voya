"""Reference baselines (no learning), scored with the exact metric.

  all_empty        predict [] for every S1 (score = singleton share)
  name_city_exact  every S2/S3 record of the same country whose core_name (names.parse_name, legal form
                   ignored) equals the S1's AND whose address contains the S1 city (word-boundary match)
  ... + no legal conflict   the same, minus pairs whose legal forms conflict (names.legal_relation)

Usage (from src/):  python -m ber.baselines --sample dev
"""
import time

import pandas as pd

from . import metrics, paths, splits
from .cli import base_parser
from .data import load
from .names import legal_relation, parse_name
from .normalize import basic, contains_phrase, s1_city


def _core(names: pd.Series) -> pd.Series:
    return names.map(lambda n: parse_name(n)["core_name"])


def _legal(names: pd.Series) -> pd.Series:
    return names.map(lambda n: parse_name(n)["legal_form"])


def name_city_pairs(b) -> pd.DataFrame:
    """All (S1, target) pairs with equal core_name (legal form ignored) and the S1 city in the target address."""
    t = time.time()
    s1 = b.s1.assign(key=_core(b.s1.business_name), city=b.s1.business_address.map(s1_city))
    tg = b.targets
    tg = tg.assign(key=_core(tg.business_name))
    print(f"  normalised {len(s1):,} S1 + {len(tg):,} targets in {time.time() - t:.0f}s", flush=True)
    s1k = s1.dropna(subset=["key"])
    tg = tg[tg.key.isin(set(s1k.key))]  # only targets that can join
    pairs = s1k[["entity_id", "country", "key", "city", "business_name", "business_address"]].merge(
        tg[["entity_id", "country", "key", "business_name", "business_address"]], on=["country", "key"],
        suffixes=("", "_t"))
    n_name = len(pairs)
    pairs = pairs[[c is not None and contains_phrase(basic(a), c)
                   for c, a in zip(pairs.city, pairs.business_address_t)]]
    print(f"  pairs: exact-name {n_name:,} -> +same-city {len(pairs):,}; "
          f"S1 with no city parsed: {s1.city.isna().mean():.2%}", flush=True)
    return pairs.reset_index(drop=True)


def to_pred(pairs: pd.DataFrame) -> dict:
    return pairs.groupby("entity_id").entity_id_t.agg(list).to_dict()


def no_legal_conflict(pairs: pd.DataFrame) -> pd.DataFrame:
    """Drop pairs whose legal forms conflict (LLC vs Corp, Pvt Ltd vs Public Ltd...)."""
    rel = [legal_relation(a, b) for a, b in zip(_legal(pairs.business_name), _legal(pairs.business_name_t))]
    return pairs[[r != "conflict" for r in rel]]


def main():
    p = base_parser(__doc__)
    p.add_argument("--legacy-random-dev", action="store_true",
                   help="evaluate on the old 5%% random DEV instead (comparison only)")
    args = p.parse_args()
    if args.split != "train":
        raise SystemExit("baselines are scored on train (labels needed)")
    sample = "dev_random" if args.legacy_random_dev and args.sample == "dev" else args.sample
    b = load("train", sample)
    print(b.describe(), flush=True)
    country = dict(zip(b.s1.entity_id, b.s1.country))
    folds = splits.load_folds().set_index("entity_id").fold
    weights = splits.dev_weights("train") if sample == "dev" else None
    pairs = name_city_pairs(b)
    preds = {"all_empty": {}, "name_city_exact": to_pred(pairs),
             "name_city_exact + no legal conflict": to_pred(no_legal_conflict(pairs))}

    out = [f"# Baselines — train/{sample}\n", b.describe() + "\n"]
    for name, pred in preds.items():
        rep = metrics.report(b.truth, pred, country=country, weights=weights)
        e = rep["per_entity"]
        per_fold = e.groupby(e.s1.map(folds)).f05.mean()
        txt = metrics.format_report(rep, name)
        txt += (f"\nper-fold macro F0.5: {', '.join(f'{v:.4f}' for v in per_fold)} "
                f"(std {per_fold.std():.4f})\n")
        print(txt, flush=True)
        out.append(txt)
    paths.REPORTS_DIR.mkdir(exist_ok=True)
    dst = paths.REPORTS_DIR / f"baselines_{sample}.md"
    dst.write_text("\n".join(out), encoding="utf-8")
    print("wrote", dst)


if __name__ == "__main__":
    main()
