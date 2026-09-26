"""How useful are the parsed name/address fields? Writes reports/parse_<sample>.md.

1. Coverage: how often each parsed field is filled, per source x country (DEV S1 plus a sample of the pool).
2. Discrimination: over the exact-name + same-city baseline pairs, how often each field agrees for true
   pairs (TP) against false pairs that belong to another S1 (FP_other) or are decoys (FP_decoy).
3. Simple rule variants on top of the baseline, scored with the exact metric.

Usage (from src/):  python -m ber.parse_report --sample dev
"""
import pandas as pd

from . import metrics, paths
from .address import parse_address
from .baselines import name_city_pairs, to_pred
from .cli import base_parser
from .data import load, load_truth
from .names import legal_relation, parse_name

ADDR_FIELDS = ["state", "city", "locality", "street", "building", "premise_no", "unit", "floor", "landmark",
               "admin", "pin"]


def _set(v):
    return set(v.split("|")) if isinstance(v, str) and v else set()


def agreement(pa: dict, pb: dict, field: str) -> str:
    """'agree' (shared value), 'conflict' (both filled, nothing shared), or 'missing'."""
    a, b = _set(pa.get(field)), _set(pb.get(field))
    if not a or not b:
        return "missing"
    return "agree" if a & b else "conflict"


def coverage(b, n=100_000) -> pd.DataFrame:
    rows = []
    for src, df in (("S1", b.s1), ("S2", b.s2), ("S3", b.s3)):
        for country, g in df.groupby("country"):
            g = g.sample(min(n, len(g)), random_state=0)
            parsed = [parse_address(a, c) for a, c in zip(g.business_address, g.country)]
            p = pd.DataFrame(parsed)
            row = {"source": src, "country": country, "n": len(g)}
            row.update({f: p[f].notna().mean() for f in ADDR_FIELDS})
            rows.append(row)
    return pd.DataFrame(rows).set_index(["source", "country"])


def main():
    args = base_parser(__doc__).parse_args()
    b = load("train", args.sample)
    print(b.describe(), flush=True)
    out = [f"# Parse report — train/{args.sample}\n", b.describe() + "\n"]

    cov = coverage(b)
    out += ["## 1. Field coverage (share of records with the field filled)\n", (cov * 100).round(1)
            .assign(n=cov.n.astype(int)).to_markdown(), ""]
    print(out[-2], flush=True)

    pairs = name_city_pairs(b)
    owner = {i: s for s, l in load_truth().items() for i in l}
    dev = set(b.s1.entity_id)
    kind = []
    for s1, t in zip(pairs.entity_id, pairs.entity_id_t):
        o = owner.get(t)
        kind.append("TP" if o == s1 else ("FP_decoy" if o is None else "FP_other"))
    pairs["kind"] = kind
    pa = [parse_address(a, c) for a, c in zip(pairs.business_address, pairs.country)]
    pb = [parse_address(a, c) for a, c in zip(pairs.business_address_t, pairs.country)]
    for f in ["premise_no", "locality_key", "street_key", "building_key", "landmark", "unit", "floor", "pin"]:
        pairs[f] = [agreement(x, y, f) for x, y in zip(pa, pb)]
    pairs["legal"] = [legal_relation(parse_name(x)["legal_form"], parse_name(y)["legal_form"])
                      for x, y in zip(pairs.business_name, pairs.business_name_t)]

    out.append("## 2. Field agreement on exact-name + same-city pairs (row % within pair kind)\n")
    for country, g in pairs.groupby("country"):
        out.append(f"### {country}  (pairs: " + ", ".join(f"{k} {v:,}" for k, v in g.kind.value_counts().items()) + ")\n")
        for f in ["premise_no", "locality_key", "street_key", "building_key", "landmark", "unit", "legal"]:
            ct = (pd.crosstab(g.kind, g[f], normalize="index") * 100).round(1)
            out += [f"**{f}**\n", ct.to_markdown(), ""]
    print("\n".join(out[-40:]), flush=True)

    # 3. rule variants
    country = dict(zip(b.s1.entity_id, b.s1.country))
    no_conf = pairs.legal != "conflict"
    prem_ok = pairs.premise_no != "conflict"
    unit_ok = pairs.unit != "conflict"
    any_addr = (pairs[["premise_no", "locality_key", "street_key", "building_key"]] == "agree").any(axis=1)
    variants = {
        "baseline (name+city)": pairs.index == pairs.index,
        "+ no legal conflict": no_conf,
        "+ no legal conflict + no premise/unit conflict": no_conf & prem_ok & unit_ok,
        "+ no legal conflict + no premise/unit conflict + India needs one address agreement":
            no_conf & prem_ok & unit_ok & ((pairs.country != "India") | any_addr),
    }
    out.append("## 3. Rule variants (macro F0.5 on this DEV)\n")
    rows = []
    from .splits import dev_weights
    weights = dev_weights("train") if args.sample == "dev" else None
    for name, m in variants.items():
        rep = metrics.report(b.truth, to_pred(pairs[m]), country=country, weights=weights)
        cols = ["macro_f05"] + (["macro_f05_w"] if weights is not None else []) + ["link_precision", "link_recall"]
        r = {"variant": name, **rep["overall"].loc["all", cols].to_dict()}
        r.update({f"F0.5 {k}": v for k, v in rep["by_country"].macro_f05.items()})
        r.update({f"prec {k}": v for k, v in rep["by_country"].link_precision.items()})
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("variant").round(4)
    out += [tab.to_markdown(), ""]
    print(tab.to_string(), flush=True)

    paths.REPORTS_DIR.mkdir(exist_ok=True)
    dst = paths.REPORTS_DIR / f"parse_{args.sample}.md"
    dst.write_text("\n".join(out), encoding="utf-8")
    print("wrote", dst)


if __name__ == "__main__":
    main()
