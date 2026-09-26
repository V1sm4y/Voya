"""Train the pair classifier (LightGBM) and produce out-of-fold predictions for every train pair.

Pass 1 (collect): stream features (features.iter_features) for a random TRAIN_FRAC of S1s (seeded, spread over
    all folds; NOT the geographic DEV). Keep all positives, the hard negatives (rank_base_s <= HARD_RANK) and a
    deterministic EASY_RATE sample of the remaining negatives, weighted 1/EASY_RATE. Written to
    data/cache/train/rows.parquet.
Pass 2 (fit): 5 LightGBM models on the fold assignment of splits.py (GroupKFold by S1; model k never sees fold
    k's S1s), with early stopping on fold k's rows. Plus two cross-country models (US-only, India-only) as
    the France proxy. Models go to data/models/.
Pass 3 (oof): stream features for EVERY train pair; each pair is scored by the model of its S1's fold, which is
    valid out-of-fold for all 2.2M S1s. The complete set of competing S1s is therefore present for the
    one-S1-per-record assignment in decide.py. Also scored with the cross-country model of the other country.
    Only pairs with p >= KEEP_P are written (data/cache/train/oof.parquet) with a few diagnostic columns.

Usage (from src/):  python -m ber.train --sample full [--stage collect|fit|oof|all]
"""
import json
import time

import numpy as np
import pandas as pd

from . import paths
from .cli import base_parser

TRAIN_FRAC = 0.30
HARD_RANK = 10
EASY_RATE = 0.05
KEEP_P = 0.02
SEED = 2026
DIAG = ["n_tset_best", "a_tset", "base", "rank_base_s", "num_both", "num_eq", "legal_rel", "t_indic",
        "t_addr_empty", "t_domain", "extra_t_cnt", "tok_t", "n_cand_s"]
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_estimators=1500, max_bin=255,
              random_state=SEED, n_jobs=-1, verbose=-1)


def work_dir():
    d = paths.CACHE_DIR / "train"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_dir():
    d = paths.WORK_DIR / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def owner_array(n_targets: int, n2: int) -> np.ndarray:
    """owner[t_global] = S1 row index of its true S1, -1 if unlinked."""
    from .data import _read_cached, load_truth
    s1_ids = _read_cached("train_source1", columns=["entity_id"]).entity_id
    t_ids = pd.concat([_read_cached(f"train_source{n}", columns=["entity_id"]).entity_id for n in (2, 3)],
                      ignore_index=True)
    s_pos = pd.Series(np.arange(len(s1_ids)), index=s1_ids.values)
    truth = load_truth()
    pairs = [(s, t) for s, ts in truth.items() for t in ts]
    tp = pd.DataFrame(pairs, columns=["s", "t"])
    owner = np.full(n_targets, -1, np.int32)
    t_idx = pd.Index(t_ids.values).get_indexer(tp.t)
    owner[t_idx] = s_pos.reindex(tp.s).to_numpy()
    return owner


def _hash01(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Deterministic uniform [0,1) per (a, b)."""
    h = (a.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ (b.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9))
    h ^= h >> np.uint64(31)
    return (h % np.uint64(1 << 30)).astype(np.float64) / float(1 << 30)


def train_sample_mask(n_s1: int) -> np.ndarray:
    return np.random.default_rng(SEED).random(n_s1) < TRAIN_FRAC


def collect(ctx, cands, folds, owner, verbose=True):
    """Stream training rows to data/cache/train/rows.parquet chunk by chunk (never all in RAM)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    want = train_sample_mask(len(folds))
    path = work_dir() / "rows.parquet"
    writer = None
    n_rows = n_pos = 0
    for _, df in iter_features_lazy(ctx, cands, want):
        s, t = df.s.to_numpy(), df.t.to_numpy()
        y = owner[t] == s
        hard = df.rank_base_s.to_numpy() <= HARD_RANK
        easy = _hash01(s, t) < EASY_RATE
        keep = y | hard | easy
        out = df[keep].copy()
        out["y"] = y[keep].astype(np.int8)
        out["w"] = np.where(y[keep] | hard[keep], 1.0, 1.0 / EASY_RATE).astype(np.float32)
        out["fold"] = folds[s[keep]].astype(np.int8)
        tab = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, tab.schema, compression="zstd")
        writer.write_table(tab)
        n_rows += len(out)
        n_pos += int(out.y.sum())
    writer.close()
    if verbose:
        print(f"  training rows: {n_rows:,} (pos {n_pos:,}) from {int(want.sum()):,} S1 -> {path}", flush=True)


def iter_features_lazy(ctx, cands, want):
    from .features import iter_features
    return iter_features(ctx, cands, want_s=want)


def fit(rows: pd.DataFrame, s1_country: np.ndarray, countries: list, verbose=True) -> dict:
    import lightgbm as lgb
    from .features import FEATURES
    X = rows[FEATURES].to_numpy(np.float32)
    y = rows.y.to_numpy()
    w = rows.w.to_numpy()
    fold = rows.fold.to_numpy()
    info = {"folds": {}, "features": FEATURES}
    imp = np.zeros(len(FEATURES))
    for k in range(5):
        t0 = time.time()
        tr, va = fold != k, fold == k
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X[tr], y[tr], sample_weight=w[tr], eval_set=[(X[va], y[va])], eval_sample_weight=[w[va]],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        m.booster_.save_model(str(model_dir() / f"fold{k}.txt"))
        imp += m.booster_.feature_importance("gain")
        info["folds"][k] = {"best_iter": int(m.best_iteration_ or m.n_estimators), "sec": round(time.time() - t0)}
        if verbose:
            print(f"  fold {k}: best_iter {info['folds'][k]['best_iter']} ({time.time() - t0:.0f}s)", flush=True)
    # cross-country models (France proxy): fit on one country, applied to the other in the OOF pass
    rc = s1_country[rows.s.to_numpy()]
    for ci, c in enumerate(countries):
        m = lgb.LGBMClassifier(**{**PARAMS, "n_estimators": int(np.median([f["best_iter"] for f in info["folds"].values()]))})
        sel = rc == ci
        m.fit(X[sel], y[sel], sample_weight=w[sel])
        m.booster_.save_model(str(model_dir() / f"country_{c}.txt"))
    (model_dir() / "train_info.json").write_text(json.dumps(info, indent=1))
    fi = pd.DataFrame({"feature": FEATURES, "gain": imp / 5}).sort_values("gain", ascending=False)
    fi["share"] = fi.gain / fi.gain.sum()
    paths.REPORTS_DIR.mkdir(exist_ok=True)
    (paths.REPORTS_DIR / "feature_importance.md").write_text(
        "# Feature importance (mean gain over 5 fold models)\n\n" + fi.round(4).to_markdown(index=False) + "\n")
    return info


def load_models():
    import lightgbm as lgb
    d = model_dir()
    folds = [lgb.Booster(model_file=str(d / f"fold{k}.txt")) for k in range(5)]
    cross = {p.stem.replace("country_", ""): lgb.Booster(model_file=str(p)) for p in d.glob("country_*.txt")}
    return folds, cross


def oof(ctx, cands, folds_arr, owner, verbose=True):
    from .features import FEATURES, iter_features
    models, cross = load_models()
    parts = []
    t_all = time.time()
    for _, df in iter_features(ctx, cands, verbose=verbose):
        X = df[FEATURES].to_numpy(np.float32)
        s, t = df.s.to_numpy(), df.t.to_numpy()
        f = folds_arr[s]
        p = np.empty(len(df), np.float32)
        for k in range(5):
            m = f == k
            if m.any():
                p[m] = models[k].predict(X[m])
        ci = ctx.s1_country[s]
        pc_ = np.full(len(df), np.nan, np.float32)
        for c, i in (("US", "India"), ("India", "US")):   # model of the OTHER country
            if c in cross and i in ctx.countries:
                m = ci == ctx.countries.index(i)
                if m.any():
                    pc_[m] = cross[c].predict(X[m])
        keep = (p >= KEEP_P) | (owner[t] == s)
        out = df.loc[keep, ["s", "t"] + DIAG].copy()
        out["p"] = p[keep]
        out["p_cross"] = pc_[keep]
        out["y"] = (owner[t] == s)[keep].astype(np.int8)
        parts.append(out)
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(work_dir() / "oof.parquet", index=False, compression="zstd")
    if verbose:
        print(f"  OOF pairs kept: {len(res):,} ({(time.time() - t_all) / 60:.1f} min)", flush=True)
    return res


def pr_auc_report(oof_df: pd.DataFrame, ctx) -> str:
    """PR-AUC over scored pairs (the p < KEEP_P tail is kept only for positives, which slightly understates
    precision; noted in the report)."""
    from sklearn.metrics import average_precision_score
    from .splits import load_folds
    lines = ["# Model report (OOF over every train pair)\n",
             f"pairs kept: p >= {KEEP_P} plus every true pair (so PR-AUC is computed on that set)\n"]
    ci = ctx.s1_country[oof_df.s.to_numpy()]
    rows = [{"segment": "all", "pairs": len(oof_df), "pos": int(oof_df.y.sum()),
             "pr_auc": average_precision_score(oof_df.y, oof_df.p)}]
    for i, c in enumerate(ctx.countries):
        m = ci == i
        rows.append({"segment": c, "pairs": int(m.sum()), "pos": int(oof_df.y[m].sum()),
                     "pr_auc": average_precision_score(oof_df.y[m], oof_df.p[m]),
                     "pr_auc_cross_country": average_precision_score(oof_df.y[m], oof_df.p_cross[m])
                     if oof_df.p_cross[m].notna().all() else np.nan})
    f = load_folds()
    metro = f[(f.country == "India") & (f.tier == "metro")]
    city = pd.Series(metro.city.values, index=metro.index)
    oc = city.reindex(oof_df.s.to_numpy()).to_numpy()
    for c in pd.unique(metro.city):
        m = oc == c
        if m.sum() > 1000:
            rows.append({"segment": f"India metro: {c}", "pairs": int(m.sum()), "pos": int(oof_df.y[m].sum()),
                         "pr_auc": average_precision_score(oof_df.y[m], oof_df.p[m])})
    lines.append(pd.DataFrame(rows).round(4).to_markdown(index=False))
    return "\n".join(lines) + "\n"


def main():
    from .blocking import load_candidates
    from .features import FeatureContext
    from .splits import load_folds
    p = base_parser(__doc__)
    p.add_argument("--stage", choices=["collect", "fit", "oof", "all"], default="all")
    a = p.parse_args()
    if a.split != "train":
        raise SystemExit("train.py runs on the train split")
    tag = f"train_{a.sample}"
    t0 = time.time()
    folds_arr = load_folds().fold.to_numpy()
    if a.stage == "fit":
        from .data import _read_cached
        ctry = _read_cached("train_source1", columns=["country"]).country
        countries = sorted(ctry.unique())
        rows = pd.read_parquet(work_dir() / "rows.parquet")
        fit(rows, ctry.map({c: i for i, c in enumerate(countries)}).to_numpy(), countries)
        print(f"total {(time.time() - t0) / 60:.1f} min")
        return
    ctx = FeatureContext("train")
    cands = load_candidates(tag)
    owner = owner_array(len(ctx.tg.num), ctx.n2)
    if a.sample == "dev":   # dev: restrict to DEV S1s (for quick smoke runs only)
        from .data import sample_masks
        m = sample_masks("train", "dev")[1]
        cands = cands[m[cands.s.to_numpy()]].reset_index(drop=True)
    if a.stage in ("collect", "all"):
        collect(ctx, cands, folds_arr, owner)
    if a.stage in ("fit", "all"):
        rows = pd.read_parquet(work_dir() / "rows.parquet")
        fit(rows, ctx.s1_country, ctx.countries)
        del rows
    if a.stage in ("oof", "all"):
        res = oof(ctx, cands, folds_arr, owner)
        rep = pr_auc_report(res, ctx)
        (paths.REPORTS_DIR / "model_report.md").write_text(rep)
        print(rep)
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
