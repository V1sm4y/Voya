"""Deterministic evaluation splits, grouped by S1 id, and the geographic DEV sample.

Each S1 entity is its own group: it goes to exactly one fold together with its whole match set.
S2/S3 match sets are disjoint across S1s (DATA_NOTES §2), so no S2/S3 record's label is shared
between folds.

Files written to data/splits/ (regenerate with `python -m ber.splits`):
  train_folds.parquet  entity_id, country, state, city, n_true, fold (0..4), dev (bool, geographic),
                       dev_random (bool, legacy 5% random sample, kept only for comparisons)
  dev_cities_train.parquet / dev_cities_test.parquet
                       the sampled (country, city) groups, with tier and S1 count
  test_dev.parquet     entity_id, country, state, city of test S1 records in the sampled test cities

Folds are stratified by (country, true match-set size clipped at 7). With one S1 per group, this is
exactly StratifiedGroupKFold, implemented with numpy so the split is fixed by SEED. The fold RNG
stream comes first, so adding or changing DEV logic never moves a fold.

Geographic DEV:
  * S1 records are grouped by (country, canonical city) from geo.py.
  * Within each country, cities are tiered by size: metro has >=1% of the country's S1, mid has
    >=0.05%, small is the rest. Random cities are drawn in each (country, tier) until about 5% of
    the tier's S1 records are covered, never overshooting and with at least one city per tier. Every
    country therefore contributes about one metro plus a spread of towns, and DEV stays close to 5%.
  * DEV S1 = every S1 in the drawn cities, so whole cities are present, including every S1 that
    competes for the same records.
  * DEV pool = every S2/S3 record in the same (country, state group) as a DEV S1, plus the records
    whose state is unknown (mostly empty addresses). Records with a wrong or missing city are
    therefore kept. Andhra Pradesh and Telangana form one state group, because S2/S3 still put
    Hyderabad in AP (DATA_NOTES §8).

Evaluation protocol: never split the S2/S3 pool; score only the evaluated S1 ids.
"""
import numpy as np
import pandas as pd

from . import paths

SEED = 2026
N_FOLDS = 5
DEV_FRAC = 0.05
SIZE_CLIP = 7
TIERS = [("metro", 0.01), ("mid", 0.0005), ("small", 0.0)]   # share of the country's S1 records
STATE_GROUPS = {"telangana": "andhra pradesh+telangana", "andhra pradesh": "andhra pradesh+telangana"}


def state_group(s):
    """Map a state (or a Series of states) to its pooling group."""
    if isinstance(s, pd.Series):
        return s.map(lambda x: STATE_GROUPS.get(x, x))
    return STATE_GROUPS.get(s, s)


def _stratified_order(strata: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Rank of each row within its stratum after a seeded shuffle."""
    rank = np.empty(len(strata), np.int64)
    for _, idx in strata.groupby(strata, sort=True).indices.items():
        perm = rng.permutation(len(idx))
        rank[idx[perm]] = np.arange(len(idx))
    return rank


def city_tiers(g1: pd.DataFrame) -> pd.DataFrame:
    """(country, city, n_s1, tier) for every S1 city."""
    cnt = g1.dropna(subset=["city"]).groupby(["country", "city"]).size().rename("n_s1").reset_index()
    share = cnt.n_s1 / cnt.groupby("country").n_s1.transform("sum")
    cnt["tier"] = np.select([share >= TIERS[0][1], share >= TIERS[1][1]], [TIERS[0][0], TIERS[1][0]], TIERS[2][0])
    return cnt


def sample_cities(g1: pd.DataFrame, rng: np.random.Generator, frac: float = DEV_FRAC) -> pd.DataFrame:
    """g1: S1 geo frame (country, city). Returns the chosen (country, city, tier, n_s1) rows.

    Within each (country, tier), cities are visited in random order and a city is taken when it keeps
    the tier's DEV S1 count within frac x (tier S1 count). At least one city per tier is taken. City
    sizes are very skewed (19 Indian cities hold half the S1s), so drawing a fixed share of *cities*
    gives the right size only on average: a single draw of Mumbai made DEV 11% of India."""
    cnt = city_tiers(g1)
    chosen = []
    for (country, tier), grp in cnt.sort_values(["country", "tier", "city"]).groupby(["country", "tier"], sort=True):
        budget = frac * grp.n_s1.sum()
        order = rng.permutation(len(grp))
        taken, total = [], 0
        for i in order:
            n = grp.n_s1.iloc[i]
            if total + n <= budget:
                taken.append(i); total += n
        if not taken:
            taken = [order[0]]
        chosen.append(grp.iloc[sorted(taken)])
    return pd.concat(chosen, ignore_index=True)


def make_splits(seed: int = SEED) -> None:
    from .data import _read_cached, load_truth
    from .geo import load_geo
    paths.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- folds (unchanged RNG stream: folds, then legacy random dev, then legacy test dev)
    s1 = _read_cached("train_source1", columns=["entity_id", "country"])
    truth = load_truth()
    s1["n_true"] = s1.entity_id.map(lambda i: len(truth[i])).astype(np.int16)
    strata = s1.country + "|" + s1.n_true.clip(upper=SIZE_CLIP).astype(str)
    s1["fold"] = (_stratified_order(strata, rng) % N_FOLDS).astype(np.int8)
    rank2 = _stratified_order(strata, rng)
    size = strata.map(strata.value_counts())
    s1["dev_random"] = rank2 < np.maximum(1, np.round(size * DEV_FRAC))
    t1 = _read_cached("test_source1", columns=["entity_id", "country"])
    _stratified_order(t1.country, rng)  # legacy draw, kept so the stream above never shifts

    # --- geographic DEV (independent stream)
    grng = np.random.default_rng(seed + 1)
    for split, frame in (("train", s1), ("test", t1)):
        g = load_geo(split, 1)[["entity_id", "state", "city"]]
        frame[["state", "city"]] = frame[["entity_id"]].merge(g, on="entity_id", how="left")[["state", "city"]].values
        tiers = city_tiers(frame)
        frame["tier"] = frame[["country", "city"]].merge(tiers, on=["country", "city"], how="left").tier.values
        cities = sample_cities(frame, grng)
        cities.to_parquet(paths.SPLITS_DIR / f"dev_cities_{split}.parquet", index=False)
        key = set(zip(cities.country, cities.city))
        frame["dev"] = [(c, ci) in key for c, ci in zip(frame.country, frame.city)]
    s1.to_parquet(paths.SPLITS_DIR / "train_folds.parquet", index=False)
    t1[t1.dev].drop(columns="dev").to_parquet(paths.SPLITS_DIR / "test_dev.parquet", index=False)


def load_folds() -> pd.DataFrame:
    f = paths.SPLITS_DIR / "train_folds.parquet"
    if not f.exists():
        make_splits()
    return pd.read_parquet(f)


def load_dev_s1(split: str, legacy_random: bool = False) -> pd.DataFrame:
    """DEV S1 rows (entity_id, country, state, city). legacy_random=True gives the old 5% random sample."""
    if split == "train":
        f = load_folds()
        return f[f.dev_random if legacy_random else f.dev][["entity_id", "country", "state", "city"]]
    if legacy_random:
        raise ValueError("legacy random DEV exists only for train")
    p = paths.SPLITS_DIR / "test_dev.parquet"
    if not p.exists():
        make_splits()
    return pd.read_parquet(p)


def dev_weights(split: str = "train") -> pd.Series:
    """Inverse-sampling-rate weight per DEV S1 (index entity_id): N(country, tier) / n_DEV(country, tier).

    Geographic DEV is a stratified cluster sample whose country and tier shares differ from the population
    (India is 31% of DEV vs 40% of train, and one metro stands in for all metros). A weighted macro F0.5 with
    these weights estimates the full-data score; the unweighted score does not. S1s without a parsed city
    (<2%) belong to no tier and are not represented."""
    f = load_folds() if split == "train" else None
    if f is None:
        raise ValueError("weights need the full S1 table; only train is stored")
    pop = f.dropna(subset=["tier"]).groupby(["country", "tier"]).size()
    dev = f[f.dev].groupby(["country", "tier"]).size()
    rate = (pop / dev).rename("w")
    d = f[f.dev][["entity_id", "country", "tier"]].merge(rate.reset_index(), on=["country", "tier"], how="left")
    return d.set_index("entity_id").w


def load_dev_ids(split: str) -> set:
    return set(load_dev_s1(split).entity_id)


def kfold(ids=None):
    """Yield (fold, train_ids, eval_ids) for the 5 folds, optionally restricted to `ids` (e.g. dev)."""
    f = load_folds()
    if ids is not None:
        f = f[f.entity_id.isin(set(ids))]
    for k in range(N_FOLDS):
        m = f.fold == k
        yield k, set(f.entity_id[~m]), set(f.entity_id[m])


CROSS_COUNTRY = [("US", "India"), ("India", "US")]


def cross_country(ids=None):
    """Yield (name, train_ids, eval_ids): fit on one country, evaluate on the other.
    This is a proxy for transfer to the unlabelled France test data."""
    f = load_folds()
    if ids is not None:
        f = f[f.entity_id.isin(set(ids))]
    for src, dst in CROSS_COUNTRY:
        yield f"{src}->{dst}", set(f.entity_id[f.country == src]), set(f.entity_id[f.country == dst])


if __name__ == "__main__":
    make_splits()
    f = load_folds()
    print(pd.crosstab([f.country, f.n_true.clip(upper=SIZE_CLIP)], f.fold, margins=True).to_string())
    for split in ("train", "test"):
        c = pd.read_parquet(paths.SPLITS_DIR / f"dev_cities_{split}.parquet")
        print(f"\n[{split}] sampled cities per (country, tier):")
        print(c.groupby(["country", "tier"]).agg(cities=("city", "size"), s1=("n_s1", "sum")).to_string())
        print("metros drawn:", c[c.tier == "metro"][["country", "city", "n_s1"]].to_dict("records"))
    d = f[f.dev]
    print(f"\ntrain DEV S1: {len(d):,} ({len(d) / len(f):.2%}); by country:",
          {k: f"{v:,} ({v / (f.country == k).sum():.1%})" for k, v in d.country.value_counts().items()})
    print("DEV per fold:", d.fold.value_counts().sort_index().to_dict())
    print("DEV singleton share: %.4f (all train %.4f)" % ((d.n_true == 0).mean(), (f.n_true == 0).mean()))
    t = pd.read_parquet(paths.SPLITS_DIR / "test_dev.parquet")
    print(f"test DEV S1: {len(t):,}", t.country.value_counts().to_dict())
