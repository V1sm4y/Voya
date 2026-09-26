"""Per-record (state, city) for every source file, cached at data/cache/geo_<split>_source<n>.parquet.

It uses address.geo (the fast path of the address parser). When a record has no parseable state
but has a city, the state is inferred from S1 of the same split and country: the most common state
among S1 records with that city. S1 is fully parsed (100% have a state), so this is in-data
inference, not an external gazetteer. Records whose state is still unknown have state=None, and
DEV pools include them (see data.load).
"""
import os
from multiprocessing import Pool

import pandas as pd

from . import paths
from .address import geo

_CHUNK = 20_000


def _geo_chunk(args):
    addrs, countries = args
    return [geo(a, c) for a, c in zip(addrs, countries)]


def compute_geo(df: pd.DataFrame, procs: int = None) -> pd.DataFrame:
    procs = procs or max(1, (os.cpu_count() or 2) - 1)
    a, c = df.business_address.tolist(), df.country.tolist()
    chunks = [(a[i:i + _CHUNK], c[i:i + _CHUNK]) for i in range(0, len(a), _CHUNK)]
    with Pool(procs) as pool:
        res = [x for part in pool.imap(_geo_chunk, chunks) for x in part]
    return pd.DataFrame({"entity_id": df.entity_id.values, "country": df.country.values,
                         "state": [r[0] for r in res], "city": [r[1] for r in res]})


def geo_path(split: str, source: int):
    return paths.CACHE_DIR / f"geo_{split}_source{source}.parquet"


def build_geo(split: str, force: bool = False, verbose: bool = True) -> None:
    from .data import _read_cached
    if all(geo_path(split, s).exists() for s in (1, 2, 3)) and not force:
        return
    g1 = compute_geo(_read_cached(f"{split}_source1"))
    city_state = (g1.dropna(subset=["city", "state"]).groupby(["country", "city"]).state
                  .agg(lambda s: s.value_counts().index[0]))
    for s, g in [(1, g1)] + [(s, None) for s in (2, 3)]:
        if g is None:
            g = compute_geo(_read_cached(f"{split}_source{s}"))
        miss = g.state.isna() & g.city.notna()
        key = pd.MultiIndex.from_arrays([g.country[miss], g.city[miss]])
        g.loc[miss, "state"] = city_state.reindex(key).values
        g["state_inferred"] = miss & g.state.notna()
        g.to_parquet(geo_path(split, s), index=False)
        if verbose:
            print(f"geo {split}_source{s}: {len(g):,} rows | state parsed {(g.state.notna() & ~g.state_inferred).mean():.1%}"
                  f" + inferred {g.state_inferred.mean():.1%} | unknown {g.state.isna().mean():.1%}"
                  f" | city {g.city.notna().mean():.1%}", flush=True)


def load_geo(split: str, source: int) -> pd.DataFrame:
    if not geo_path(split, source).exists():
        build_geo(split)
    return pd.read_parquet(geo_path(split, source))


if __name__ == "__main__":
    import sys
    for sp in ("train", "test"):
        build_geo(sp, force="--force" in sys.argv)
