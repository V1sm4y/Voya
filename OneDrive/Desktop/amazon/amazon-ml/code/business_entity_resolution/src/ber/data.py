"""Raw-data access: TSV -> parquet cache, and sample-aware loading (--sample dev|full)."""
import csv
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import paths

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]
SPLITS = ("train", "test")
SAMPLES = ("dev", "full")


def read_tsv(path) -> pd.DataFrame:
    """Read a challenge TSV exactly: all strings, empty stays '', no quote handling."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, engine="c")


def _names(split: str) -> List[str]:
    names = [f"{split}_source{s}" for s in (1, 2, 3)]
    return names + (["train_ground_truth"] if split == "train" else [])


def build_cache(force: bool = False, verbose: bool = True) -> None:
    """Convert every dataset TSV to data/cache/<name>.parquet (row counts checked against raw lines)."""
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for name in _names(split):
            src = paths.DATA_DIR / split / f"{name}.tsv"
            dst = paths.CACHE_DIR / f"{name}.parquet"
            if dst.exists() and not force:
                continue
            t = time.time()
            df = read_tsv(src)
            with open(src, encoding="utf-8") as f:
                n_lines = sum(1 for _ in f) - 1
            if len(df) != n_lines:
                raise ValueError(f"{src}: parsed {len(df)} rows but file has {n_lines} data lines")
            df.to_parquet(dst, index=False)
            if verbose:
                print(f"cached {name}: {len(df):,} rows in {time.time() - t:.0f}s", flush=True)


def _read_cached(name: str, columns=None) -> pd.DataFrame:
    path = paths.CACHE_DIR / f"{name}.parquet"
    if not path.exists():
        build_cache()
    return pd.read_parquet(path, columns=columns)


def parse_id_list(s: str) -> List[str]:
    return [i for i in s.split(",") if i] if s else []


def load_truth() -> Dict[str, List[str]]:
    """Train ground truth as {s1_id: [s2/s3 ids]} (empty list = singleton)."""
    gt = _read_cached("train_ground_truth")
    return dict(zip(gt.source1_entity_id, map(parse_id_list, gt.matched_entity_ids)))


@dataclass
class Bundle:
    split: str
    sample: str
    s1: pd.DataFrame
    s2: pd.DataFrame
    s3: pd.DataFrame
    truth: Optional[Dict[str, List[str]]]  # restricted to s1 ids; None for test

    @property
    def targets(self) -> pd.DataFrame:
        """S2 and S3 stacked (entity_id prefix tells the source)."""
        return pd.concat([self.s2, self.s3], ignore_index=True)

    def describe(self) -> str:
        return (f"[{self.split}/{self.sample}] S1 {len(self.s1):,} | S2 {len(self.s2):,} | "
                f"S3 {len(self.s3):,} | countries {sorted(self.s1.country.unique())}")


def load(split: str, sample: str) -> Bundle:
    """Load one split.

    sample='full'  every record.
    sample='dev'   geographic DEV (splits.py): S1 = every S1 in ~5% of cities (stratified by country and
                   city-size tier); pool = every S2/S3 record in the same (country, state group) as those
                   S1s, plus the records whose state is unknown.
    sample='dev_random'  legacy 5% random S1 sample plus the full S2/S3 of its countries. Internal only,
                   for comparing against the old DEV; the CLI does not offer it.
    """
    if split not in SPLITS or sample not in SAMPLES + ("dev_random",):
        raise ValueError(f"bad split/sample: {split}/{sample}")
    s1 = _read_cached(f"{split}_source1")
    s2 = _read_cached(f"{split}_source2")
    s3 = _read_cached(f"{split}_source3")
    if sample != "full":
        from .splits import load_dev_s1, state_group
        dev = load_dev_s1(split, legacy_random=sample == "dev_random")
        s1 = s1[s1.entity_id.isin(set(dev.entity_id))].reset_index(drop=True)
        countries = set(dev.country)
        if sample == "dev":
            from .geo import load_geo
            keep_states = set(zip(dev.country, state_group(dev.state)))
            s2, s3 = (_pool_by_state(s, load_geo(split, n), keep_states) for n, s in ((2, s2), (3, s3)))
        else:
            s2 = s2[s2.country.isin(countries)].reset_index(drop=True)
            s3 = s3[s3.country.isin(countries)].reset_index(drop=True)
    truth = None
    if split == "train":
        full = load_truth()
        truth = {i: full[i] for i in s1.entity_id}
    return Bundle(split, sample, s1, s2, s3, truth)


def _pool_by_state(df: pd.DataFrame, geo: pd.DataFrame, keep: set) -> pd.DataFrame:
    assert (df.entity_id.values == geo.entity_id.values).all(), "geo cache out of sync with source cache"
    return df[_state_mask(geo, keep)].reset_index(drop=True)


def _state_mask(geo: pd.DataFrame, keep: set) -> np.ndarray:
    from .splits import state_group
    countries = {k[0] for k in keep}
    grp = state_group(geo.state)
    known = set(keep)
    return np.fromiter(((c, st) in known or ((st is None or st != st) and c in countries)
                        for c, st in zip(geo.country, grp)), dtype=bool, count=len(geo))


def sample_masks(split: str, sample: str, extra_s1=None) -> dict:
    """Row masks aligned with the cache order of each source (raw, geo and prep tables share it).

    Returns {1: mask, 2: mask, 3: mask}. The rules are the same as load(): 'dev' = geographic DEV,
    'full' = everything. `extra_s1` adds S1 ids (e.g. an evaluation probe) and widens the S2/S3
    pool to their state groups."""
    from .geo import load_geo
    from .splits import load_dev_s1, state_group
    g1 = load_geo(split, 1)
    if sample == "full":
        m1 = np.ones(len(g1), bool)
    else:
        m1 = g1.entity_id.isin(set(load_dev_s1(split).entity_id)).values
    if extra_s1 is not None:
        m1 |= g1.entity_id.isin(set(extra_s1)).values
    out = {1: m1}
    if sample == "full":
        for n in (2, 3):
            out[n] = np.ones(len(load_geo(split, n)), bool)
    else:
        sel = g1[m1]
        keep = set(zip(sel.country, state_group(sel.state)))
        for n in (2, 3):
            out[n] = _state_mask(load_geo(split, n), keep)
    return out
