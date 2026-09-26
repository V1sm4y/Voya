"""Normalised and parsed record tables, cached by a hash of the normalisation code.

`prep_table(split, source)` returns one row per record, in the same order as the raw cache,
with the name fields (names.parse_name), the address fields (address.parse_address) and the
derived blocking strings below.

The cache lives at data/cache/prep/<code-hash>/<split>_source<n>.parquet. The hash covers the
source of every module that affects the output (CODE_FILES), so editing the normalisation code
rebuilds the tables on next use, and nothing else does. Building one split (~12M records) takes
a few minutes on all cores.

Derived blocking strings:
  name_tx    core_name romanised, lower-case ASCII, doubled vowels collapsed ("raam" -> "ram"),
             so Indic and Latin spellings share character n-grams
  alias_tx / alias_key  the same (and the phonetic key) for the S3 alias (f/k/a, t/a ...)
  places     '|'-joined canonical place names in the address: the city plus admin units and plain
             localities. It is used to test "same city" when the target puts the town, the district,
             or both in different slots.
  addr_tx    street-level address text without city or state: premise numbers + street + locality +
             building, used for address-only TF-IDF
"""
import hashlib
import json
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import paths
from .address import canon_city, parse_address
from .names import parse_name
from .normalize import basic
from .translit import phonetic, romanize

CODE_FILES = ["normalize.py", "translit.py", "names.py", "address.py", "prep.py"]
NAME_COLS = ["core_name", "core_key", "legal_form", "alias"]
ADDR_COLS = ["state", "city", "locality", "street", "building", "premise_no", "premise_typed", "unit", "floor",
             "landmark", "admin", "careof", "pin", "locality_key", "street_key", "building_key"]
DERIVED = ["name_tx", "alias_tx", "alias_key", "places", "addr_tx"]
COLUMNS = ["entity_id", "country"] + NAME_COLS + ADDR_COLS + DERIVED
_BATCH = 500_000
_CHUNK = 10_000
_QUALIFIER = re.compile(r"^(sector|phase|block|stage|pocket|ward|zone|extension)\b")


def code_hash() -> str:
    h = hashlib.sha256()
    here = Path(__file__).parent
    for f in CODE_FILES:
        h.update(f.encode())
        h.update((here / f).read_bytes())
    return h.hexdigest()[:12]


def prep_dir() -> Path:
    return paths.CACHE_DIR / "prep" / code_hash()


def translit_text(s) -> str:
    if not s:
        return ""
    t = basic(romanize(s))
    return re.sub(r"([aeiou])\1+", r"\1", t)


def _places(addr: dict, country: str) -> str:
    vals = []
    if addr["city"]:
        vals.append(addr["city"])
    for f in ("admin", "locality"):
        for v in (addr[f] or "").split("|"):
            if v and not _QUALIFIER.match(v) and not re.search(r"\d", v):
                c = canon_city(v, country)
                if c:
                    vals.append(c)
    return "|".join(dict.fromkeys(vals))


def _addr_tx(addr: dict) -> str:
    parts = (addr["premise_no"] or "").split("|")[:2] + [addr["street"], addr["locality"], addr["building"]]
    return translit_text(" ".join(p.replace("|", " ") for p in parts if p))


def prep_record(name: str, address: str, country: str) -> tuple:
    n = parse_name(name)
    a = parse_address(address, country)
    alias_tx = translit_text(n["alias"])
    derived = (translit_text(n["core_name"]), alias_tx or None,
               phonetic(n["alias"]) if n["alias"] else None, _places(a, country) or None, _addr_tx(a) or None)
    return tuple(n[c] for c in NAME_COLS) + tuple(a[c] for c in ADDR_COLS) + derived


def _chunk(args):
    names, addrs, countries = args
    return [prep_record(n, a, c) for n, a, c in zip(names, addrs, countries)]


def _build(split: str, source: int, dst: Path, procs: int) -> None:
    from .data import _read_cached
    raw = _read_cached(f"{split}_source{source}")
    schema = pa.schema([(c, pa.string()) for c in COLUMNS])
    tmp = dst.with_suffix(".tmp")
    t = time.time()
    with pq.ParquetWriter(tmp, schema) as w, Pool(procs) as pool:
        for start in range(0, len(raw), _BATCH):
            b = raw.iloc[start:start + _BATCH]
            n, a, c = b.business_name.tolist(), b.business_address.tolist(), b.country.tolist()
            chunks = [(n[i:i + _CHUNK], a[i:i + _CHUNK], c[i:i + _CHUNK]) for i in range(0, len(n), _CHUNK)]
            rows = [r for part in pool.imap(_chunk, chunks) for r in part]
            cols = {"entity_id": b.entity_id.tolist(), "country": c}
            for j, col in enumerate(COLUMNS[2:]):
                cols[col] = [r[j] for r in rows]
            w.write_table(pa.table(cols, schema=schema))
    tmp.rename(dst)
    print(f"  prep {split}_source{source}: {len(raw):,} rows in {time.time() - t:.0f}s", flush=True)


def build(splits=("train", "test"), procs: int = None, force: bool = False) -> Path:
    d = prep_dir()
    d.mkdir(parents=True, exist_ok=True)
    procs = procs or max(1, (os.cpu_count() or 2) - 1)
    for split in splits:
        for s in (1, 2, 3):
            dst = d / f"{split}_source{s}.parquet"
            if force or not dst.exists():
                _build(split, s, dst, procs)
    (d / "manifest.json").write_text(json.dumps({"hash": code_hash(), "files": CODE_FILES}, indent=1))
    return d


def prep_table(split: str, source: int, columns=None) -> pd.DataFrame:
    """Cached prep table (built on first use for the current code hash). Strings are pyarrow-backed."""
    p = prep_dir() / f"{split}_source{source}.parquet"
    if not p.exists():
        print(f"prep cache miss for hash {code_hash()} -> building {split}", flush=True)
        build(splits=(split,))
    return pd.read_parquet(p, columns=columns, dtype_backend="pyarrow")


def stale_dirs():
    root = paths.CACHE_DIR / "prep"
    cur = code_hash()
    return [p for p in root.iterdir() if p.is_dir() and p.name != cur] if root.exists() else []


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the hash-keyed prep cache")
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prune", action="store_true", help="delete prep dirs of older code hashes")
    a = ap.parse_args()
    print("code hash:", code_hash())
    build(splits=("train", "test") if a.split == "both" else (a.split,), force=a.force)
    if a.prune:
        import shutil
        for p in stale_dirs():
            shutil.rmtree(p)
            print("pruned", p)
    elif stale_dirs():
        print("stale prep dirs (use --prune):", [p.name for p in stale_dirs()])
