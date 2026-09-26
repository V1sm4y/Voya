"""Candidate generation (blocking). Every blocker runs inside one country; no cross-country pairs.

Candidates = UNION of these blockers (bit flags):
  name_tfidf      char 3-gram TF-IDF on the transliterated core name (prep.name_tx); top-K neighbours per
                  S1 in S2 and in S3 separately, within (country, state group). Targets with an unknown state
                  are searched from every state group.
  rare_token      (place, rare core-name token): the S1's 2 rarest tokens (target doc-freq <= RARE_DF)
  phonetic_city   (place, phonetic key of the whole core name): cross-script / spelling-variant names
  alias           S3 aliases (f/k/a, t/a, ...) used as extra names in the three name blockers above
  addr_key        (place, premise number, street key): house number + street + city
  addr_tfidf      char 3-gram TF-IDF top-K on the street-level address alone (no city/state, numbers kept as
                  soft tokens). Catches matches whose names are unrelated.
  in_loc_name     India: (place, locality key, core-name phonetic)
  in_loc_premise  India: (place, locality key, premise number)
  name_phon_tfidf char 3-4-gram TF-IDF on the phonetic skeleton of the core name (translit.phonetic): Indic-script
                  and typo'd names; top-K within (country, state group)
  placeless_name  name TF-IDF against only the targets with no parseable place (mostly empty addresses), per
                  country, so they do not compete with located records for the top-K slots
  place_premise   (place, premise number) with a posting cap of 1000, kept only if the names are loosely similar
                  (token_set_ratio >= 50 on name_tx, or ratio >= 60 on the skeleton; rapidfuzz cpdist, vectorised).
                  Catches typo'd names at number-only addresses ("H.no 35, Karnal, HR").
  placeless_token (country, rare token) against targets with no parseable place. Char n-grams rank a typo'd
                  rare word below intact common ones ("Ballard Atnioiy" loses to ten "* Antimony" names), but
                  the other rare token ("ballard") still matches exactly.
  state_keys      city-substitution safety net within (country, state group), cap 25: rarest token
                  (df <= 500), house number + street key, and the whole-name phonetic key
                  (Stanford -> Stanfordville, Brookhaven -> Miller Place)

The top-K blockers keep each pair's rank (0 = best), so recall and candidate counts for any K <= k_max can
be read from one run (--sweep).

"place" means any canonical place in the record (prep.places: city + admin units + plain localities).
Every key that contains a premise/house number has a number-free backup (addr_tfidf, in_loc_name,
phonetic_city), because numbers are often truncated or padded (3424 -> 424). Exact-key postings with
more than CAP targets are skipped, and the finer India keys cover the common-name cases that the cap drops.

Scalability: sparse TF-IDF with sparse_dot_topn (multithreaded, chunked top-K), exact keys hashed to
uint64 and joined with sorted searchsorted ranges, and the union built as one np.unique over packed pair
ids with bitwise-OR of the flags. There are no Python loops over pairs.

Usage (from src/):  python -m ber.blocking --sample dev [--split test]
"""
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .prep import prep_table
from .splits import state_group

BLOCKERS = ["name_tfidf", "rare_token", "phonetic_city", "alias", "addr_key", "addr_tfidf", "in_loc_name",
            "in_loc_premise", "name_phon_tfidf", "placeless_name", "place_premise", "placeless_token", "state_keys"]
RANKED = ["name_tfidf", "name_phon_tfidf", "addr_tfidf", "placeless_name"]   # top-K blockers (rank kept)
BIT = {b: np.uint16(1 << i) for i, b in enumerate(BLOCKERS)}


@dataclass
class Config:
    k_name: int = 15          # top-K per S1 per source for name TF-IDF
    k_addr: int = 10          # top-K per S1 per source for address TF-IDF
    k_alias: int = 1          # top-K per S1 among S3 alias strings
    k_phon: int = 3           # top-K per S1 per source for phonetic-skeleton TF-IDF
    k_placeless: int = 2      # top-K per S1 per source among targets with no parseable place
    # defaults from blocking_tune on DEV (reports/blocking_tune_train_dev.md). k_name 10 -> 15 because full
    # train at 10 gave 0.9798 (DEV is optimistic for India); DEV Pareto point 0.9836 at 65 cands/S1.
    k_max: int = 0            # >0: generate top-k_max for every ranked blocker (K sweep); selection by k_* later
    premise_cap: int = 1000   # posting cap for (place, premise) before the name filter
    state_cap: int = 25       # posting cap for the state-level keys
    placeless_cap: int = 20   # posting cap for the placeless token key (1 rarest token per S1)
    state_rare_df: int = 500  # stricter rarity for the state-level token key
    premise_name_sim: float = 50   # token_set_ratio on name_tx ...
    premise_phon_sim: float = 60   # ... or ratio on the phonetic skeleton
    tfidf_threshold: float = 0.1
    tfidf_max_df: float = 0.05
    rare_df: int = 2000       # a token is "rare" if at most this many targets in the country carry it
    rare_per_s1: int = 2
    cap: int = 300            # max targets per exact-key posting
    threads: int = max(1, (os.cpu_count() or 2) - 1)


S1_COLS = ["entity_id", "country", "state", "places", "name_tx", "core_key", "addr_tx", "premise_no",
           "street_key", "locality_key"]
TG_COLS = S1_COLS + ["alias_tx", "alias_key"]


def load_side(split: str, masks: dict):
    """Prep rows for the sample: S1 frame and stacked S2+S3 targets (src column 2/3)."""
    s1 = prep_table(split, 1, columns=S1_COLS)[masks[1]].reset_index(drop=True)
    parts = []
    for n in (2, 3):
        t = prep_table(split, n, columns=TG_COLS)[masks[n]].reset_index(drop=True)
        t["src"] = np.int8(n)
        parts.append(t)
    tg = pd.concat(parts, ignore_index=True)
    for df in (s1, tg):
        st = df.state.astype(object)
        df["sgroup"] = state_group(st.where(st.notna(), None)).astype(object)
        df["sgroup"] = df.sgroup.where(df.sgroup.notna(), None)   # plain None, never pd.NA
    return s1, tg


# ------------------------------------------------------------------ exact keys
_MIX = np.uint64(0x9E3779B97F4A7C15)


def _combos(idx: np.ndarray, cols) -> tuple:
    """For each row, the cartesian product of the '|'-separated values of `cols` (all aligned with idx);
    returns (row idx, uint64 key) per combination.

    Memory-safe: each column is exploded and hashed to uint64 on its own, and the per-column hashes are
    combined arithmetically. The concatenated key strings are never built (they reached tens of millions
    of Python strings on India full and pushed the run into swap)."""
    cur = None
    for j, c in enumerate(cols):
        vals = pd.Series(np.asarray(c.astype(object)))
        ex = vals.dropna().str.split("|").explode()
        ex = ex[ex.notna() & (ex != "")]
        if ex.empty:
            return np.empty(0, np.int64), np.empty(0, np.uint64)
        part = pd.DataFrame({"r": ex.index.to_numpy(np.int64),
                             "h": pd.util.hash_array(ex.to_numpy(dtype=object))})
        del ex, vals
        if cur is None:
            cur = part
        else:
            cur = cur.merge(part, on="r", suffixes=("", "_n"))      # per-row cartesian product
            with np.errstate(over="ignore"):
                cur["h"] = (cur.h.to_numpy() * _MIX) ^ (cur.h_n.to_numpy() + np.uint64(j))
            cur = cur.drop(columns="h_n")
        if cur.empty:
            return np.empty(0, np.int64), np.empty(0, np.uint64)
    return np.asarray(idx)[cur.r.to_numpy()].astype(np.int64), cur.h.to_numpy(np.uint64)


def _key_join(s_i, s_h, t_i, t_h, cap: int):
    """All (s, t) with equal keys, skipping keys whose target posting exceeds `cap`. Vectorised."""
    if len(s_h) == 0 or len(t_h) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    m = np.isin(t_h, np.unique(s_h))
    t_i, t_h = t_i[m], t_h[m]
    if len(t_h) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    order = np.argsort(t_h, kind="stable")
    t_i, t_h = t_i[order], t_h[order]
    u, start, cnt = np.unique(t_h, return_index=True, return_counts=True)
    keep = cnt <= cap
    pos = np.searchsorted(u, s_h)
    pos = np.clip(pos, 0, len(u) - 1)
    hit = (u[pos] == s_h) & keep[pos]
    s_i, lo, n = s_i[hit], start[pos[hit]], cnt[pos[hit]]
    rep_s = np.repeat(s_i, n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    return rep_s, t_i[np.repeat(lo, n) + offs]


def _rare_tokens(s1c, tgc, cfg, scope_s="places", scope_t="places", max_df=None, per_s1=None, cap=None, df_from=None):
    """(scope, rare token) keys; S1 keeps its `per_s1` rarest tokens. Doc-freq is over `df_from` (default: tgc)."""
    max_df, per_s1, cap = max_df or cfg.rare_df, per_s1 or cfg.rare_per_s1, cap or cfg.cap
    if df_from is not None:
        dfx = df_from.name_tx.astype(object).str.split().explode().dropna()
        dfx = dfx[dfx.str.len() >= 3].reset_index().drop_duplicates()
        df_all = dfx.iloc[:, 1].value_counts()
    ex = tgc.name_tx.astype(object).str.split().explode().dropna()
    ex = ex[ex.str.len() >= 3].reset_index().drop_duplicates()
    ex.columns = ["i", "tok"]
    df = ex.tok.value_counts() if df_from is None else df_all
    rare = df[df <= max_df]
    ex = ex[ex.tok.isin(rare.index)]
    sx = s1c.name_tx.astype(object).str.split().explode().dropna().reset_index().drop_duplicates()
    sx.columns = ["i", "tok"]
    sx = sx[sx.tok.isin(rare.index)]
    sx["df"] = sx.tok.map(rare).values
    sx = sx.sort_values(["i", "df"]).groupby("i").head(per_s1)
    if sx.empty or ex.empty:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    s_tok = sx.groupby("i").tok.agg("|".join)
    t_tok = ex.groupby("i").tok.agg("|".join)
    s_i, s_h = _combos(s_tok.index.values, [s1c[scope_s].loc[s_tok.index], s_tok])
    t_i, t_h = _combos(t_tok.index.values, [tgc[scope_t].loc[t_tok.index], t_tok])
    return _key_join(s_i, s_h, t_i, t_h, cap)


def _exact(s1c, tgc, s_cols, t_cols, cfg, cap=None):
    s_i, s_h = _combos(s1c.index.values, [s1c[c] for c in s_cols])
    t_i, t_h = _combos(tgc.index.values, [tgc[c] for c in t_cols])
    return _key_join(s_i, s_h, t_i, t_h, cap or cfg.cap)


def _place_premise(s1c, tgc, cfg):
    """(place, premise) key join, then a vectorised loose-name filter."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    s_i, s_h = _combos(s1c.index.values, [s1c.places, s1c.premise_no])
    t_i, t_h = _combos(tgc.index.values, [tgc.places, tgc.premise_no])
    a, b = _key_join(s_i, s_h, t_i, t_h, cfg.premise_cap)
    if len(a) == 0:
        return a, b
    u = np.unique(a * np.int64(1 << 32) + b)
    a, b = u >> 32, u & ((1 << 32) - 1)
    sn = s1c.name_tx.astype(object).fillna("").reindex(a).tolist()
    tn = tgc.name_tx.astype(object).fillna("").reindex(b).tolist()
    sk = s1c.core_key.astype(object).fillna("").reindex(a).tolist()
    tk = tgc.core_key.astype(object).fillna("").reindex(b).tolist()
    ok = (cpdist(sn, tn, scorer=fuzz.token_set_ratio, workers=-1) >= cfg.premise_name_sim) | \
         (cpdist(sk, tk, scorer=fuzz.ratio, workers=-1) >= cfg.premise_phon_sim)
    return a[ok], b[ok]


def _first_premise(s: pd.Series) -> pd.Series:
    return s.astype(object).fillna("").str.split("|").str[0].replace("", None)


# ------------------------------------------------------------------ TF-IDF top-K
def _tfidf_topk(s_text: pd.Series, s_grp: pd.Series, t_text: pd.Series, t_grp: pd.Series, t_src: pd.Series,
                k: int, cfg: Config, ngram=(3, 3)):
    """Top-k per S1 row among targets of each source, within the S1's state group (+ unknown-state targets).
    Returns (s rows, t rows, rank) with rank 0 = most similar."""
    s_text = s_text.astype(object).fillna("")
    t_text = t_text.astype(object).fillna("")
    s_ok = (s_text.str.len() > 0).to_numpy()
    t_ok = (t_text.str.len() > 0).to_numpy()
    if not s_ok.any() or not t_ok.any():
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.uint8)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=2, max_df=cfg.tfidf_max_df,
                          sublinear_tf=True, dtype=np.float32)
    T = vec.fit_transform(t_text[t_ok].tolist())
    S = vec.transform(s_text[s_ok].tolist())
    s_rows = np.flatnonzero(s_ok)
    t_rows = np.flatnonzero(t_ok)
    sg = s_grp.astype(object).to_numpy()[s_rows]
    tg = t_grp.astype(object).to_numpy()[t_rows]
    tsrc = t_src.to_numpy()[t_rows]
    t_unknown = pd.isna(tg)
    out_s, out_t, out_r = [], [], []
    for g in pd.unique(sg):
        si = np.flatnonzero(sg == g)
        for src in np.unique(tsrc):
            ti = np.flatnonzero(((tg == g) | t_unknown) & (tsrc == src))
            if len(ti) == 0:
                continue
            C = sp_matmul_topn(S[si], T[ti].T.tocsr(), top_n=k, threshold=cfg.tfidf_threshold, sort=True,
                               n_threads=cfg.threads)
            nnz_row = np.diff(C.indptr)
            rows = np.repeat(np.arange(C.shape[0]), nnz_row)
            rank = np.arange(C.nnz) - np.repeat(C.indptr[:-1], nnz_row)
            out_s.append(s_rows[si[rows]])
            out_t.append(t_rows[ti[C.indices]])
            out_r.append(rank)
    if not out_s:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.uint8)
    return (np.concatenate(out_s).astype(np.int64), np.concatenate(out_t).astype(np.int64),
            np.minimum(np.concatenate(out_r), 254).astype(np.uint8))


# ------------------------------------------------------------------ driver
def generate(s1: pd.DataFrame, tg: pd.DataFrame, cfg: Config = Config(), verbose: bool = True) -> pd.DataFrame:
    """Candidate pairs, one row per unique (s1 row, target row): bits (uint16 blocker flags) and, per ranked
    blocker, rank_<name> (uint8, 255 = not produced by it)."""
    all_s, all_t, all_b, all_r = [], [], [], []
    per_country = []

    def add(name, s_i, t_i, t0, rank=None):
        all_s.append(np.asarray(s_i, np.int32)); all_t.append(np.asarray(t_i, np.int32))
        all_b.append(np.full(len(s_i), BIT[name], np.uint16))
        all_r.append(np.zeros(len(s_i), np.uint8) if rank is None else rank)
        if verbose:
            print(f"    {name:15s} {len(s_i):>12,} pairs  {time.time() - t0:6.1f}s", flush=True)

    k = (lambda kk: max(kk, cfg.k_max))
    for country in sorted(s1.country.astype(object).unique()):
        s1c = s1[s1.country == country]
        tgc = tg[tg.country == country]
        if verbose:
            print(f"  [{country}] S1 {len(s1c):,}  targets {len(tgc):,}", flush=True)
        t0 = time.time()
        s_i, t_i, r = _tfidf_topk(s1c.name_tx, s1c.sgroup, tgc.name_tx, tgc.sgroup, tgc.src, k(cfg.k_name), cfg)
        add("name_tfidf", s1c.index.values[s_i], tgc.index.values[t_i], t0, r)

        t0 = time.time()
        s_i, t_i, r = _tfidf_topk(s1c.core_key, s1c.sgroup, tgc.core_key, tgc.sgroup, tgc.src, k(cfg.k_phon), cfg,
                                  ngram=(3, 4))
        add("name_phon_tfidf", s1c.index.values[s_i], tgc.index.values[t_i], t0, r)

        t0 = time.time()
        pl = tgc[tgc.places.isna()]
        one = pd.Series("all", index=s1c.index)
        s_i, t_i, r = _tfidf_topk(s1c.name_tx, one, pl.name_tx, pd.Series("all", index=pl.index), pl.src,
                                  k(cfg.k_placeless), cfg)
        add("placeless_name", s1c.index.values[s_i], pl.index.values[t_i], t0, r)

        t0 = time.time()
        s_i, t_i = _rare_tokens(s1c, tgc, cfg)
        add("rare_token", s_i, t_i, t0)

        t0 = time.time()
        plc = pl.assign(scope="all")
        s_i, t_i = _rare_tokens(s1c.assign(scope="all"), plc, cfg, "scope", "scope", per_s1=1, cap=cfg.placeless_cap,
                                df_from=tgc)
        add("placeless_token", s_i, t_i, t0)

        t0 = time.time()
        s1g = s1c.assign(p1=_first_premise(s1c.premise_no))
        parts = [_rare_tokens(s1g, tgc, cfg, "sgroup", "sgroup", max_df=cfg.state_rare_df, per_s1=1,
                              cap=cfg.state_cap),
                 _exact(s1g, tgc, ["sgroup", "p1", "street_key"], ["sgroup", "premise_no", "street_key"], cfg,
                        cfg.state_cap),
                 _exact(s1g, tgc, ["sgroup", "core_key"], ["sgroup", "core_key"], cfg, cfg.state_cap)]
        add("state_keys", np.concatenate([x[0] for x in parts]), np.concatenate([x[1] for x in parts]), t0)

        t0 = time.time()
        s_i, t_i = _exact(s1c, tgc, ["places", "core_key"], ["places", "core_key"], cfg)
        add("phonetic_city", s_i, t_i, t0)

        # aliases (S3): TF-IDF among alias strings + phonetic/place key on the alias
        t0 = time.time()
        al = tgc[tgc.alias_tx.notna()]
        sa, ta = [np.empty(0, np.int64)], [np.empty(0, np.int64)]
        if len(al):
            s_i, t_i, _ = _tfidf_topk(s1c.name_tx, s1c.sgroup, al.alias_tx, al.sgroup, al.src, cfg.k_alias, cfg)
            sa.append(s1c.index.values[s_i]); ta.append(al.index.values[t_i])
            s_i, t_i = _exact(s1c, al, ["places", "core_key"], ["places", "alias_key"], cfg)
            sa.append(s_i); ta.append(t_i)
        add("alias", np.concatenate(sa), np.concatenate(ta), t0)

        t0 = time.time()
        s1p = s1c.assign(p1=_first_premise(s1c.premise_no))
        s_i, t_i = _exact(s1p, tgc, ["places", "p1", "street_key"], ["places", "premise_no", "street_key"], cfg)
        add("addr_key", s_i, t_i, t0)

        t0 = time.time()
        s_i, t_i = _place_premise(s1c, tgc, cfg)
        add("place_premise", s_i, t_i, t0)

        t0 = time.time()
        s_i, t_i, r = _tfidf_topk(s1c.addr_tx, s1c.sgroup, tgc.addr_tx, tgc.sgroup, tgc.src, k(cfg.k_addr), cfg)
        add("addr_tfidf", s1c.index.values[s_i], tgc.index.values[t_i], t0, r)

        if country == "India":
            t0 = time.time()
            s_i, t_i = _exact(s1c, tgc, ["places", "locality_key", "core_key"], ["places", "locality_key", "core_key"], cfg)
            add("in_loc_name", s_i, t_i, t0)
            t0 = time.time()
            s_i, t_i = _exact(s1c, tgc, ["places", "locality_key", "premise_no"],
                              ["places", "locality_key", "premise_no"], cfg)
            add("in_loc_premise", s_i, t_i, t0)

        per_country.append(_union(all_s, all_t, all_b, all_r, len(tg)))
        all_s.clear(); all_t.clear(); all_b.clear(); all_r.clear()
    out = pd.concat(per_country, ignore_index=True) if per_country else _union([], [], [], [], len(tg))
    return out.sort_values(["s", "t"], kind="stable").reset_index(drop=True) if len(per_country) > 1 else out


def _union(all_s, all_t, all_b, all_r, n_t) -> pd.DataFrame:
    """One row per unique (s, t): OR of blocker bits, min rank per ranked blocker."""
    if not all_s:
        cols = {"s": np.empty(0, np.int32), "t": np.empty(0, np.int32), "bits": np.empty(0, np.uint16)}
        cols.update({f"rank_{n}": np.empty(0, np.uint8) for n in RANKED})
        return pd.DataFrame(cols)
    key = (np.concatenate(all_s).astype(np.int64) << 32) | np.concatenate(all_t).astype(np.int64)
    b = np.concatenate(all_b)
    r = np.concatenate(all_r)
    order = np.argsort(key, kind="stable")
    key, b, r = key[order], b[order], r[order]
    del order
    uk, start = np.unique(key, return_index=True)
    del key
    out = pd.DataFrame({"s": (uk >> 32).astype(np.int32), "t": (uk & 0xFFFFFFFF).astype(np.int32),
                        "bits": np.bitwise_or.reduceat(b, start).astype(np.uint16)})
    for name in RANKED:
        rr = np.where(b == BIT[name], r, np.uint8(255)).astype(np.uint8)
        out[f"rank_{name}"] = np.minimum.reduceat(rr, start)
    return out


def select(cands: pd.DataFrame, cfg: Config, k: dict = None) -> np.ndarray:
    """Mask of candidates kept under per-blocker K (default: cfg.k_*). Unranked blockers are always kept."""
    k = k or {"name_tfidf": cfg.k_name, "name_phon_tfidf": cfg.k_phon, "addr_tfidf": cfg.k_addr,
              "placeless_name": cfg.k_placeless}
    unranked = np.uint16(sum(int(BIT[b]) for b in BLOCKERS if b not in RANKED))
    keep = (cands.bits.to_numpy() & unranked) > 0
    for name in RANKED:
        keep |= cands[f"rank_{name}"].to_numpy() < k[name]
    return keep


# ------------------------------------------------------------------ evaluation & report
def truth_pairs(s1: pd.DataFrame, tg: pd.DataFrame, truth: dict) -> pd.DataFrame:
    """True (s, t) index pairs for the sample's S1; t = -1 when the target is outside the pool."""
    rows = [(i, tid) for i, sid in enumerate(s1.entity_id.astype(object)) for tid in truth.get(sid, ())]
    tp = pd.DataFrame(rows, columns=["s", "tid"])
    tp["t"] = pd.Index(tg.entity_id.astype(object)).get_indexer(tp.tid)
    tp["src"] = tp.tid.str[:2]
    return tp


def score(cands: pd.DataFrame, tp: pd.DataFrame, n_t: int) -> pd.DataFrame:
    """Adds `bits` (0 = missed) to the true pairs. `cands` must be sorted by (s, t), as generate() returns it."""
    ck = cands.s.to_numpy(np.int64) * n_t + cands.t.to_numpy(np.int64)
    tk = tp.s.to_numpy(np.int64) * n_t + tp.t.to_numpy(np.int64)
    pos = np.searchsorted(ck, tk)
    pos = np.clip(pos, 0, max(len(ck) - 1, 0))
    hit = (tp.t.to_numpy() >= 0) & (len(ck) > 0) & (ck[pos] == tk)
    out = tp.copy()
    out["bits"] = np.where(hit, cands.bits.to_numpy()[pos], 0).astype(np.uint16)
    return out


def _recall_table(tpb: pd.DataFrame, by) -> pd.DataFrame:
    g = tpb.assign(hit=tpb.bits > 0).groupby(by)
    return g.agg(true_pairs=("hit", "size"), recall=("hit", "mean"))


def missed_reasons(miss: pd.DataFrame, s1: pd.DataFrame, tg: pd.DataFrame, split: str, masks: dict) -> pd.DataFrame:
    """Missed true pairs with raw text, flags and one 'likely reason'."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    from .data import _read_cached
    from .translit import has_indic
    raw1 = _read_cached(f"{split}_source1")[masks[1]].reset_index(drop=True)
    raws = [ _read_cached(f"{split}_source{n}")[masks[n]].reset_index(drop=True) for n in (2, 3)]
    rawt = pd.concat(raws, ignore_index=True)
    m = miss[miss.t >= 0].copy()
    a, b = raw1.loc[m.s], rawt.loc[m.t]
    sa, tb = s1.loc[m.s], tg.loc[m.t]
    m["s1_name"], m["s1_addr"] = a.business_name.values, a.business_address.values
    m["t_name"], m["t_addr"] = b.business_name.values, b.business_address.values
    m["country"] = sa.country.astype(object).values
    m["name_sim"] = cpdist(sa.name_tx.astype(object).fillna("").tolist(), tb.name_tx.astype(object).fillna("").tolist(),
                           scorer=fuzz.token_set_ratio, workers=-1)
    m["addr_sim"] = cpdist(sa.addr_tx.astype(object).fillna("").tolist(), tb.addr_tx.astype(object).fillna("").tolist(),
                           scorer=fuzz.token_set_ratio, workers=-1)
    m["t_addr_empty"] = m.t_addr.str.strip().eq("").values
    m["t_indic_name"] = [has_indic(x) for x in m.t_name]
    m["same_sgroup"] = (sa.sgroup.values == tb.sgroup.values) | pd.isna(tb.sgroup.values)
    sp_ = sa.places.astype(object).fillna("").str.split("|")
    tp_ = tb.places.astype(object).fillna("").str.split("|")
    m["shares_place"] = [bool(set(x) & set(y) - {""}) for x, y in zip(sp_, tp_)]
    conds = [
        m.t_addr_empty & (m.name_sim >= 80),
        m.t_addr_empty & (m.name_sim < 80),
        ~m.same_sgroup,
        m.t_indic_name & (m.addr_sim < 70),
        (m.name_sim < 50) & (m.addr_sim < 70),
        (m.name_sim < 50),
        (m.name_sim >= 80) & ~m.shares_place,
        (m.name_sim >= 80),
        (m.addr_sim >= 80),
    ]
    labels = ["empty target address, similar name (ranked out)", "empty target address + noisy name",
              "different state/region",
              "Indic-script name + noisy address", "unrelated name + noisy address",
              "unrelated name, address similar but ranked out", "similar name, no shared place",
              "similar name + shared place, ranked out / capped", "similar address, ranked out"]
    m["reason"] = np.select(conds, labels, "noisy name + noisy address")
    m.loc[m.t < 0, "reason"] = "target outside pool"
    return m


def probe_ids(n_per_city: int = 1000, seed: int = 7) -> pd.DataFrame:
    """Random S1 sample from every India metro (train), for metro-level recall outside DEV."""
    from .splits import load_folds
    f = load_folds()
    m = f[(f.country == "India") & (f.tier == "metro") & ~f.dev]
    return m.groupby("city", group_keys=False).apply(
        lambda g: g.sample(min(len(g), n_per_city), random_state=seed))[["entity_id", "city"]]


def _sweep(cands, tp, is_probe, n_s1_main, cfg, n_t, ks=(3, 5, 10, 15, 20, 30)) -> pd.DataFrame:
    """Recall / candidates per S1 when all ranked blockers use the same K (probe S1s excluded)."""
    rows = []
    main_c = ~is_probe[cands.s.to_numpy()]
    for kk in ks:
        if cfg.k_max and kk > cfg.k_max:
            continue
        keep = select(cands, cfg, {n: (min(kk, cfg.k_placeless) if n == "placeless_name" else kk) for n in RANKED})
        c = cands[keep & main_c]
        sc = score(c, tp, n_t)
        r = {"K": kk, "cands_per_s1": len(c) / n_s1_main, "recall": (sc.bits > 0).mean()}
        for ctry in ("India", "US", "France"):
            m = sc.country == ctry
            if m.any():
                r[f"recall {ctry}"] = (sc.bits[m] > 0).mean()
        rows.append(r)
    return pd.DataFrame(rows).set_index("K")


def candidates_path(tag):
    from . import paths
    return paths.CACHE_DIR / "candidates" / f"{tag}.parquet"


def save_candidates(cands: pd.DataFrame, masks: dict, tag: str) -> None:
    """Write candidates with GLOBAL row indices: s = row in <split>_source1, t = row in concat(source2, source3)
    (full tables, raw/prep cache order). Integer ids keep a 150M-pair file around 1-2 GB, where string ids
    would not fit in memory."""
    s_glob = np.flatnonzero(masks[1]).astype(np.int32)
    n2 = len(masks[2])
    t_glob = np.concatenate([np.flatnonzero(masks[2]), n2 + np.flatnonzero(masks[3])]).astype(np.int32)
    out = cands.copy()
    out["s"] = s_glob[cands.s.to_numpy()]
    out["t"] = t_glob[cands.t.to_numpy()]
    p = candidates_path(tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False, compression="zstd")


def load_candidates(tag: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(candidates_path(tag), columns=columns)


def main():
    from . import paths
    from .cli import base_parser
    from .data import load_truth, sample_masks
    from .splits import load_folds
    p = base_parser(__doc__)
    for n in ("name", "phon", "addr", "placeless"):
        p.add_argument(f"--k-{n}", type=int, default=getattr(Config, f"k_{n}"))
    p.add_argument("--sweep", type=int, default=0, help="generate top-K up to this K and report a K sweep")
    p.add_argument("--no-probe", action="store_true", help="skip the India-metro probe S1s on dev")
    a = p.parse_args()
    cfg = Config(k_name=a.k_name, k_phon=a.k_phon, k_addr=a.k_addr, k_placeless=a.k_placeless, k_max=a.sweep)
    t_all = time.time()
    probe = None
    if a.split == "train" and a.sample == "dev" and not a.no_probe:
        probe = probe_ids()
    masks = sample_masks(a.split, a.sample, extra_s1=None if probe is None else probe.entity_id)
    s1, tg = load_side(a.split, masks)
    print(f"[{a.split}/{a.sample}] S1 {len(s1):,} (probe {0 if probe is None else len(probe):,}) | "
          f"targets {len(tg):,}  ({time.time() - t_all:.0f}s load)", flush=True)
    t0 = time.time()
    raw = generate(s1, tg, cfg)
    t_gen = time.time() - t0
    cands = raw[select(raw, cfg)].reset_index(drop=True) if cfg.k_max else raw   # no sweep: K already applied
    tag = f"{a.split}_{a.sample}"
    save_candidates(cands, masks, tag)   # before the report: generation is the expensive part
    if cfg.k_max:   # keep every generated pair with its ranks/bits for offline tuning (blocking_tune.py)
        tdir = paths.CACHE_DIR / "candidates"
        tdir.mkdir(parents=True, exist_ok=True)
        raw.assign(probe=(s1.entity_id.isin(set(probe.entity_id)).to_numpy()[raw.s] if probe is not None else False)
                   ).to_parquet(tdir / f"raw_{tag}.parquet", index=False)
    out = [f"# Blocking — {tag}\n", f"config: {cfg}\n",
           f"S1 {len(s1):,} | targets {len(tg):,} | generation {t_gen / 60:.1f} min\n"]

    s1c = s1.country.astype(object)
    is_probe = s1.entity_id.isin(set(probe.entity_id)).to_numpy() if probe is not None else np.zeros(len(s1), bool)
    main_rows = ~is_probe
    per = cands.groupby("s").size().reindex(range(len(s1)), fill_value=0)
    src = tg.src.to_numpy()[cands.t.to_numpy()]
    in_main = main_rows[cands.s.to_numpy()]
    stats = pd.DataFrame({"country": s1c[main_rows], "n": per[main_rows]}).groupby("country").n.describe(
        percentiles=[.5, .9, .99])
    stats.loc["ALL"] = per[main_rows].describe(percentiles=[.5, .9, .99])
    out += ["## Candidates per S1\n", stats.round(1).to_markdown(),
            f"\ntotal candidate pairs: {int(in_main.sum()):,} (S2 {int((src[in_main] == 2).sum()):,}, "
            f"S3 {int((src[in_main] == 3).sum()):,})\n"]

    if a.split == "train":
        truth = load_truth()
        tp = truth_pairs(s1, tg, truth)
        tp["country"] = s1c.to_numpy()[tp.s]
        if cfg.k_max:
            tp.assign(probe=is_probe[tp.s]).to_parquet(paths.CACHE_DIR / "candidates" / f"truth_{tag}.parquet",
                                                       index=False)
        tpb = score(cands, tp, len(tg))
        tpb["probe"] = is_probe[tpb.s]
        main_tp = tpb[~tpb.probe]
        out += ["## Pair recall\n",
                f"overall: **{(main_tp.bits > 0).mean():.4f}** of {len(main_tp):,} true pairs "
                f"(outside pool: {(main_tp.t < 0).mean():.4%})\n",
                _recall_table(main_tp, ["country"]).round(4).to_markdown(), "",
                _recall_table(main_tp, ["country", "src"]).round(4).to_markdown(), ""]
        cover = main_tp.assign(hit=main_tp.bits > 0).groupby(["country", "s"]).hit.all().groupby("country").mean()
        out += ["## S1 with the full match set covered (non-singletons)\n", cover.round(4).to_frame("full_cover").to_markdown(), ""]
        rows = []
        cm = cands[in_main]
        for b in BLOCKERS:
            bit = BIT[b]
            has = (main_tp.bits & bit) > 0
            only = main_tp.bits == bit
            rows.append({"blocker": b, "pairs": int(((cm.bits & bit) > 0).sum()),
                         "recall_alone": has.mean(), "unique_catches": int(only.sum()),
                         "unique_share_of_true": only.mean()})
        out += ["## Blockers (at the selected K)\n", pd.DataFrame(rows).set_index("blocker").round(4).to_markdown(), ""]
        if cfg.k_max:
            out += [f"## K sweep (all ranked blockers at K; placeless capped at {cfg.k_placeless})\n",
                    _sweep(raw, tp.assign(probe=is_probe[tp.s])[~is_probe[tp.s]], is_probe, int(main_rows.sum()), cfg, len(tg))
                    .round(4).to_markdown(), ""]
        f = load_folds().set_index("entity_id")
        ids = s1.entity_id.astype(object)
        tpb["city"] = f.city.reindex(ids).to_numpy()[tpb.s]
        tpb["tier"] = f.tier.reindex(ids).to_numpy()[tpb.s]
        ind = tpb[(tpb.country == "India") & (tpb.tier == "metro")]
        mt = ind.assign(hit=ind.bits > 0).groupby("city").agg(true_pairs=("hit", "size"), recall=("hit", "mean"),
                                                             in_probe=("probe", "mean"))
        cand_city = pd.Series(per.values, index=f.city.reindex(ids).values)
        mt["cands_per_s1"] = cand_city.groupby(level=0).mean().reindex(mt.index)
        out += ["## India metros (probe: random S1s per metro outside DEV; pool = same state groups)\n",
                mt.sort_values("recall").round(4).to_markdown(),
                f"\nall India-metro pairs: recall {(ind.bits > 0).mean():.4f}\n"]
        miss = missed_reasons(main_tp[main_tp.bits == 0], s1, tg, a.split, masks)
        rs = miss.groupby(["country", "reason"]).size().unstack(0).fillna(0).astype(int)
        rs["share_of_missed"] = rs.sum(axis=1) / len(miss)
        out += ["## Missed true pairs by likely reason\n", rs.sort_values("share_of_missed", ascending=False)
                .round(3).to_markdown(), ""]
        paths.REPORTS_DIR.mkdir(exist_ok=True)
        miss.sort_values(["reason", "country"]).to_csv(paths.REPORTS_DIR / f"blocking_missed_{tag}.tsv", sep="\t",
                                                       index=False)
    txt = "\n".join(out)
    print(txt, flush=True)
    (paths.REPORTS_DIR / f"blocking_{tag}.md").write_text(txt, encoding="utf-8")
    print(f"total {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
