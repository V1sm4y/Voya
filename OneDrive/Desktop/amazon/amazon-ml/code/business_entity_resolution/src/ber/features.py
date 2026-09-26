"""Pair features for (S1, target) candidates: vectorised, streamed by partition, float32.

Design for 16 GB RAM / low disk:
  * Record-level data is loaded once per split (`FeatureContext`):
    - prep strings stay as pyarrow arrays and are gathered per chunk;
    - multi-valued fields (premise, locality, street, building, places) become hashed sparse binary matrices,
      so "shared value" features are row-wise sparse dot products;
    - numeric per-record arrays (leading premise number, unit hash, flags) are plain numpy.
  * Pairs are processed per S1 partition (country, state group), in S1-aligned chunks. `iter_features`
    yields one chunk DataFrame at a time, so the full feature table is never materialised.
  * Context features come from a cheap base score computed for every pair of the partition: rank / gap /
    z-score within the S1's candidates, reverse rank among the S1s that share the target, and candidate
    counts. Relative context features stay meaningful when test has more candidates per S1.

Nothing here rejects a pair: legal-form conflicts, number mismatches etc. are all features. Raw country is
never a feature (the model must transfer to France).
"""
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from .blocking import BIT, BLOCKERS, RANKED
from .names import legal_relation
from .prep import prep_dir
from .splits import state_group

REC_COLS = ["country", "state", "city", "places", "core_name", "name_tx", "core_key", "legal_form", "alias_tx",
            "addr_tx", "street", "street_key", "locality", "locality_key", "building_key", "premise_no", "unit",
            "landmark"]
MULTI = ["premise_no", "locality_key", "street_key", "building_key", "places", "landmark"]
LEGAL_CODES = {"missing": 0, "same": 1, "subset": 2, "overlap": 3, "conflict": 4}

NAME_F = ["n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "k_ratio", "alias_tset", "n_tset_best", "n_cos",
          "w_shared_s", "w_shared_t", "extra_t_cnt", "extra_t_idf", "extra_t_idf_max", "extra_s_cnt",
          "extra_s_idf_max", "len_s", "len_t", "tok_s", "tok_t", "legal_rel", "s_has_legal", "t_has_legal",
          "t_domain", "t_handle", "t_initials", "t_indic", "t_junk"]
ADDR_F = ["st_ratio", "st_tset", "a_tset", "a_cos", "loc_tset", "lockey_ov", "bldg_ov", "stkey_ov", "prem_ov",
          "num_both", "num_eq", "num_absdiff", "num_diff_1_10", "num_prefix", "num_suffix", "num_lev",
          "num_one_missing", "unit_both", "unit_eq", "unit_conflict", "places_ov", "s_city_in_t", "sgroup_eq",
          "state_eq", "t_state_missing", "t_addr_empty", "t_placeless", "landmark_ov"]
CTX_F = ["base", "rank_base_s", "gap_base_s", "z_base_s", "n_cand_s", "n_cand_s_src", "rank_name_s",
         "gap_name_s", "rank_addr_s", "gap_addr_s", "n_cand_t", "rank_base_t", "gap_base_t", "is_best_s_for_t",
         "src_s3"] + [f"b_{b}" for b in BLOCKERS] + [f"r_{b}" for b in RANKED]
# iteration 1 (error analysis): name commonness + sibling-relative features
FREQ_F = ["core_eq", "s_core_freq", "t_core_freq"]
SIB_BASE = ["a_cos", "n_tset_best", "n_cos", "prem_ov", "stkey_ov", "num_eq"]
SIB_F = [f"g_max_{k}" for k in SIB_BASE] + [f"d_{k}" for k in SIB_BASE] + ["sib_exactnum_decoy", "d_extra_t_cnt"]
FEATURES = NAME_F + ADDR_F + CTX_F + FREQ_F + SIB_F


def _pipe_tokens(s):
    return s.split("|") if s else []


_HASH = HashingVectorizer(analyzer=_pipe_tokens, n_features=2 ** 22, binary=True, norm=None,
                          alternate_sign=False, dtype=np.float32)


def _strs(arr: pa.Array, idx: np.ndarray) -> list:
    """Gather strings (None -> '') for positions idx."""
    out = pc.take(arr, pa.array(idx, type=pa.int64())).to_pylist()
    return [x or "" for x in out]


def _rowdot(A, B) -> np.ndarray:
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)


def _lead_num(s: pd.Series):
    """Leading integer of the first premise id and its digit count (-1 / 0 when absent)."""
    first = s.str.split("|").str[0]
    num = first.str.extract(r"(\d{1,9})", expand=False)
    n = pd.to_numeric(num, errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    d = num.str.len().fillna(0).astype(np.int8).to_numpy()
    return n, d


@dataclass
class Side:
    tab: dict           # column -> pa.Array (strings)
    num: np.ndarray     # leading premise number
    ndig: np.ndarray
    unit: np.ndarray    # uint64 hash of first unit (0 = none)
    legal: np.ndarray   # int code of the legal_form string
    sgroup: np.ndarray  # int code
    state: np.ndarray   # int code (-1 = unknown)
    multi: dict         # field -> CSR binary
    flags: dict         # name -> int8/float arrays


class FeatureContext:
    """Everything record-level for one split. Memory ~3-4 GB for train."""

    def __init__(self, split: str, verbose: bool = True):
        t0 = time.time()
        self.split = split
        d = prep_dir()
        s1 = pq.read_table(d / f"{split}_source1.parquet", columns=REC_COLS)
        tg = pa.concat_tables([pq.read_table(d / f"{split}_source{n}.parquet", columns=REC_COLS) for n in (2, 3)])
        self.n2 = pq.read_metadata(d / f"{split}_source2.parquet").num_rows
        from .data import _read_cached
        raw_t = pd.concat([_read_cached(f"{split}_source{n}", columns=["business_name", "business_address"])
                           for n in (2, 3)], ignore_index=True)
        self.legal_vocab = {}
        self.s1 = self._side(s1, None)
        self.tg = self._side(tg, raw_t)
        del raw_t
        # how common each core name is: number of S1s / targets in this split carrying it
        c1 = pd.Series(s1.column("core_name").to_pandas(), dtype=object)
        ct = pd.Series(tg.column("core_name").to_pandas(), dtype=object)
        v1, vt = c1.value_counts(), ct.value_counts()
        self.core_id_s1 = pd.util.hash_array(c1.fillna("").to_numpy(dtype=object))
        self.core_id_t = pd.util.hash_array(ct.fillna("").to_numpy(dtype=object))
        self.s1.flags["core_freq_s1"] = np.log1p(c1.map(v1).fillna(0).to_numpy(np.float32))
        self.s1.flags["core_freq_t"] = np.log1p(c1.map(vt).fillna(0).to_numpy(np.float32))
        self.tg.flags["core_freq_t"] = np.log1p(ct.map(vt).fillna(0).to_numpy(np.float32))
        del c1, ct
        self.countries = sorted(set(self.s1.tab["country"].to_pylist()))
        cidx = {c: i for i, c in enumerate(self.countries)}
        self.s1_country = np.fromiter((cidx[c] for c in self.s1.tab["country"].to_pylist()), np.int64,
                                      count=len(self.s1.num))
        self._vec = {}
        if verbose:
            print(f"  feature context {split}: S1 {s1.num_rows:,} targets {tg.num_rows:,} ({time.time() - t0:.0f}s)",
                  flush=True)

    def _codes(self, arr: pa.Array, vocab: dict) -> np.ndarray:
        vals = arr.to_pylist()
        return np.fromiter((vocab.setdefault(v or "", len(vocab)) for v in vals), dtype=np.int32, count=len(vals))

    def _side(self, t: pa.Table, raw: pd.DataFrame) -> Side:
        tab = {c: t.column(c).combine_chunks() for c in REC_COLS}
        prem = pd.Series(tab["premise_no"].to_pandas(), dtype=object).fillna("")
        num, ndig = _lead_num(prem)
        unit_first = pd.Series(tab["unit"].to_pandas(), dtype=object).fillna("").str.split("|").str[0]
        unit = np.where(unit_first.eq("").to_numpy(), np.uint64(0),
                        pd.util.hash_array(unit_first.to_numpy(dtype=object)))
        legal = self._codes(tab["legal_form"], self.legal_vocab)
        st = pd.Series(tab["state"].to_pandas(), dtype=object)
        sg = state_group(st.where(st.notna(), None))
        if not hasattr(self, "_sg_vocab"):
            self._sg_vocab, self._st_vocab = {}, {}
        sgroup = np.fromiter((self._sg_vocab.setdefault(v, len(self._sg_vocab)) if isinstance(v, str) else -1
                              for v in sg), dtype=np.int32, count=len(sg))
        state = np.fromiter((self._st_vocab.setdefault(v, len(self._st_vocab)) if isinstance(v, str) else -1
                             for v in st), dtype=np.int32, count=len(st))
        multi = {f: _HASH.transform(pd.Series(tab[f].to_pandas(), dtype=object).fillna("").tolist()).tocsr()
                 for f in MULTI}
        city = pd.Series(tab["city"].to_pandas(), dtype=object).fillna("").tolist()
        multi["city"] = _HASH.transform(city).tocsr()
        flags = {}
        name_tx = pd.Series(tab["name_tx"].to_pandas(), dtype=object).fillna("")
        flags["len"] = name_tx.str.len().to_numpy(np.float32)
        flags["tok"] = name_tx.str.count(" ").add(1).where(name_tx.ne(""), 0).to_numpy(np.float32)
        flags["has_legal"] = pd.Series(tab["legal_form"].to_pandas(), dtype=object).fillna("").ne("").to_numpy(np.int8)
        addr_empty = pd.Series(tab["addr_tx"].to_pandas(), dtype=object).isna() & \
            pd.Series(tab["places"].to_pandas(), dtype=object).isna()
        flags["placeless"] = pd.Series(tab["places"].to_pandas(), dtype=object).isna().to_numpy(np.int8)
        if raw is not None:
            nm = raw.business_name
            flags["domain"] = nm.str.contains(r"\.(?:com|in|net|org|co|biz|fr)\b|www\.", case=False).to_numpy(np.int8)
            flags["handle"] = nm.str.contains(r"^\s*@|\s@\w", regex=True).to_numpy(np.int8)
            core = pd.Series(tab["core_name"].to_pandas(), dtype=object).fillna("")
            flags["initials"] = (core.str.fullmatch(r"[a-z]{1,4}") | core.str.fullmatch(r"(?:[a-z] ){1,4}[a-z]")) \
                .fillna(False).to_numpy(np.int8)
            flags["indic"] = nm.str.contains(r"[ऀ-෿]").to_numpy(np.int8)
            flags["junk"] = nm.str.contains(r"^\s*(?:--|\*+|<<|>>|\[|\.\.|#|~)").to_numpy(np.int8)
            flags["addr_empty"] = raw.business_address.str.strip().eq("").to_numpy(np.int8)
        else:
            flags["addr_empty"] = addr_empty.to_numpy(np.int8)
        return Side(tab, num, ndig, unit, legal, sgroup, state, multi, flags)

    # ---- per (country) text vectorisers, fitted on a target sample of this split
    def vectorisers(self, country: str):
        if country in self._vec:
            return self._vec[country]
        ctry = self.tg.tab["country"]
        idx = np.flatnonzero(pc.equal(ctry, country).to_numpy(zero_copy_only=False))
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, size=min(len(idx), 600_000), replace=False))
        names = _strs(self.tg.tab["name_tx"], idx)
        addrs = _strs(self.tg.tab["addr_tx"], idx)
        v = {
            "name_c": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True,
                                      dtype=np.float32).fit(names),
            "addr_c": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True,
                                      dtype=np.float32).fit(addrs),
            "name_w": TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]{2,}", binary=True, norm=None,
                                      min_df=1, dtype=np.float32).fit(names),
        }
        self._vec[country] = v
        return v

    def legal_rel_table(self) -> np.ndarray:
        inv = {c: s for s, c in self.legal_vocab.items()}
        n = len(inv)
        tab = np.zeros((n, n), np.int8)
        for i in range(n):
            for j in range(n):
                tab[i, j] = LEGAL_CODES[legal_relation(inv[i], inv[j])]
        return tab


# ------------------------------------------------------------------ pair features (one chunk)
def _num_features(a: Side, b: Side, s, t) -> dict:
    n1, n2 = a.num[s], b.num[t]
    d1, d2 = a.ndig[s].astype(np.int64), b.ndig[t].astype(np.int64)
    both = (n1 >= 0) & (n2 >= 0)
    one = (n1 >= 0) ^ (n2 >= 0)
    diff = np.where(both, np.abs(n1 - n2), -1)
    p1 = np.power(10, np.clip(d1, 0, 18)).astype(np.int64)
    p2 = np.power(10, np.clip(d2, 0, 18)).astype(np.int64)
    suf = both & (((d1 >= d2) & (n1 % np.maximum(p2, 1) == n2)) | ((d2 > d1) & (n2 % np.maximum(p1, 1) == n1)))
    pre = both & (((d1 > d2) & (n1 // np.maximum(np.power(10, np.clip(d1 - d2, 0, 18)), 1) == n2)) |
                  ((d2 > d1) & (n2 // np.maximum(np.power(10, np.clip(d2 - d1, 0, 18)), 1) == n1)))
    return {"num_both": both, "num_eq": both & (n1 == n2),
            "num_absdiff": np.where(both, np.log1p(np.maximum(diff, 0)), np.nan),
            "num_diff_1_10": both & (diff >= 1) & (diff <= 10), "num_prefix": pre & (n1 != n2),
            "num_suffix": suf & (n1 != n2), "num_one_missing": one}


TIMER = None   # set to a dict to collect per-step seconds (profiling)


def _tick(name, t0):
    if TIMER is not None:
        TIMER[name] = TIMER.get(name, 0.0) + time.time() - t0
    return time.time()


def pair_features(ctx: FeatureContext, s: np.ndarray, t: np.ndarray, vec: dict, S1v: dict, Tv: dict,
                  s_pos: np.ndarray, t_pos: np.ndarray, legal_tab: np.ndarray) -> dict:
    """Name + address features for pairs (s, t) (global indices). S1v/Tv hold the partition's vectorised rows
    (indexed by s_pos/t_pos)."""
    a, b = ctx.s1, ctx.tg
    f = {}
    t0 = time.time()
    ns, nt = _strs(a.tab["name_tx"], s), _strs(b.tab["name_tx"], t)
    t0 = _tick("gather_names", t0)
    f["n_ratio"] = cpdist(ns, nt, scorer=fuzz.ratio, workers=-1)
    f["n_tset"] = cpdist(ns, nt, scorer=fuzz.token_set_ratio, workers=-1)
    f["n_tsort"] = cpdist(ns, nt, scorer=fuzz.token_sort_ratio, workers=-1)
    f["n_partial"] = cpdist(ns, nt, scorer=fuzz.partial_ratio, workers=-1)
    f["n_jw"] = cpdist(ns, nt, scorer=JaroWinkler.normalized_similarity, workers=-1)
    del ns, nt
    t0 = _tick("name_fuzzy", t0)
    f["k_ratio"] = cpdist(_strs(a.tab["core_key"], s), _strs(b.tab["core_key"], t), scorer=fuzz.ratio, workers=-1)
    al = _strs(b.tab["alias_tx"], t)
    has_al = np.fromiter((bool(x) for x in al), bool, count=len(al))
    at = np.full(len(t), np.nan, np.float32)
    if has_al.any():
        ii = np.flatnonzero(has_al)
        at[ii] = cpdist(_strs(a.tab["name_tx"], s[ii]), [al[i] for i in ii], scorer=fuzz.token_set_ratio, workers=-1)
    f["alias_tset"] = at
    f["n_tset_best"] = np.fmax(f["n_tset"], at)
    t0 = _tick("alias_key", t0)
    f["n_cos"] = _rowdot(S1v["name_c"][s_pos], Tv["name_c"][t_pos])
    t0 = _tick("name_cos", t0)
    A, B = S1v["name_w"][s_pos], Tv["name_w"][t_pos]
    Ab, Bb = A.copy(), B.copy()
    Ab.data[:] = 1
    Bb.data[:] = 1
    shared = _rowdot(A, Bb)
    s_tot = np.asarray(A.sum(1)).ravel()
    t_tot = np.asarray(B.sum(1)).ravel()
    f["w_shared_s"] = np.where(s_tot > 0, shared / np.maximum(s_tot, 1e-6), np.nan)
    f["w_shared_t"] = np.where(t_tot > 0, shared / np.maximum(t_tot, 1e-6), np.nan)
    nshared = _rowdot(Ab, Bb)
    extra_t = B - B.multiply(Ab)
    extra_s = A - A.multiply(Bb)
    f["extra_t_cnt"] = np.diff(B.indptr) - nshared
    f["extra_t_idf"] = t_tot - shared
    f["extra_t_idf_max"] = extra_t.max(axis=1).toarray().ravel()
    f["extra_s_cnt"] = np.diff(A.indptr) - nshared
    f["extra_s_idf_max"] = extra_s.max(axis=1).toarray().ravel()
    del A, B, Ab, Bb, extra_t, extra_s
    t0 = _tick("idf_tokens", t0)
    f["len_s"], f["len_t"] = a.flags["len"][s], b.flags["len"][t]
    f["tok_s"], f["tok_t"] = a.flags["tok"][s], b.flags["tok"][t]
    f["legal_rel"] = legal_tab[a.legal[s], b.legal[t]]
    f["s_has_legal"], f["t_has_legal"] = a.flags["has_legal"][s], b.flags["has_legal"][t]
    for k in ("domain", "handle", "initials", "indic", "junk"):
        f[f"t_{k}"] = b.flags[k][t]
    t0 = _tick("flags", t0)
    # address
    sts, stt = _strs(a.tab["street"], s), _strs(b.tab["street"], t)
    f["st_ratio"] = cpdist(sts, stt, scorer=fuzz.ratio, workers=-1)
    f["st_tset"] = cpdist(sts, stt, scorer=fuzz.token_set_ratio, workers=-1)
    del sts, stt
    f["a_tset"] = cpdist(_strs(a.tab["addr_tx"], s), _strs(b.tab["addr_tx"], t), scorer=fuzz.token_set_ratio,
                         workers=-1)
    t0 = _tick("addr_fuzzy", t0)
    f["a_cos"] = _rowdot(S1v["addr_c"][s_pos], Tv["addr_c"][t_pos])
    t0 = _tick("addr_cos", t0)
    f["loc_tset"] = cpdist(_strs(a.tab["locality"], s), _strs(b.tab["locality"], t), scorer=fuzz.token_set_ratio,
                           workers=-1)
    for name, fld in (("lockey_ov", "locality_key"), ("bldg_ov", "building_key"), ("stkey_ov", "street_key"),
                      ("prem_ov", "premise_no"), ("places_ov", "places"), ("landmark_ov", "landmark")):
        f[name] = _rowdot(a.multi[fld][s], b.multi[fld][t])
    t0 = _tick("loc_fuzzy", t0)
    f["s_city_in_t"] = _rowdot(a.multi["city"][s], b.multi["places"][t])
    t0 = _tick("multi_overlaps", t0)
    f.update(_num_features(a, b, s, t))
    p1, p2 = _strs(a.tab["premise_no"], s), _strs(b.tab["premise_no"], t)
    f["num_lev"] = cpdist([x.split("|")[0] for x in p1], [x.split("|")[0] for x in p2], scorer=Levenshtein.distance,
                          workers=-1).astype(np.float32)
    f["num_lev"][(a.num[s] < 0) | (b.num[t] < 0)] = np.nan
    u1, u2 = a.unit[s], b.unit[t]
    f["unit_both"] = (u1 > 0) & (u2 > 0)
    f["unit_eq"] = f["unit_both"] & (u1 == u2)
    f["unit_conflict"] = f["unit_both"] & (u1 != u2)
    f["sgroup_eq"] = (a.sgroup[s] == b.sgroup[t]) & (b.sgroup[t] >= 0)
    f["state_eq"] = (a.state[s] == b.state[t]) & (b.state[t] >= 0)
    f["t_state_missing"] = b.state[t] < 0
    f["t_addr_empty"] = b.flags["addr_empty"][t]
    f["t_placeless"] = b.flags["placeless"][t]
    f["core_eq"] = (ctx.core_id_s1[s] == ctx.core_id_t[t]) & (a.flags["len"][s] > 0)
    f["s_core_freq"] = a.flags["core_freq_s1"][s]      # how many S1s share the S1's core name
    f["t_core_freq"] = b.flags["core_freq_t"][t]       # how many targets share the target's core name
    _tick("numbers_units_geo", t0)
    return f


def sibling_features(df: pd.DataFrame) -> dict:
    """Relative to the S1's other candidates (df holds whole S1 groups). The decoy's real twin is usually a
    sibling candidate with the exact number and no extra token."""
    g = df.groupby("s", sort=False)
    out = {}
    for k in SIB_BASE:
        mx = g[k].transform("max").to_numpy(np.float32)
        out[f"g_max_{k}"] = mx
        out[f"d_{k}"] = mx - df[k].to_numpy(np.float32)
    twin = ((df.num_eq > 0) & (df.n_tset_best >= 80)).astype(np.float32)
    has_twin = twin.groupby(df.s).transform("max").to_numpy(np.float32)
    out["sib_exactnum_decoy"] = has_twin * (1 - df.num_eq.to_numpy(np.float32)) * (df.num_both.to_numpy(np.float32))
    out["d_extra_t_cnt"] = (df.extra_t_cnt - g.extra_t_cnt.transform("min")).to_numpy(np.float32)
    return out


def _group_ctx(key: np.ndarray, val: np.ndarray):
    """Per-group dense descending rank (1 = best), gap to group max, z-score and group size."""
    df = pd.DataFrame({"k": key, "v": val})
    g = df.groupby("k").v
    mx = g.transform("max").to_numpy()
    mu = g.transform("mean").to_numpy()
    sd = g.transform("std").fillna(0).to_numpy()
    rank = g.rank(method="min", ascending=False).to_numpy()
    n = g.transform("size").to_numpy()
    return rank.astype(np.float32), (mx - val).astype(np.float32), ((val - mu) / (sd + 1e-6)).astype(np.float32), \
        n.astype(np.float32)


def context_features(s, t, src3, base, name_sc, addr_sc, bits, ranks) -> dict:
    f = {"base": base}
    f["rank_base_s"], f["gap_base_s"], f["z_base_s"], f["n_cand_s"] = _group_ctx(s, base)
    f["n_cand_s_src"] = pd.Series(np.ones(len(s))).groupby([s, src3]).transform("size").to_numpy(np.float32)
    f["rank_name_s"], f["gap_name_s"], _, _ = _group_ctx(s, name_sc)
    f["rank_addr_s"], f["gap_addr_s"], _, _ = _group_ctx(s, addr_sc)
    f["rank_base_t"], f["gap_base_t"], _, f["n_cand_t"] = _group_ctx(t, base)
    f["is_best_s_for_t"] = (f["rank_base_t"] == 1).astype(np.float32)
    f["src_s3"] = src3.astype(np.float32)
    for bname in BLOCKERS:
        f[f"b_{bname}"] = ((bits & BIT[bname]) > 0).astype(np.float32)
    for bname in RANKED:
        r = ranks[bname].astype(np.float32)
        f[f"r_{bname}"] = np.where(r >= 255, np.nan, r)
    return f


# ------------------------------------------------------------------ partition streaming
def partition_keys(ctx: FeatureContext, s: np.ndarray) -> np.ndarray:
    """Partition id per pair = (country, state group) of the S1, as one int (country_index * 10000 + sgroup)."""
    return ctx.s1_country[s] * 10_000 + np.maximum(ctx.s1.sgroup[s], 0)


def base_scores(ctx: FeatureContext, s, t, chunk=2_000_000):
    """Cheap name / address scores for every pair (drive the context features)."""
    name_sc = np.empty(len(s), np.float32)
    addr_sc = np.empty(len(s), np.float32)
    for i in range(0, len(s), chunk):
        ss, tt = s[i:i + chunk], t[i:i + chunk]
        ns = _strs(ctx.s1.tab["name_tx"], ss)
        n1 = cpdist(ns, _strs(ctx.tg.tab["name_tx"], tt), scorer=fuzz.token_set_ratio, workers=-1)
        al = _strs(ctx.tg.tab["alias_tx"], tt)
        ii = np.flatnonzero([bool(x) for x in al])
        if len(ii):
            n1[ii] = np.fmax(n1[ii], cpdist([ns[k] for k in ii], [al[k] for k in ii], scorer=fuzz.token_set_ratio,
                                            workers=-1))
        name_sc[i:i + chunk] = n1 / 100
        addr_sc[i:i + chunk] = cpdist(_strs(ctx.s1.tab["addr_tx"], ss), _strs(ctx.tg.tab["addr_tx"], tt),
                                      scorer=fuzz.token_set_ratio, workers=-1) / 100
    return name_sc, addr_sc, (0.5 * name_sc + 0.5 * addr_sc).astype(np.float32)


def iter_features(ctx: FeatureContext, cands: pd.DataFrame, want_s: np.ndarray = None, chunk: int = 1_500_000,
                  verbose: bool = True):
    """Yield (partition key, DataFrame[s, t, FEATURES...]) chunks. `cands` has global s/t, bits, rank_* columns
    and must be sorted by s. `want_s` (bool array over S1 rows) restricts which S1s get full features;
    context still uses every pair of the partition."""
    s_all = cands.s.to_numpy(np.int64)
    t_all = cands.t.to_numpy(np.int64)
    part = partition_keys(ctx, s_all)
    order = np.argsort(part, kind="stable")          # keeps s-order inside a partition
    bounds = np.flatnonzero(np.diff(part[order])) + 1
    legal_tab = ctx.legal_rel_table()
    for grp in np.split(order, bounds):
        t0 = time.time()
        s, t = s_all[grp], t_all[grp]
        country = ctx.countries[int(part[grp[0]] // 10_000)]
        src3 = (t >= ctx.n2).astype(np.int8)
        bits = cands.bits.to_numpy()[grp]
        ranks = {b: cands[f"rank_{b}"].to_numpy()[grp] for b in RANKED}
        name_sc, addr_sc, base = base_scores(ctx, s, t)
        cx = context_features(s, t, src3, base, name_sc, addr_sc, bits, ranks)
        keep = np.ones(len(s), bool) if want_s is None else want_s[s]
        if not keep.any():
            continue
        vec = ctx.vectorisers(country)
        us, s_inv = np.unique(s[keep], return_inverse=True)
        ut, t_inv = np.unique(t[keep], return_inverse=True)
        S1v = {"name_c": vec["name_c"].transform(_strs(ctx.s1.tab["name_tx"], us)),
               "addr_c": vec["addr_c"].transform(_strs(ctx.s1.tab["addr_tx"], us)),
               "name_w": vec["name_w"].transform(_strs(ctx.s1.tab["name_tx"], us))}
        Tv = {"name_c": vec["name_c"].transform(_strs(ctx.tg.tab["name_tx"], ut)),
              "addr_c": vec["addr_c"].transform(_strs(ctx.tg.tab["addr_tx"], ut)),
              "name_w": vec["name_w"].transform(_strs(ctx.tg.tab["name_tx"], ut))}
        kidx = np.flatnonzero(keep)
        ks = s[kidx]
        cut = np.flatnonzero(np.diff(ks)) + 1                         # S1 boundaries
        starts = [0]
        for c in cut:
            if c - starts[-1] >= chunk:
                starts.append(c)
        starts.append(len(kidx))
        n_done = 0
        for a0, a1 in zip(starts[:-1], starts[1:]):
            sel = kidx[a0:a1]
            pf = pair_features(ctx, s[sel], t[sel], vec, S1v, Tv, s_inv[a0:a1], t_inv[a0:a1], legal_tab)
            df = pd.DataFrame({"s": s[sel].astype(np.int32), "t": t[sel].astype(np.int32)})
            for k in NAME_F + ADDR_F + FREQ_F:
                df[k] = np.asarray(pf[k], dtype=np.float32)
            for k in CTX_F:
                df[k] = np.asarray(cx[k][sel], dtype=np.float32)
            for k, v in sibling_features(df).items():
                df[k] = v
            n_done += len(sel)
            yield int(part[grp[0]]), df
        if verbose:
            print(f"    partition {country}/{int(part[grp[0]] % 10_000)}: pairs {len(s):,} featurised {n_done:,} "
                  f"({time.time() - t0:.0f}s)", flush=True)
