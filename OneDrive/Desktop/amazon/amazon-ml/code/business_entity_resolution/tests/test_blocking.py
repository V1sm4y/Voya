import numpy as np
import pandas as pd

from ber.blocking import BIT, _combos, _key_join, score


def brute(s_i, s_k, t_i, t_k, cap):
    post = pd.Series(t_k).value_counts()
    return sorted((a, b) for a, ka in zip(s_i, s_k) for b, kb in zip(t_i, t_k) if ka == kb and post[kb] <= cap)


def test_key_join_matches_brute_force():
    rng = np.random.default_rng(0)
    s_i, t_i = np.arange(200), np.arange(1000)
    s_k = rng.integers(0, 50, 200).astype(np.uint64)
    t_k = rng.integers(0, 80, 1000).astype(np.uint64)
    for cap in (5, 15, 1000):
        a, b = _key_join(s_i, s_k, t_i, t_k, cap)
        assert sorted(zip(a.tolist(), b.tolist())) == brute(s_i, s_k, t_i, t_k, cap)


def test_key_join_empty():
    a, b = _key_join(np.arange(3), np.array([1, 2, 3], np.uint64), np.arange(2), np.array([9, 9], np.uint64), 10)
    assert len(a) == len(b) == 0


def test_combos_cartesian_and_nulls():
    idx = np.array([10, 11, 12])
    places = pd.Series(["pune|haweli", None, "mumbai"])
    toks = pd.Series(["balaji", "x", "a|b"])
    i, h = _combos(idx, [places, toks])
    assert sorted(i.tolist()) == [10, 10, 12, 12]
    i2, h2 = _combos(np.array([5]), [pd.Series(["haweli"]), pd.Series(["balaji"])])
    assert h2[0] in set(h.tolist())          # same (place, token) -> same hash across calls


def test_score_marks_hits_and_misses():
    cands = pd.DataFrame({"s": [0, 0, 1], "t": [3, 5, 2], "bits": np.array([BIT["name_tfidf"], BIT["addr_key"], 3], np.uint16)})
    tp = pd.DataFrame({"s": [0, 1, 1], "t": [5, 2, -1]})
    out = score(cands, tp, n_t=10)
    assert out.bits.tolist() == [BIT["addr_key"], 3, 0]
