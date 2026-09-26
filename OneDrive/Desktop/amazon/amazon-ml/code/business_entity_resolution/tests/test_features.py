import numpy as np
import pandas as pd

from ber.features import _group_ctx, _lead_num, _num_features


class _S:  # minimal stand-in for features.Side
    def __init__(self, prem):
        self.num, self.ndig = _lead_num(pd.Series(prem, dtype=object).fillna(""))


def test_soft_house_number_features():
    a = _S(["3424", "425", "203", "1795", "", "9062/17"])
    b = _S(["424", "430", "0203", "1795", "12", "9062/11"])
    idx = np.arange(6)
    f = _num_features(a, b, idx, idx)
    assert f["num_suffix"].tolist() == [True, False, False, False, False, False]    # 3424 -> 424 truncation
    assert f["num_eq"].tolist() == [False, False, True, True, False, True]          # zero-padding = equal
    assert f["num_diff_1_10"].tolist() == [False, True, False, False, False, False]  # decoy-style shift
    assert f["num_one_missing"].tolist() == [False, False, False, False, True, False]
    assert np.isnan(f["num_absdiff"][4]) and f["num_absdiff"][0] > 0


def test_prefix_containment():
    a, b = _S(["1201"]), _S(["120"])
    f = _num_features(a, b, np.array([0]), np.array([0]))
    assert f["num_prefix"].tolist() == [True]


def test_group_context():
    key = np.array([1, 1, 1, 2])
    val = np.array([0.9, 0.5, 0.9, 0.3], np.float32)
    rank, gap, z, n = _group_ctx(key, val)
    assert rank.tolist() == [1, 3, 1, 1]
    assert np.allclose(gap, [0, 0.4, 0, 0])
    assert n.tolist() == [3, 3, 3, 1]


def test_sibling_features_flag_decoy():
    from ber.features import sibling_features
    df = pd.DataFrame({"s": [1, 1, 2], "a_cos": [0.9, 0.95, 0.5], "n_tset_best": [95, 96, 90], "n_cos": [.9, .9, .8],
                       "prem_ov": [1, 0, 0], "stkey_ov": [1, 1, 0], "num_eq": [1, 0, 0], "num_both": [1, 1, 1],
                       "extra_t_cnt": [0, 1, 0]})
    f = sibling_features(df)
    assert f["sib_exactnum_decoy"].tolist() == [0, 1, 0]      # row 2 is the decoy twin of row 1
    assert f["d_extra_t_cnt"].tolist() == [0, 1, 0]
    assert np.allclose(f["d_a_cos"], [0.05, 0, 0])
