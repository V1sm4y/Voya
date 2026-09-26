import math

import pytest

from ber.metrics import f05, per_entity, report


def test_singleton_empty_prediction_scores_one():
    assert f05([], []) == 1.0


def test_singleton_any_prediction_scores_zero():
    assert f05([], ["S2-1"]) == 0.0


def test_nonsingleton_empty_prediction_scores_zero():
    assert f05(["S2-1"], []) == 0.0


def test_no_overlap_scores_zero():
    assert f05(["S2-1", "S3-2"], ["S2-9"]) == 0.0


def test_perfect_scores_one():
    assert f05(["S2-1", "S3-2"], ["S3-2", "S2-1"]) == 1.0


def test_readme_example():
    # README: predict [47,193,812], truth [47,812] -> P=2/3, R=1 -> 0.714
    got = f05(["S2-00047", "S3-00812"], ["S2-00047", "S2-00193", "S3-00812"])
    assert math.isclose(got, 1.25 * (2 / 3) / (0.25 * (2 / 3) + 1))
    assert round(got, 3) == 0.714


@pytest.mark.parametrize("k,j,expected", [(4, 1, 0.625), (4, 2, 2.5 / 3), (4, 3, 0.9375)])
def test_partial_recall_full_precision(k, j, expected):
    truth = [f"S2-{i}" for i in range(k)]
    assert math.isclose(f05(truth, truth[:j]), expected)


def test_one_false_positive_on_complete_set():
    truth = [f"S2-{i}" for i in range(4)]
    assert math.isclose(f05(truth, truth + ["S3-99"]), 1.25 * 4 / (1.25 * 4 + 1))


def test_duplicates_in_prediction_are_set_semantics():
    assert f05(["S2-1"], ["S2-1", "S2-1"]) == 1.0


def test_missing_prediction_counts_as_empty():
    truth = {"S1-a": [], "S1-b": ["S2-1"]}
    e = per_entity(truth, {})
    assert list(e.f05) == [1.0, 0.0]


def test_macro_not_micro():
    # S1-a: 1 of 1 correct (F=1). S1-b: truth 9 links, predicts 1 correct -> F=1.25/(0.25*9+1)
    truth = {"S1-a": ["S2-1"], "S1-b": [f"S3-{i}" for i in range(9)]}
    pred = {"S1-a": ["S2-1"], "S1-b": ["S3-0"]}
    rep = report(truth, pred)
    expected = (1.0 + 1.25 / (0.25 * 9 + 1)) / 2
    assert math.isclose(rep["overall"].loc["all", "macro_f05"], expected)


def test_breakdowns_and_extra_predictions():
    truth = {"S1-a": [], "S1-b": ["S2-1", "S3-1"], "S1-c": ["S2-2"]}
    pred = {"S1-a": ["S2-5"], "S1-b": ["S2-1", "S3-1"], "S1-c": [], "S1-zzz": ["S2-7"]}
    country = {"S1-a": "US", "S1-b": "US", "S1-c": "India"}
    rep = report(truth, pred, country=country)
    assert rep["overall"].loc["all", "pred_s1_not_evaluated"] == 1
    assert rep["by_kind"].loc["singleton", "macro_f05"] == 0.0
    assert rep["by_kind"].loc["non-singleton", "macro_f05"] == 0.5
    assert rep["by_country"].loc["US", "macro_f05"] == 0.5
    assert rep["by_country"].loc["India", "macro_f05"] == 0.0
    assert list(rep["by_true_size"].index) == ["0", "1", "2"]


def test_eval_ids_subset_and_unknown():
    truth = {"S1-a": [], "S1-b": ["S2-1"]}
    assert per_entity(truth, {}, ids=["S1-a"]).f05.tolist() == [1.0]
    with pytest.raises(KeyError):
        per_entity(truth, {}, ids=["S1-nope"])


def test_size_bucket_clipping():
    truth = {f"S1-{n}": [f"S2-{n}-{i}" for i in range(n)] for n in range(11)}
    rep = report(truth, truth, max_size=8)
    assert rep["by_true_size"].index[-1] == "8+"
    assert rep["by_true_size"].loc["8+", "n_s1"] == 3
    assert rep["overall"].loc["all", "macro_f05"] == 1.0


def test_weighted_macro():
    truth = {"S1-a": ["S2-1"], "S1-b": ["S2-2"], "S1-c": []}
    pred = {"S1-a": ["S2-1"]}                       # f05: a=1, b=0, c=1
    rep = report(truth, pred, weights={"S1-a": 3.0, "S1-b": 1.0})   # c missing -> weight 0
    assert math.isclose(rep["overall"].loc["all", "macro_f05"], 2 / 3)
    assert math.isclose(rep["overall"].loc["all", "macro_f05_w"], 3 / 4)
