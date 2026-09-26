import numpy as np

from ber.decide import decide, fast_macro_f05
from ber.metrics import f05


def test_fast_macro_matches_reference():
    rng = np.random.default_rng(0)
    n_s1 = 300
    n_true = rng.integers(0, 5, n_s1)
    s = rng.integers(0, n_s1, 5000)
    t = np.arange(5000)
    y = (rng.random(5000) < 0.3).astype(np.int8)
    pred = rng.random(5000) < 0.2
    # reference: build sets; true pairs = scored y==1 plus unscored ones implied by n_true
    ref = []
    for i in range(n_s1):
        scored_true = set(t[(s == i) & (y == 1)])
        n_extra = max(0, n_true[i] - len(scored_true))
        n_true[i] = len(scored_true) + n_extra
        truth = [f"T{x}" for x in scored_true] + [f"X{i}_{k}" for k in range(n_extra)]
        ref.append(f05(truth, [f"T{x}" for x in t[(s == i) & pred]]))
    assert abs(fast_macro_f05(n_s1, n_true, s, y, pred) - np.mean(ref)) < 1e-12


def test_decide_assignment_ratio_null_cap():
    s = np.array([0, 0, 1, 1, 2, 2, 2])
    t = np.array([10, 11, 10, 12, 13, 14, 15])
    p = np.array([0.9, 0.5, 0.8, 0.3, 0.2, 0.25, 0.1], np.float32)
    k = decide(s, t, p, t_keep=0.25, ratio=0.5, t_null=0.4, assign=True)
    # t=10 goes to S1 0 (0.9 > 0.8); S1 1 keeps only t=12 (0.3) whose best is 0.3 < t_null -> empty
    assert k.tolist() == [True, True, False, False, False, False, False]
    k2 = decide(s, t, p, t_keep=0.0, ratio=0.0, t_null=0.0, assign=False, cap=1)
    assert k2.tolist() == [True, False, True, False, False, True, False]


def test_fast_decider_equals_decide():
    from ber.decide import FastDecider
    rng = np.random.default_rng(3)
    n = 4000
    s = rng.integers(0, 300, n)
    t = rng.integers(0, 1500, n)
    p = rng.random(n).astype(np.float32)
    fd = FastDecider(s, t, p, cap=3)
    for tk, r, tn, asg in [(0.2, 0.0, 0.3, True), (0.5, 0.6, 0.5, True), (0.3, 0.4, 0.7, False),
                           (0.1, 0.9, 0.1, False)]:
        assert (fd(tk, r, tn, asg) == decide(s, t, p, tk, r, tn, asg, cap=3)).all()
