import numpy as np
import pandas as pd

from ber.splits import sample_cities, state_group


def test_sample_cities_budget_and_min_one():
    # one giant city and many small ones: the giant must not blow the 5% budget of its tier
    rows = [("IN", "big", 5000)] + [("IN", f"m{i}", 1) for i in range(1)] + \
           [("IN", f"s{i}", 2) for i in range(2000)] + [("US", f"u{i}", 3) for i in range(1000)]
    g1 = pd.DataFrame([(c, city) for c, city, n in rows for _ in range(n)], columns=["country", "city"])
    ch = sample_cities(g1, np.random.default_rng(0), frac=0.05)
    cnt = g1.groupby(["country", "city"]).size()
    share = cnt / cnt.groupby(level=0).transform("sum")
    tier = np.select([share >= 0.01, share >= 0.0005], ["metro", "mid"], "small")
    tier_tot = pd.Series(cnt.values, index=pd.MultiIndex.from_arrays([cnt.index.get_level_values(0), tier])) \
        .groupby(level=[0, 1]).sum()
    got = ch.groupby(["country", "tier"]).agg(n=("n_s1", "sum"), k=("city", "size"))
    assert set(got.index) == set(tier_tot.index)       # every tier contributes
    for key, row in got.iterrows():
        assert row.k == 1 or row.n <= 0.05 * tier_tot[key]   # within budget unless the forced single city
    assert got.loc[("IN", "metro"), "k"] == 1          # the giant city is the forced pick of its own tier
    assert set(ch.country) == {"IN", "US"}


def test_state_group():
    assert state_group("telangana") == state_group("andhra pradesh")
    assert state_group("kerala") == "kerala"
