import numpy as np
import pandas as pd
import pytest

from musical.experiment import (
    direction_from_proba, move_proba, perm_pvalue, quantile_cutoff, trade_r,
)


def meta_rows(rows):
    # rows: (label, entry, upper, exit_price, bars_held)
    df = pd.DataFrame(rows, columns=["label", "entry", "upper", "exit_price", "bars_held"])
    df["lower"] = 2 * df["entry"] - df["upper"]
    return df


def test_trade_r_scores_barrier_outcomes_by_direction():
    meta = meta_rows([
        ("UP", 100, 102, 102, 3),
        ("DOWN", 100, 102, 98, 3),
        ("UP", 100, 102, 102, 3),
    ])
    r = trade_r(meta, np.array([1, 1, -1]), horizon=10)
    assert r.tolist() == [1.0, -1.0, -1.0]


def test_trade_r_time_exit_uses_partial_move():
    meta = meta_rows([("NONE", 100, 102, 101, 10)])
    assert trade_r(meta, np.array([-1]), horizon=10)[0] == pytest.approx(-0.5)


def test_trade_r_same_bar_ambiguity_counts_as_stop():
    meta = meta_rows([("NONE", 100, 102, 100.5, 4)])
    assert trade_r(meta, np.array([1]), horizon=10)[0] == -1.0


def test_direction_picks_more_likely_barrier():
    proba = np.array([[0.5, 0.2, 0.3], [0.1, 0.6, 0.3]])
    assert direction_from_proba(proba).tolist() == [-1, 1]


def test_move_proba_is_up_plus_down():
    # columns are DOWN, NONE, UP
    proba = np.array([[0.2, 0.7, 0.1], [0.4, 0.1, 0.5]])
    np.testing.assert_allclose(move_proba(proba), [0.3, 0.9])


def test_quantile_cutoff_keeps_top_fraction():
    vals = np.arange(100, dtype=float)
    c = quantile_cutoff(vals, 0.2)
    assert (vals >= c).sum() == 20


def test_quantile_cutoff_of_one_keeps_everything():
    vals = np.array([0.5, 0.1, 0.9])
    assert (vals >= quantile_cutoff(vals, 1.0)).all()


def test_perm_pvalue_counts_null_draws_at_or_above_observed():
    null = np.array([0.0, 0.1, 0.2, 0.3])
    assert perm_pvalue(0.2, null) == pytest.approx(0.5)
    assert perm_pvalue(1.0, null) == 0.0
