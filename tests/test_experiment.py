import numpy as np
import pandas as pd
import pytest

from musical.experiment import direction_from_proba, trade_r


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
