import numpy as np
import pandas as pd

from musical.labels import DOWN, NONE, UP, triple_barrier


def bars_from(rows):
    # rows: list of (open, high, low, close)
    idx = pd.date_range("2020-01-01", periods=len(rows), freq="B")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1
    return df


def flat(n, px=100.0):
    return [(px, px + 0.1, px - 0.1, px)] * n


def test_upper_barrier_hit_first_is_up():
    # Event at bar 0; entry at bar 1 open = 100; barriers 100 +/- 2*1
    rows = flat(2) + [(100, 102.5, 99.5, 102)] + flat(5, 102)
    atr = pd.Series(1.0, index=bars_from(rows).index)
    out = triple_barrier(bars_from(rows), atr, k=2.0, horizon=5)
    assert out.loc[out.index[0], "label"] == UP
    assert out.loc[out.index[0], "bars_held"] == 2


def test_lower_barrier_hit_first_is_down():
    rows = flat(3) + [(100, 100.2, 97.0, 97.5)] + flat(5, 97.5)
    atr = pd.Series(1.0, index=bars_from(rows).index)
    out = triple_barrier(bars_from(rows), atr, k=2.0, horizon=5)
    assert out.loc[out.index[0], "label"] == DOWN


def test_no_barrier_within_horizon_is_none():
    rows = flat(10)
    atr = pd.Series(1.0, index=bars_from(rows).index)
    out = triple_barrier(bars_from(rows), atr, k=2.0, horizon=5)
    assert out.loc[out.index[0], "label"] == NONE
    assert out.loc[out.index[0], "bars_held"] == 5


def test_both_barriers_in_one_bar_is_none():
    rows = flat(2) + [(100, 103, 97, 100)] + flat(5)
    atr = pd.Series(1.0, index=bars_from(rows).index)
    out = triple_barrier(bars_from(rows), atr, k=2.0, horizon=5)
    assert out.loc[out.index[0], "label"] == NONE


def test_events_without_full_horizon_are_dropped():
    rows = flat(10)
    atr = pd.Series(1.0, index=bars_from(rows).index)
    out = triple_barrier(bars_from(rows), atr, k=2.0, horizon=5)
    # event t needs bars t+1..t+5 -> last valid event is index 4
    assert len(out) == 5


def test_only_requested_events_are_labeled():
    rows = flat(10)
    bars = bars_from(rows)
    atr = pd.Series(1.0, index=bars.index)
    events = bars.index[[1, 3]]
    out = triple_barrier(bars, atr, k=2.0, horizon=5, events=events)
    assert list(out.index) == list(events)


def test_result_reports_entry_and_return_in_atr_units():
    rows = flat(2) + [(100, 102.5, 99.5, 102)] + flat(5, 102)
    atr = pd.Series(1.0, index=bars_from(rows).index)
    out = triple_barrier(bars_from(rows), atr, k=2.0, horizon=5)
    row = out.iloc[0]
    assert row["entry"] == 100
    assert np.isclose(row["max_excursion_atr"], 2.5)
