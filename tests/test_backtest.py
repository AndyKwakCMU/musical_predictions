import numpy as np
import pandas as pd
import pytest

from musical.backtest import CostModel, metrics, run_portfolio, simulate_trades
from tests.test_labels import bars_from, flat

FREE = CostModel(commission_per_share=0.0, commission_min=0.0, slippage_bps=0.0)


def one(rows, direction, k=2.0, horizon=5, cost=FREE):
    bars = bars_from(rows)
    atr = pd.Series(1.0, index=bars.index)
    signals = pd.DataFrame({
        "ticker": ["X"], "timestamp": [bars.index[0]], "direction": [direction],
    })
    trades = simulate_trades({"X": bars}, {"X": atr}, signals, k, horizon, cost)
    return trades.iloc[0]


def test_long_hits_target():
    t = one(flat(2) + [(100, 102.5, 99.5, 102)] + flat(5, 102), +1)
    assert t["exit_reason"] == "target"
    assert t["exit"] == 102
    assert t["r_multiple"] == pytest.approx(1.0)


def test_long_gap_through_stop_fills_at_open():
    t = one(flat(2) + [(96, 96.5, 95, 96)] + flat(5, 96), +1)
    assert t["exit_reason"] == "stop"
    assert t["exit"] == 96
    assert t["r_multiple"] == pytest.approx(-2.0)


def test_same_bar_stop_and_target_assumes_stop():
    t = one(flat(2) + [(100, 103, 97, 100)] + flat(5), +1)
    assert t["exit_reason"] == "stop"
    assert t["r_multiple"] == pytest.approx(-1.0)


def test_time_exit_at_close_of_horizon():
    rows = flat(5) + [(100, 100.5, 99.5, 100.4)] + flat(3, 100.4)
    t = one(rows, +1)
    assert t["exit_reason"] == "time"
    assert t["exit"] == pytest.approx(100.4)


def test_short_hits_target_on_way_down():
    t = one(flat(3) + [(100, 100.2, 97.0, 97.5)] + flat(5, 97.5), -1)
    assert t["exit_reason"] == "target"
    assert t["r_multiple"] == pytest.approx(1.0)


def test_overlapping_signal_on_same_ticker_is_skipped():
    bars = bars_from(flat(20))
    atr = pd.Series(1.0, index=bars.index)
    signals = pd.DataFrame({
        "ticker": ["X", "X", "X"],
        "timestamp": bars.index[[0, 2, 8]],
        "direction": [1, 1, 1],
    })
    trades = simulate_trades({"X": bars}, {"X": atr}, signals, 2.0, 5, FREE)
    assert list(trades["signal_time"]) == list(bars.index[[0, 8]])


def test_slippage_worsens_both_fills():
    cost = CostModel(commission_per_share=0.0, commission_min=0.0, slippage_bps=10)
    t = one(flat(2) + [(100, 102.5, 99.5, 102)] + flat(5, 102), +1, cost=cost)
    assert t["entry_fill"] == pytest.approx(100 * 1.001)
    assert t["exit_fill"] == pytest.approx(102 * 0.999)


def test_portfolio_risks_fraction_of_equity_and_caps_positions():
    trades = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "signal_time": pd.to_datetime(["2020-01-01"] * 3),
        "entry_time": pd.to_datetime(["2020-01-02"] * 3),
        "exit_time": pd.to_datetime(["2020-01-05"] * 3),
        "direction": [1, 1, 1],
        "entry_fill": [100.0] * 3,
        "exit_fill": [102.0] * 3,
        "risk_per_share": [2.0] * 3,
    })
    log, equity = run_portfolio(trades, FREE, start_equity=100_000,
                                risk_frac=0.01, max_positions=2)
    assert len(log) == 2                      # third trade: no slot
    # risk $1000 / $2 = 500 shares, notional $50k <= equity/2 cap
    assert (log["shares"] == 500).all()
    assert equity.iloc[-1] == pytest.approx(100_000 + 2 * 500 * 2.0)


def test_portfolio_notional_cap_limits_shares():
    trades = pd.DataFrame({
        "ticker": ["A"], "signal_time": pd.to_datetime(["2020-01-01"]),
        "entry_time": pd.to_datetime(["2020-01-02"]),
        "exit_time": pd.to_datetime(["2020-01-05"]), "direction": [1],
        "entry_fill": [100.0], "exit_fill": [100.0], "risk_per_share": [0.1],
    })
    log, _ = run_portfolio(trades, FREE, start_equity=100_000,
                           risk_frac=0.01, max_positions=10)
    assert log["shares"].iloc[0] == 100      # $10k cap / $100


def test_commission_minimum_applies():
    cost = CostModel(commission_per_share=0.0035, commission_min=0.35, slippage_bps=0)
    assert cost.commission(10) == pytest.approx(0.35)
    assert cost.commission(1000) == pytest.approx(3.5)


def test_metrics_report_core_numbers():
    log = pd.DataFrame({"pnl": [100.0, -50.0, 200.0], "r_net": [1.0, -0.5, 2.0]})
    equity = pd.Series([100_000, 100_100, 100_050, 100_250],
                       index=pd.date_range("2020-01-01", periods=4, freq="B"))
    m = metrics(log, equity)
    assert m["trades"] == 3
    assert m["hit_rate"] == pytest.approx(2 / 3)
    assert m["expectancy_r"] == pytest.approx(2.5 / 3)
    assert m["max_drawdown"] == pytest.approx(-50 / 100_100)


def test_entry_fills_at_next_bar_open():
    rows = [(100, 100.1, 99.9, 100)] + [(101, 101.1, 100.9, 101)] + flat(5, 101)
    t = one(rows, +1)
    assert t["entry"] == 101
    assert t["entry_time"] == bars_from(rows).index[1]
