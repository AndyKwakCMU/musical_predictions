import numpy as np
import pandas as pd

from musical.tsa import COMPRESSED, NOISE, WARMUP, TSAConfig, analyze


def synthetic_bars(n=600, seed=0, calm=None):
    # Random walk; bars in `calm` (a slice) get 1/5 the volatility
    rng = np.random.default_rng(seed)
    scale = np.full(n, 0.02)
    if calm is not None:
        scale[calm] = 0.004
    rets = rng.normal(0, scale)
    close = 100 * np.exp(np.cumsum(rets))
    spread = close * scale
    idx = pd.date_range("2016-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": close, "high": close + spread, "low": close - spread,
        "close": close, "volume": 1000,
    }, index=idx)


def test_output_has_state_and_metrics():
    out = analyze(synthetic_bars())
    for col in ["state", "vol_pct", "bw_pct", "atr_ratio", "atr"]:
        assert col in out.columns
    assert set(out["state"].unique()) <= {COMPRESSED, NOISE, WARMUP}


def test_warmup_rows_are_marked_until_lookback_filled():
    cfg = TSAConfig()
    out = analyze(synthetic_bars(), cfg)
    assert (out["state"].iloc[: cfg.lookback] == WARMUP).all()
    assert (out["state"].iloc[cfg.lookback + cfg.atr_long:] != WARMUP).all()


def test_calm_stretch_after_volatile_history_is_compressed():
    bars = synthetic_bars(calm=slice(450, 520))
    out = analyze(bars)
    calm_states = out["state"].iloc[480:520]
    loud_states = out["state"].iloc[350:440]
    assert (calm_states == COMPRESSED).mean() > 0.8
    assert (loud_states == COMPRESSED).mean() < 0.3


def test_no_lookahead_truncating_future_does_not_change_past():
    bars = synthetic_bars(calm=slice(450, 520))
    full = analyze(bars)
    cut = analyze(bars.iloc[:470])
    pd.testing.assert_frame_equal(full.iloc[:470], cut)


def intraday_bars(days=160, per_day=26, seed=0, calm_days=None):
    # 15m-style bars with a strong U-shaped time-of-day volatility pattern
    rng = np.random.default_rng(seed)
    pos = np.tile(np.arange(per_day), days)
    u_shape = 1.0 + 3.0 * ((pos - (per_day - 1) / 2) / ((per_day - 1) / 2)) ** 2
    scale = 0.002 * u_shape
    if calm_days is not None:
        day = np.repeat(np.arange(days), per_day)
        scale[(day >= calm_days.start) & (day < calm_days.stop)] /= 5
    close = 100 * np.exp(np.cumsum(rng.normal(0, scale)))
    spread = close * scale
    sessions = pd.bdate_range("2020-01-02", periods=days)
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + 15 * i)
                            for d in sessions for i in range(per_day)])
    return pd.DataFrame({"open": close, "high": close + spread,
                         "low": close - spread, "close": close,
                         "volume": 1000}, index=idx)


def compressed_share_by_time(out):
    c = out[out["state"] != WARMUP]
    return (c["state"] == COMPRESSED).groupby(c.index.time).mean()


def test_naive_gate_is_distorted_by_time_of_day():
    # 20-bar windows ending mid-afternoon skip the volatile open/close,
    # so a naive gate flags those times far more often than the morning
    share = compressed_share_by_time(analyze(intraday_bars()))
    assert share.max() - share.min() > 0.25


def test_seasonal_gate_is_flat_across_time_of_day():
    cfg = TSAConfig(seasonal_days=40)
    share = compressed_share_by_time(analyze(intraday_bars(), cfg))
    assert share.max() - share.min() < 0.15


def test_seasonal_gate_still_finds_a_calm_stretch():
    cfg = TSAConfig(seasonal_days=40)
    out = analyze(intraday_bars(calm_days=slice(120, 130)), cfg)
    sessions = pd.Series(out.index.normalize()).rank(method="dense").to_numpy() - 1
    calm = out["state"].to_numpy()[(sessions >= 122) & (sessions < 130)]
    before = out["state"].to_numpy()[(sessions >= 90) & (sessions < 118)]
    assert (calm == COMPRESSED).mean() > 0.6
    assert (before == COMPRESSED).mean() < 0.3


def test_seasonal_gate_has_no_lookahead():
    cfg = TSAConfig(seasonal_days=40)
    bars = intraday_bars(calm_days=slice(120, 130))
    full = analyze(bars, cfg)
    cut = analyze(bars.iloc[:3000], cfg)
    pd.testing.assert_frame_equal(full.iloc[:3000], cut)
