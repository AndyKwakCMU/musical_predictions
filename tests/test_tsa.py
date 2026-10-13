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
