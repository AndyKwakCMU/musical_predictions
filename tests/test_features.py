import numpy as np
import pandas as pd

from musical.features import CLOCK_COLS, PRICE_COLS, add_context, bar_features


def intraday_bars(days=60, per_day=26, seed=0, start="2020-01-02",
                  sessions=None, short_days=()):
    # 15m bars from 09:30; `short_days` (session indexes) get only 14 bars
    rng = np.random.default_rng(seed)
    sessions = sessions if sessions is not None else pd.bdate_range(start, periods=days)
    idx = [d + pd.Timedelta(minutes=570 + 15 * i)
           for k, d in enumerate(sessions)
           for i in range(14 if k in short_days else per_day)]
    n = len(idx)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.001, n))
    spread = close * 0.002
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread, "close": close,
        "volume": rng.integers(1_000, 5_000, n).astype(float),
    }, index=pd.DatetimeIndex(idx, name="timestamp"))


def after_warmup(feats, bars, sessions=41):
    first_day = bars.index.normalize().unique()[sessions]
    return feats[feats.index >= first_day]


def test_columns_are_price_then_clock():
    feats = bar_features(intraday_bars())
    assert list(feats.columns) == PRICE_COLS + CLOCK_COLS


def test_features_are_causal():
    bars = intraday_bars(days=150)
    full = bar_features(bars)
    cut = bar_features(bars.iloc[:3000])
    pd.testing.assert_frame_equal(full.iloc[:3000], cut)


def test_first_bar_flag_marks_each_session_start_including_after_holiday():
    days = pd.bdate_range("2020-01-02", periods=10)
    days = days.delete(4)                       # a holiday gap mid-week
    bars = intraday_bars(sessions=days)
    feats = bar_features(bars)
    first = feats.index[feats["first_bar"] == 1.0]
    assert list(first) == [d + pd.Timedelta(minutes=570) for d in days]
    assert feats["first_bar"].sum() == len(days)


def test_half_day_session_has_finite_features():
    bars = intraday_bars(days=60, short_days={50})
    feats = after_warmup(bar_features(bars), bars)
    assert np.isfinite(feats.to_numpy()).all()


def test_zero_volume_and_zero_range_bars_stay_finite():
    bars = intraday_bars(days=60)
    flat = slice(1200, 1230)                    # 30 bars: no range, no volume
    price = bars["close"].iloc[1199]
    for c in ["open", "high", "low", "close"]:
        bars.iloc[flat, bars.columns.get_loc(c)] = price
    bars.iloc[flat, bars.columns.get_loc("volume")] = 0.0
    feats = after_warmup(bar_features(bars), bars)
    assert np.isfinite(feats.to_numpy()).all()


def test_vol_surprise_is_zero_for_constant_volume_profile():
    bars = intraday_bars(days=60)
    slot = bars.index.hour * 60 + bars.index.minute
    bars["volume"] = (1000 + 10 * slot).astype(float)   # U-free but slot-dependent
    feats = after_warmup(bar_features(bars), bars)
    assert np.allclose(feats["vol_surprise"], 0.0, atol=1e-9)


def test_context_aligns_on_timestamp_and_fills_missing_with_zero():
    bars = intraday_bars(days=60)
    feats = bar_features(bars)
    spy = bar_features(intraday_bars(days=60, seed=1))
    missing = spy.index[1500]
    spy = spy.drop(index=missing)
    out = add_context(feats, {"SPY": spy})
    assert len(out) == len(feats)
    assert out.loc[missing, "SPY_ret_atr"] == 0.0
    assert out.loc[missing, "SPY_vol_surprise"] == 0.0
    other = spy.index[1540]
    assert out.loc[other, "SPY_ret_atr"] == spy.loc[other, "ret_atr"]
