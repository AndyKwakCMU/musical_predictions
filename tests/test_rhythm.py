import numpy as np
import pandas as pd

from musical.rhythm import onset_strength, tempo


def pulse(n, period, seed=0, noise=0.1, start=0):
    # onset envelope: spikes every `period` bars plus gaussian noise
    rng = np.random.default_rng(seed)
    x = rng.normal(0, noise, n)
    x[start::period] += 1.0
    return x


def as_series(x):
    return pd.Series(x, index=pd.RangeIndex(len(x)))


def random_feats(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ret_atr": rng.normal(0, 0.5, n),
        "range_atr": rng.gamma(2.0, 0.5, n),
        "vol_surprise": rng.normal(0, 0.3, n),
    }, index=pd.date_range("2020-01-02 09:30", periods=n, freq="15min"))


def test_tempo_recovers_period_of_synthetic_pulse():
    out = tempo(as_series(pulse(3000, 5)))
    ready = out.dropna()
    assert ready["period"].median() == 5
    assert ready["strength"].median() > 0.5


def test_tempo_strength_is_low_for_white_noise():
    rng = np.random.default_rng(1)
    out = tempo(as_series(rng.normal(0, 1, 3000)))
    assert out["strength"].dropna().median() < 0.2


def test_tempo_tracks_a_slowdown():
    x = np.r_[pulse(2000, 4, seed=2), pulse(2000, 7, seed=3)]
    out = tempo(as_series(x))
    assert out["period"].iloc[-500:].median() == 7


def test_tempo_is_nan_before_window_is_filled():
    out = tempo(as_series(pulse(1000, 5)), window=520)
    assert out.iloc[:519].isna().all().all()


def test_onset_and_tempo_are_causal():
    feats = random_feats()
    full = onset_strength(feats)
    cut = onset_strength(feats.iloc[:2000])
    pd.testing.assert_series_equal(full.iloc[:2000], cut)
    t_full = tempo(full)
    t_cut = tempo(cut)
    pd.testing.assert_frame_equal(t_full.iloc[:2000], t_cut)


def test_onset_is_high_on_big_moves_volume_bursts_and_turns():
    feats = random_feats()
    feats.iloc[2500, feats.columns.get_loc("ret_atr")] = 6.0
    feats.iloc[2500, feats.columns.get_loc("vol_surprise")] = 3.0
    onset = onset_strength(feats)
    assert onset.iloc[2500] > onset.iloc[600:].quantile(0.99)


def test_persistent_but_aperiodic_activity_is_not_a_short_pulse():
    # volatility clustering: AR(1) activity decays smoothly with lag and
    # has no beat; it must not read as "period = min_lag"
    rng = np.random.default_rng(4)
    x = np.zeros(3000)
    for t in range(1, 3000):
        x[t] = 0.8 * x[t - 1] + rng.normal()
    out = tempo(as_series(x)).iloc[600:]
    assert (out["period"] == 2).mean() < 0.2
    assert out["strength"].median() < 0.2
