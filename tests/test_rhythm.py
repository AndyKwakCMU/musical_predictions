import numpy as np
import pandas as pd

from musical.rhythm import meter, onset_strength, tempo, track_beats


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


def const(n, value):
    return as_series(np.full(n, float(value)))


def beat_positions(beats):
    return np.flatnonzero(beats.to_numpy())


def test_beats_follow_a_steady_pulse():
    beats = track_beats(as_series(pulse(3000, 6)), const(3000, 6))
    gaps = np.diff(beat_positions(beats)[beat_positions(beats) > 600])
    assert np.median(gaps) == 6


def test_beats_keep_time_through_a_missing_onset():
    x = pulse(3000, 6)
    x[1500] -= 1.0                       # the spike at 1500 goes silent
    pos = beat_positions(track_beats(as_series(x), const(3000, 6)))
    assert np.min(np.abs(pos - 1500)) <= 2


def test_beats_carry_the_last_tempo_through_no_pulse_stretches():
    period = const(3000, 6)
    period.iloc[2000:2100] = np.nan      # tempo has no peak for a while
    pos = beat_positions(track_beats(as_series(pulse(3000, 6)), period))
    assert ((pos > 2000) & (pos < 2100)).sum() >= 12


def test_meter_finds_three_beat_accent_pattern():
    x = pulse(4000, 4)
    x[::12] += 2.0                       # every 3rd spike is 3x larger
    s = as_series(x)
    beats = track_beats(s, const(4000, 4))
    out = meter(s, beats).iloc[1000:]
    assert out["beats_per_measure"].mode().iloc[0] == 3
    down = np.flatnonzero(out["downbeat"].to_numpy()) + 1000
    assert np.mean(down % 12 == 0) > 0.8


def test_beats_and_meter_are_causal():
    s = as_series(pulse(3000, 5, seed=7))
    period = tempo(s)["period"]
    full_b = track_beats(s, period)
    cut_b = track_beats(s.iloc[:2000], period.iloc[:2000])
    pd.testing.assert_series_equal(full_b.iloc[:2000], cut_b)
    pd.testing.assert_frame_equal(meter(s, full_b).iloc[:2000],
                                  meter(s.iloc[:2000], cut_b))


def test_half_day_sessions_do_not_break_tracking():
    days = pd.bdate_range("2020-01-02", periods=60)
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + 15 * i)
                            for k, d in enumerate(days)
                            for i in range(14 if k == 40 else 26)])
    s = pd.Series(pulse(len(idx), 5, seed=8), index=idx)
    beats = track_beats(s, tempo(s)["period"])
    out = meter(s, beats)
    assert beats.sum() > 0 and len(out) == len(idx)
