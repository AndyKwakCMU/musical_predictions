import numpy as np
import pandas as pd

from musical.rhythm import tempo
from musical.surrogates import clock_shuffle


def session_frame(days=120, per_day=26, seed=0):
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2020-01-02", periods=days)
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + 15 * i)
                            for d in sessions for i in range(per_day)])
    n = len(idx)
    slot = np.tile(np.arange(per_day), days)
    onset = rng.normal(0, 0.1, n)
    onset[::5] += 1.0                                  # pulse of 5 bars, not clock-locked
    u_shape = 1 + 3 * ((slot - 12.5) / 12.5) ** 2      # clock-driven volume
    return pd.DataFrame({
        "onset": onset,
        "volume": u_shape * rng.lognormal(0, 0.2, n),
        "tag": np.arange(n),
    }, index=idx)


def test_clock_shuffle_preserves_each_slot_distribution():
    df = session_frame()
    out = clock_shuffle(df, ["onset", "volume"], np.random.default_rng(0))
    for _, grp in df.groupby(df.index.time):
        shuffled = out.loc[grp.index]
        assert np.array_equal(np.sort(grp["onset"]), np.sort(shuffled["onset"]))
    assert out.index.equals(df.index)


def test_clock_shuffle_moves_columns_together_and_leaves_others():
    df = session_frame()
    df["onset_copy"] = df["onset"]
    out = clock_shuffle(df, ["onset", "onset_copy"], np.random.default_rng(1))
    assert np.array_equal(out["onset"], out["onset_copy"])
    assert np.array_equal(out["tag"], df["tag"])
    assert not np.array_equal(out["onset"], df["onset"])


def test_clock_shuffle_destroys_within_session_periodicity():
    df = session_frame()
    real = tempo(df["onset"])["strength"].dropna().median()
    out = clock_shuffle(df, ["onset"], np.random.default_rng(2))
    fake = tempo(out["onset"])["strength"].dropna().median()
    assert real > 0.5
    assert fake < 0.2


def test_clock_shuffle_keeps_clock_driven_pattern():
    df = session_frame()
    out = clock_shuffle(df, ["volume"], np.random.default_rng(3))
    before = df["volume"].groupby(df.index.time).mean()
    after = out["volume"].groupby(out.index.time).mean()
    assert np.allclose(before, after)
