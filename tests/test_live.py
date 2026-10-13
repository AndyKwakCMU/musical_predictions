import numpy as np

from musical.cnn import CadenceCNN, predict_proba
from musical.dataset import build_events
from musical.live import latest_signal
from musical.tsa import analyze
from tests.test_tsa import synthetic_bars


def test_signal_matches_backtest_dataset_for_same_bar():
    bars = synthetic_bars(n=700, calm=slice(450, 520))
    model = CadenceCNN(n_channels=8, ticker_vocab=["TEST"])
    X, _, meta = build_events(bars, "TEST", width=32, k=2.0, horizon=5)
    event_time = meta["timestamp"].iloc[-1]
    t = bars.index.get_loc(event_time)

    sig = latest_signal(bars.iloc[: t + 1], "TEST", model, width=32)

    expected = predict_proba(model, X[-1:], ["TEST"])[0]
    assert sig["state"] == "COMPRESSED"
    assert sig["timestamp"] == event_time
    np.testing.assert_allclose(sig["proba"], expected, atol=1e-6)
    assert sig["direction"] in (1, -1)


def test_no_signal_when_not_compressed():
    bars = synthetic_bars(n=700)
    tsa = analyze(bars)
    t = int(np.flatnonzero(tsa["state"].to_numpy() == "NOISE")[-1])
    model = CadenceCNN(n_channels=8, ticker_vocab=["TEST"])
    sig = latest_signal(bars.iloc[: t + 1], "TEST", model, width=32)
    assert sig["state"] == "NOISE"
    assert sig["direction"] == 0
    assert sig["proba"] is None


def test_incomplete_daily_bar_is_dropped_before_close():
    import pandas as pd
    from musical.live import drop_incomplete_daily
    bars = synthetic_bars(n=5)
    bars.index = pd.DatetimeIndex(["2026-10-07", "2026-10-08", "2026-10-09",
                                   "2026-10-12", "2026-10-13"])
    during = pd.Timestamp("2026-10-13 11:00", tz="America/New_York")
    after = pd.Timestamp("2026-10-13 16:30", tz="America/New_York")
    assert drop_incomplete_daily(bars, during).index[-1] == pd.Timestamp("2026-10-12")
    assert drop_incomplete_daily(bars, after).index[-1] == pd.Timestamp("2026-10-13")
