import numpy as np
import pandas as pd

from musical.dataset import (
    CHANNELS, LABEL_TO_INT, build_events, make_window, time_split,
)
from musical.tsa import analyze
from tests.test_tsa import synthetic_bars


def test_window_shape_matches_channels_and_width():
    bars = synthetic_bars()
    tsa = analyze(bars)
    x = make_window(bars, tsa, t=400, width=32)
    assert x.shape == (len(CHANNELS), 32)
    assert np.isfinite(x).all()


def test_window_uses_only_bars_up_to_t():
    bars = synthetic_bars()
    tsa = analyze(bars)
    before = make_window(bars, tsa, t=400, width=32)

    future = bars.copy()
    future.iloc[401:, :4] *= 3.0
    after = make_window(future, analyze(future), t=400, width=32)
    np.testing.assert_allclose(before, after)


def test_build_events_only_uses_compressed_bars_with_labels():
    bars = synthetic_bars(calm=slice(450, 520))
    X, y, meta = build_events(bars, "TEST", width=32, k=2.0, horizon=5)
    tsa = analyze(bars)
    assert len(X) == len(y) == len(meta) > 0
    assert (tsa.loc[meta["timestamp"], "state"] == "COMPRESSED").all()
    assert set(np.unique(y)) <= set(LABEL_TO_INT.values())
    assert (meta["horizon_end"] > meta["timestamp"]).all()


def test_time_split_embargoes_labels_that_cross_boundaries():
    ts = pd.date_range("2019-12-20", periods=20, freq="B")
    meta = pd.DataFrame({
        "timestamp": ts,
        "horizon_end": ts + pd.Timedelta(days=7),
    })
    split = time_split(meta, val_start="2020-01-01", test_start="2020-01-10")
    train = meta[split == "train"]
    val = meta[split == "val"]
    assert (train["horizon_end"] < pd.Timestamp("2020-01-01")).all()
    assert (val["timestamp"] >= pd.Timestamp("2020-01-01")).all()
    assert (val["horizon_end"] < pd.Timestamp("2020-01-10")).all()
    assert "embargo" in set(split)
