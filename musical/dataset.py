##############################################################################

#           dataset
#           Andy Kwak

##############################################################################

#       Builds the CNN's "cadence library": one sample per COMPRESSED bar.
#       Input is the previous `width` bars as a (channels, width) array,
#       normalized in ATR units relative to the event bar so every ticker
#       and price level looks alike. Label is the triple-barrier outcome.


#       @requires protocol bars for one ticker
#       @ensures X (N, C, W) float32, y (N,) int64, meta DataFrame

##############################################################################

import numpy as np
import pandas as pd

from musical.labels import DOWN, NONE, UP, triple_barrier
from musical.tsa import COMPRESSED, analyze

CHANNELS = [
    "ret_atr",      # close-to-close move / ATR(t)
    "range_atr",    # high - low / ATR(t)
    "body_atr",     # close - open / ATR(t)
    "pos_atr",      # close - close(t) / ATR(t): shape of the path into t
    "vol_rel",      # log volume vs window mean
    "vol_pct",      # TSA metrics carried along as extra "voices"
    "bw_pct",
    "atr_ratio",
]
LABEL_TO_INT = {DOWN: 0, NONE: 1, UP: 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def make_window(bars, tsa, t, width):
    #@requires 0 <= t < len(bars), tsa = analyze(bars)
    #@ensures \result shape (len(CHANNELS), width), built from bars <= t;
    #         may contain NaN if the window reaches into TSA warm-up
    lo = t - width + 1
    if lo < 1:
        return np.full((len(CHANNELS), width), np.nan, dtype=np.float32)

    w = bars.iloc[lo:t + 1]
    prev_close = bars["close"].iloc[lo - 1:t].to_numpy(float)
    atr = tsa["atr"].iloc[t]
    c_t = bars["close"].iloc[t]
    close = w["close"].to_numpy(float)
    volume = w["volume"].to_numpy(float) + 1.0
    m = tsa.iloc[lo:t + 1]

    x = np.stack([
        (close - prev_close) / atr,
        (w["high"].to_numpy(float) - w["low"].to_numpy(float)) / atr,
        (close - w["open"].to_numpy(float)) / atr,
        (close - c_t) / atr,
        np.log(volume / volume.mean()),
        m["vol_pct"].to_numpy(float),
        m["bw_pct"].to_numpy(float),
        m["atr_ratio"].to_numpy(float),
    ])
    return x.astype(np.float32)


def build_events(bars, ticker, width, k, horizon):
    #@requires bars time-ordered for a single ticker
    #@ensures one sample per COMPRESSED bar that has a finite window
    #         and a full labeling horizon
    tsa = analyze(bars)
    events = tsa.index[tsa["state"] == COMPRESSED]
    lab = triple_barrier(bars, tsa["atr"], k, horizon, events=events)
    if lab.empty:
        return (np.empty((0, len(CHANNELS), width), np.float32),
                np.empty(0, np.int64), pd.DataFrame())

    positions = bars.index.get_indexer(lab.index)
    X, keep = [], []
    for i, t in enumerate(positions):
        x = make_window(bars, tsa, t, width)
        if np.isfinite(x).all():
            X.append(x)
            keep.append(i)

    lab = lab.iloc[keep]
    positions = positions[keep]
    meta = lab.reset_index()
    meta.insert(0, "ticker", ticker)
    meta["horizon_end"] = bars.index[positions + horizon]
    meta["atr"] = tsa["atr"].iloc[positions].to_numpy()
    y = meta["label"].map(LABEL_TO_INT).to_numpy(np.int64)
    X = np.stack(X) if X else np.empty((0, len(CHANNELS), width), np.float32)
    return X, y, meta


def time_split(meta, val_start, test_start, test_end=None):
    #@requires meta has timestamp and horizon_end columns
    #@ensures \result labels each row train / val / test / embargo / out;
    #         a sample whose label horizon crosses into the next split is
    #         embargoed so labels never leak across boundaries
    ts = pd.to_datetime(meta["timestamp"])
    end = pd.to_datetime(meta["horizon_end"])
    val_start, test_start = pd.Timestamp(val_start), pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end) if test_end is not None else None

    split = np.full(len(meta), "out", dtype=object)
    in_train = ts < val_start
    in_val = (ts >= val_start) & (ts < test_start)
    in_test = ts >= test_start
    if test_end is not None:
        in_test &= ts < test_end

    split[in_train] = np.where(end[in_train] < val_start, "train", "embargo")
    split[in_val] = np.where(end[in_val] < test_start, "val", "embargo")
    if test_end is not None:
        split[in_test] = np.where(end[in_test] < test_end, "test", "embargo")
    else:
        split[in_test] = "test"
    return pd.Series(split, index=meta.index)
