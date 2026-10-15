##############################################################################

#           features
#           Andy Kwak

##############################################################################

#       What the "listener" hears on every 15m bar: price motion in ATR
#       units, volume relative to what is normal for that time of day,
#       and the clock itself (time of day, day of week, where we are in
#       the session). The clock is an input to learn from, not something
#       to strip out. Market context (SPY/QQQ) is joined on timestamp:
#       the rest of the orchestra.
#
#       Every value at bar t uses bars <= t (no lookahead).


#       @requires protocol bars (open,high,low,close,volume), time-ordered
#       @ensures one row per bar: PRICE_COLS + CLOCK_COLS

##############################################################################

import numpy as np
import pandas as pd

from musical.tsa import true_range

PRICE_COLS = ["ret_atr", "range_atr", "body_atr", "vol_surprise", "gap_atr"]
CLOCK_COLS = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "session_pos", "first_bar"]
CONTEXT_FIELDS = ["ret_atr", "vol_surprise"]


def safe_atr(bars, atr_n):
    #@ensures \result[t] > 0 once any range has been seen; a flat stretch
    #         keeps the last positive ATR instead of dividing by zero
    atr = true_range(bars).rolling(atr_n).mean()
    return atr.where(atr > 0).ffill()


def seasonal_log_volume_mean(bars, seasonal_days):
    #@ensures \result[t] = mean log(volume+1) at t's time of day over the
    #         previous `seasonal_days` sessions (today excluded)
    logv = np.log(bars["volume"].clip(lower=0) + 1.0)
    return logv.groupby(bars.index.time, group_keys=False).transform(
        lambda s: s.shift(1).rolling(seasonal_days).mean())


def bar_features(bars, seasonal_days=40, atr_n=14):
    #@requires bars indexed by exchange-local timestamp, RTH 15m bars
    #@ensures \result indexed like bars, columns PRICE_COLS + CLOCK_COLS
    idx = bars.index
    atr = safe_atr(bars, atr_n)
    prev_close = bars["close"].shift(1)
    date = pd.Series(idx.normalize(), index=idx)
    first = (date != date.shift(1)).astype(float)

    logv = np.log(bars["volume"].clip(lower=0) + 1.0)
    minutes = (idx.hour * 60 + idx.minute - 570) / 390.0
    dow = idx.dayofweek / 5.0

    out = pd.DataFrame({
        "ret_atr": (bars["close"] - prev_close) / atr,
        "range_atr": (bars["high"] - bars["low"]) / atr,
        "body_atr": (bars["close"] - bars["open"]) / atr,
        "vol_surprise": logv - seasonal_log_volume_mean(bars, seasonal_days),
        "gap_atr": ((bars["open"] - prev_close) / atr).where(first == 1.0, 0.0),
        "tod_sin": np.sin(2 * np.pi * minutes),
        "tod_cos": np.cos(2 * np.pi * minutes),
        "dow_sin": np.sin(2 * np.pi * dow),
        "dow_cos": np.cos(2 * np.pi * dow),
        "session_pos": date.groupby(date).cumcount() / 25.0,
        "first_bar": first,
    }, index=idx)
    return out[PRICE_COLS + CLOCK_COLS]


def add_context(feats, context):
    #@requires context maps a name (e.g. "SPY") to that name's bar_features
    #@ensures \result = feats plus {name}_ret_atr, {name}_vol_surprise,
    #         aligned on timestamp; missing or warm-up values are 0.0
    out = feats.copy()
    for name, ctx in context.items():
        aligned = ctx[CONTEXT_FIELDS].reindex(feats.index)
        for field in CONTEXT_FIELDS:
            out[f"{name}_{field}"] = aligned[field].fillna(0.0).to_numpy()
    return out
