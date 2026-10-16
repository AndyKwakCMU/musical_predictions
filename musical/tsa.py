##############################################################################

#           tsa
#           Andy Kwak

##############################################################################

#       Time-Series Analysis: the "are we in the V function?" gate.
#       Labels each bar COMPRESSED (volatility is unusually quiet relative
#       to this ticker's own recent history, so a breakout setup may be
#       forming) or NOISE, and emits continuous metrics the CNN can use.
#
#       Every value at bar t uses only bars <= t (no lookahead).


#       @requires protocol bars (open,high,low,close,volume), time-ordered
#       @ensures one row per bar: state + metrics

##############################################################################

from dataclasses import dataclass

import numpy as np
import pandas as pd

COMPRESSED = "COMPRESSED"
NOISE = "NOISE"
WARMUP = "WARMUP"


@dataclass(frozen=True)
class TSAConfig:
    n_vol: int = 20          # realized-vol window
    bb_n: int = 20           # Bollinger window
    atr_n: int = 14          # ATR used as the unit for barriers
    atr_short: int = 5
    atr_long: int = 50
    lookback: int = 252      # history the percentiles are ranked against
    pct_threshold: float = 0.20
    # Intraday: rank each bar only against the same time of day over the
    # last `seasonal_days` sessions, so the open/close volatility U-shape
    # isn't mistaken for compression. None = daily bars (rank vs lookback).
    seasonal_days: int | None = None


def true_range(bars):
    #@ensures \result[t] uses bars t and t-1 only
    prev_close = bars["close"].shift(1)
    return pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def rank_pct(series, cfg):
    #@ensures \result[t] is series[t]'s percentile among its own recent
    #         history (same time of day only, if cfg.seasonal_days)
    if cfg.seasonal_days is None:
        return series.rolling(cfg.lookback).rank(pct=True)
    return series.groupby(series.index.time, group_keys=False).transform(
        lambda s: s.rolling(cfg.seasonal_days).rank(pct=True))


def analyze(bars, cfg=TSAConfig()):
    #@requires bars has open,high,low,close columns, time-ordered
    #@ensures \result indexed like bars with columns
    #         state, vol_pct, bw_pct, atr_ratio, atr
    close = bars["close"]
    logret = np.log(close).diff()

    rv = logret.rolling(cfg.n_vol).std()
    vol_pct = rank_pct(rv, cfg)

    sma = close.rolling(cfg.bb_n).mean()
    bandwidth = 4 * close.rolling(cfg.bb_n).std() / sma
    bw_pct = rank_pct(bandwidth, cfg)

    tr = true_range(bars)
    atr = tr.rolling(cfg.atr_n).mean()
    atr_ratio = tr.rolling(cfg.atr_short).mean() / tr.rolling(cfg.atr_long).mean()

    out = pd.DataFrame({
        "vol_pct": vol_pct,
        "bw_pct": bw_pct,
        "atr_ratio": atr_ratio,
        "atr": atr,
    }, index=bars.index)

    ready = out.notna().all(axis=1)
    quiet = (vol_pct <= cfg.pct_threshold) & (bw_pct <= cfg.pct_threshold)
    out.insert(0, "state", np.where(~ready, WARMUP,
                                    np.where(quiet, COMPRESSED, NOISE)))
    return out
