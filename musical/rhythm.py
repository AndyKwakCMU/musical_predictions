##############################################################################

#           rhythm
#           Andy Kwak

##############################################################################

#       Finds the market's own pulse in the 15m bars, the way a beat
#       tracker listens to audio: an onset envelope ("something new
#       starts here"), the tempo it keeps repeating at, then a beat grid
#       and the meter (which beats are strong). The bars are raw samples;
#       the beats are inferred.
#
#       Every value at bar t uses bars <= t (no lookahead).


#       @requires bar_features output (ret_atr, range_atr, vol_surprise)
#       @ensures onset, tempo, beat and meter series indexed like the bars

##############################################################################

import numpy as np
import pandas as pd

Z_WINDOW = 520          # trailing bars for z-scores (~20 sessions of 15m)
SUBHARMONIC = 0.9       # prefer the shortest lag within 90% of the peak


def causal_z(series, window=Z_WINDOW):
    #@ensures \result[t] = (x[t] - mean) / std over the trailing window
    #         ending at t; NaN until the window fills or if std == 0
    roll = series.rolling(window)
    std = roll.std()
    return (series - roll.mean()) / std.where(std > 0)


def turns(ret):
    #@ensures \result[t] = 1.0 where the move changes direction vs the
    #         previous nonzero move, else 0.0
    sign = np.sign(ret)
    prev = sign.replace(0.0, np.nan).ffill().shift(1)
    return ((sign != 0) & prev.notna() & (sign != prev)).astype(float)


def onset_strength(feats):
    #@requires feats has ret_atr, vol_surprise
    #@ensures \result[t] = mean of causal z-scores of |move|, volume
    #         burst, and direction turn; NaN during warm-up
    parts = pd.concat([
        causal_z(feats["ret_atr"].abs()),
        causal_z(feats["vol_surprise"].clip(lower=0)),
        causal_z(turns(feats["ret_atr"])),
    ], axis=1)
    return parts.mean(axis=1, skipna=False).rename("onset")


def tempo(onset, window=Z_WINDOW, min_lag=2, max_lag=24):
    #@ensures \result has period (bars) and strength from the trailing
    #         `window` onsets; NaN until filled.
    #         A beat is a local PEAK in the autocorrelation, scored by its
    #         prominence over the neighbouring lags. Volatility clustering
    #         only makes the autocorrelation decay smoothly (no peak), so it
    #         is not mistaken for a fast pulse. With no peak: period NaN,
    #         strength 0. Among peaks near the best, the shortest lag wins,
    #         so a pulse of 5 reads as 5 rather than 10, 15, 20
    lags = np.arange(min_lag - 1, max_lag + 2)
    acf = np.column_stack([
        onset.rolling(window).corr(onset.shift(lag)).to_numpy() for lag in lags
    ])
    ready = np.isfinite(acf).all(axis=1)
    mid = acf[:, 1:-1]
    left, right = acf[:, :-2], acf[:, 2:]
    is_peak = (mid > left) & (mid > right)
    prominence = np.where(is_peak, mid - (left + right) / 2, -np.inf)
    best = prominence.max(axis=1)
    has_peak = ready & np.isfinite(best)
    near = prominence >= np.where(has_peak, SUBHARMONIC * best, np.inf)[:, None]
    choice = near.argmax(axis=1)
    period = np.where(has_peak, lags[1:-1][choice], np.nan)
    strength = np.where(has_peak, prominence[np.arange(len(acf)), choice],
                        np.where(ready, 0.0, np.nan))
    return pd.DataFrame({"period": period, "strength": strength}, index=onset.index)
