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


def track_beats(onset, period, tol=0.34, q=0.6):
    #@requires period from tempo() (NaN where no pulse was heard)
    #@ensures \result[t] True if bar t is a beat, decided at t from bars
    #         <= t. A beat lands on a strong onset near the expected
    #         spacing; if none comes, one is forced to keep the pulse.
    #         "Strong" = in the top 1/period of recent onsets (at least the
    #         q-quantile), since a real beat occupies about 1/period of bars.
    #         Through no-pulse stretches the last heard tempo is kept.
    rank = onset.rolling(Z_WINDOW).rank(pct=True).to_numpy()
    per = period.ffill().to_numpy()
    beats = np.zeros(len(onset), dtype=bool)
    last = None
    for t in range(len(onset)):
        p = per[t]
        if not np.isfinite(p):
            continue
        if last is None:
            if np.isfinite(rank[t]) and rank[t] >= max(q, 1 - 1 / p):
                beats[t], last = True, t
            continue
        d = t - last
        strong = np.isfinite(rank[t]) and rank[t] >= max(q, 1 - 1 / p)
        if (d >= round(p * (1 - tol)) and strong) or d >= round(p * (1 + tol)):
            beats[t], last = True, t
    return pd.Series(beats, index=onset.index, name="beat")


def meter(onset, beats, n_beats=32, choices=(2, 3, 4)):
    #@ensures \result has beats_per_measure (carried forward between
    #         beats; NaN until n_beats beats are heard) and downbeat (True on
    #         beats in the strongest accent phase). Uses beats <= t only.
    pos = np.flatnonzero(beats.to_numpy())
    accent = onset.to_numpy()[pos]
    per_beat = np.full(len(pos), np.nan)
    down = np.zeros(len(pos), dtype=bool)
    for i in range(n_beats - 1, len(pos)):
        a = accent[i - n_beats + 1:i + 1]
        if not np.isfinite(a).all():
            continue
        scores = [np.corrcoef(a[m:], a[:-m])[0, 1] for m in choices]
        if not np.isfinite(scores).any():
            continue
        m = choices[int(np.nanargmax(scores))]
        idx = np.arange(i - n_beats + 1, i + 1)
        phase_means = [a[(idx % m) == ph].mean() for ph in range(m)]
        per_beat[i] = m
        down[i] = (i % m) == int(np.argmax(phase_means))
    bpm = np.full(len(onset), np.nan)
    bpm[pos] = per_beat
    downbeat = np.zeros(len(onset), dtype=bool)
    downbeat[pos] = down
    return pd.DataFrame({
        "beats_per_measure": pd.Series(bpm, index=onset.index).ffill(),
        "downbeat": downbeat,
    }, index=onset.index)
