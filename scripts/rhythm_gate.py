##############################################################################

#           rhythm_gate
#           Andy Kwak

##############################################################################

#       The Rhythm gate (plan 1, pre-registered in docs/experiments.md).
#       Is there a market pulse in the 15m bars that is more than the
#       trading-day clock?
#         1. periodicity: real tempo strength beats a clock-preserving
#            surrogate in >= 70% of ticker-years
#         2. stability: real tempo drifts less session-to-session than
#            the surrogate in >= 70% of ticker-years
#         3. beyond the clock: the full listener predicts upcoming onsets
#            >= 2% better than the clock-only listener overall, and better
#            in >= 70% of tickers (val 2021-2022)
#       Loads nothing at or after config.LISTENER_VAL_END.


#       @requires the two listeners from listener_train.py (full, clock)
#       @ensures printed tables and "Rhythm gate: PASS|FAIL"

##############################################################################

import argparse

import numpy as np
import pandas as pd
import torch

from musical import config
from musical.listener import _Data, evaluate, load_listener, load_series, make_windows, spaced
from musical.rhythm import causal_z, meter, onset_strength, tempo, track_beats, turns
from musical.surrogates import clock_shuffle

SHARE = 0.70
MIN_GAIN = 0.02


def session_log_period_change(period):
    #@ensures median |delta log period| between consecutive sessions, using
    #         each session's period at its last bar (sessions with a period)
    last = period.groupby(period.index.normalize()).last().dropna()
    return np.log(last).diff().abs().dropna()


def rhythm_by_year(feats, rng):
    #@ensures per-year table of real vs surrogate tempo strength and drift,
    #         plus real beats/meter for the informational report
    real_onset = feats["onset"]
    real = tempo(real_onset)
    fake_feats = clock_shuffle(feats, ["ret_atr", "vol_surprise"], rng)
    fake = tempo(onset_strength(fake_feats))

    year = feats.index.year
    rows = []
    for y in np.unique(year):
        sel = year == y
        r_drift = session_log_period_change(real["period"][sel])
        f_drift = session_log_period_change(fake["period"][sel])
        rows.append({
            "year": y,
            "strength_real": real["strength"][sel].median(),
            "strength_fake": fake["strength"][sel].median(),
            "drift_real": r_drift.median() if len(r_drift) else np.nan,
            "drift_fake": f_drift.median() if len(f_drift) else np.nan,
        })
    beats = track_beats(real_onset, real["period"])
    return pd.DataFrame(rows), beats, meter(real_onset, beats)


def ingredient_beat_corr(feats, beats):
    #@ensures correlation of each onset ingredient (z-scored) with beat bars
    parts = {
        "abs_move": causal_z(feats["ret_atr"].abs()),
        "volume_burst": causal_z(feats["vol_surprise"].clip(lower=0)),
        "turn": causal_z(turns(feats["ret_atr"])),
    }
    b = beats.astype(float)
    return {k: v.corr(b) for k, v in parts.items()}


def listener_losses(full, clock, series, device):
    #@ensures per-ticker val onset MSE for both listeners on the SAME windows
    cols = list(dict.fromkeys(full.input_cols + clock.input_cols))
    window = full.cfg.window
    items = spaced(make_windows(series, cols, config.LISTENER_TRAIN_END,
                                config.LISTENER_VAL_END, window), window // 2)
    out = {}
    for name, model in [("full", full), ("clock", clock)]:
        sums = evaluate(model.to(device), _Data(series, model.input_cols), items, window, device)
        out[name] = {t: v[1] / v[2] for t, v in sums.items()}
        out[name + "_pooled"] = sum(v[1] for v in sums.values()) / sum(v[2] for v in sums.values())
    return out


def main():
    par = argparse.ArgumentParser()
    par.add_argument("full_model")
    par.add_argument("clock_model")
    par.add_argument("--seed", type=int, default=0)
    par.add_argument("--raw-dir", default=config.INTRADAY_RAW_DIR)
    args = par.parse_args()

    series = load_series(args.raw_dir, config.LISTENER_VAL_END)
    assert max(df.index.max() for df in series.values()) < pd.Timestamp(config.LISTENER_VAL_END)
    rng = np.random.default_rng(args.seed)

    tables, info = [], []
    for ticker, feats in series.items():
        tab, beats, met = rhythm_by_year(feats, rng)
        tables.append(tab.assign(ticker=ticker))
        gaps = np.diff(np.flatnonzero(beats.to_numpy()))
        on_beats = met["beats_per_measure"][beats]
        info.append({"ticker": ticker,
                     "median_beat_bars": float(np.median(gaps)) if len(gaps) else np.nan,
                     **{f"meter_{m}": float((on_beats == m).mean()) for m in (2, 3, 4)},
                     **ingredient_beat_corr(feats, beats)})
    ty = pd.concat(tables).dropna()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    losses = listener_losses(load_listener(args.full_model, device),
                             load_listener(args.clock_model, device), series, device)
    per_ticker = pd.DataFrame({"full": losses["full"], "clock": losses["clock"]})
    gain = 1 - losses["full_pooled"] / losses["clock_pooled"]

    c1 = (ty["strength_real"] > ty["strength_fake"]).mean()
    c2 = (ty["drift_real"] < ty["drift_fake"]).mean()
    c3_share = (per_ticker["full"] < per_ticker["clock"]).mean()
    checks = {
        f"1 periodicity: real > surrogate in {c1:.0%} of {len(ty)} ticker-years (need {SHARE:.0%})": c1 >= SHARE,
        f"2 stability: real drifts less in {c2:.0%} of ticker-years (need {SHARE:.0%})": c2 >= SHARE,
        f"3 beyond clock: pooled onset MSE full {losses['full_pooled']:.4f} vs clock "
        f"{losses['clock_pooled']:.4f} (gain {gain:.1%}, need {MIN_GAIN:.0%}); full better in "
        f"{c3_share:.0%} of tickers (need {SHARE:.0%})": gain >= MIN_GAIN and c3_share >= SHARE,
    }

    fmt = lambda x: f"{x:.3f}"
    print("Rhythm gate  (15m, 2016-2022; listeners scored on val 2021-2022)\n")
    print("per year, pooled over tickers (medians):")
    print(ty.groupby("year")[["strength_real", "strength_fake", "drift_real", "drift_fake"]]
          .median().to_string(float_format=fmt))
    print("\nper ticker: listener val onset MSE + rhythm info")
    print(per_ticker.join(pd.DataFrame(info).set_index("ticker")).to_string(float_format=fmt))
    print()
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("Rhythm gate:", "PASS" if all(checks.values()) else "FAIL")


if __name__ == "__main__":
    main()
