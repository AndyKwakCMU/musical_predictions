##############################################################################

#           tsa_eval
#           Andy Kwak

##############################################################################

#       Backtest Test A: does the TSA gate find "V chords"?
#       Compares what happens after COMPRESSED bars vs NOISE bars on the
#       development period only (before config.DEV_END).


#       @requires protocol daily CSVs in data/raw
#       @ensures printed comparison table + per ticker-year consistency

##############################################################################

import argparse
from pathlib import Path

import pandas as pd

from musical import config
from musical.data import load_bars
from musical.labels import NONE, triple_barrier
from musical.tsa import COMPRESSED, NOISE, analyze


def collect(raw_dir, k, horizon, interval="1d"):
    #@ensures \result has one labeled row per ready bar, tagged with
    #         ticker, state, and whether it starts a compressed episode
    frames = []
    cfg = config.tsa_config(interval)
    for path in sorted(Path(raw_dir).glob(f"*_{interval}.csv")):
        ticker = path.name.split("_")[0]
        bars = load_bars(path)
        bars = bars[bars.index < config.DEV_END]
        tsa = analyze(bars, cfg)
        ready = tsa.index[tsa["state"] != "WARMUP"]
        lab = triple_barrier(bars, tsa["atr"], k, horizon, events=ready)
        lab["ticker"] = ticker
        lab["state"] = tsa.loc[lab.index, "state"]
        lab["vol_pct"] = tsa.loc[lab.index, "vol_pct"]
        prev = tsa["state"].shift(1).loc[lab.index]
        lab["episode_start"] = (lab["state"] == COMPRESSED) & (prev != COMPRESSED)
        lab["hit"] = (lab["label"] != NONE) | (lab["bars_held"] < horizon)
        frames.append(lab)
    return pd.concat(frames)


def summarize(df):
    #@ensures one row of summary stats for the group df
    return pd.Series({
        "bars": len(df),
        "hit_rate": df["hit"].mean(),
        "median_excursion_atr": df["max_excursion_atr"].median(),
        "median_abs_ret_pct": 100 * df["fwd_ret"].abs().median(),
    })


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--interval", default="1d", help="1d (yfinance) or 15m (IBKR)")
    par.add_argument("--raw-dir", default=None)
    par.add_argument("--k", type=float, default=config.BARRIER_K)
    par.add_argument("--horizon", type=int, default=config.HORIZON)
    args = par.parse_args()

    raw = args.raw_dir or config.raw_dir(args.interval)
    df = collect(raw, args.k, args.horizon, args.interval)
    groups = {
        "ALL": df,
        "NOISE": df[df["state"] == NOISE],
        "LOW_VOL_CONTROL": df[(df["state"] == NOISE) & (df["vol_pct"] <= 0.2)],
        "COMPRESSED": df[df["state"] == COMPRESSED],
        "EPISODE_START": df[df["episode_start"]],
    }
    table = pd.DataFrame({name: summarize(g) for name, g in groups.items()}).T
    print(f"Test A  {args.interval}  (k={args.k}, horizon={args.horizon}, dev period < {config.DEV_END})")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))

    df["year"] = df.index.year
    per = df.groupby(["ticker", "year", "state"])["max_excursion_atr"].median().unstack()
    per = per.dropna(subset=[COMPRESSED, NOISE])
    wins = (per[COMPRESSED] > per[NOISE]).mean()
    print(f"\nticker-years where COMPRESSED median excursion > NOISE: "
          f"{wins:.0%} of {len(per)}")
    by_ticker = per.groupby(level="ticker").apply(lambda p: (p[COMPRESSED] > p[NOISE]).mean())
    print("per-ticker win rate:", " ".join(f"{t}={v:.0%}" for t, v in by_ticker.items()))


if __name__ == "__main__":
    main()
