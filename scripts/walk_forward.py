##############################################################################

#           walk_forward
#           Andy Kwak

##############################################################################

#       Backtest Test C: the full system, walk-forward.
#       For each test year Y: train the CNN on events before Y-1, early-stop
#       on year Y-1, then trade year Y's TSA-gated events through the real
#       backtester (costs, fills, sizing). Baselines trade the same events.
#
#       Only touches years < config.DEV_END unless --final is passed,
#       which should happen exactly once, after Test B passes.


#       @requires data/raw CSVs and the matching event dataset
#       @ensures per-fold and overall metrics; trade log CSV per strategy

##############################################################################

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from musical import config
from musical.backtest import CostModel, metrics, run_portfolio, simulate_trades
from musical.cnn import TrainConfig, predict_proba, train
from musical.data import load_bars
from musical.dataset import time_split
from musical.experiment import direction_from_proba, load_dataset
from musical.tsa import analyze


def load_market(raw_dir, end):
    #@ensures (bars_by_ticker, atr_by_ticker) truncated before `end`
    bars, atrs = {}, {}
    for path in sorted(Path(raw_dir).glob("*_1d.csv")):
        ticker = path.name.split("_")[0]
        b = load_bars(path)
        b = b[b.index < end]
        bars[ticker] = b
        atrs[ticker] = analyze(b)["atr"]
    return bars, atrs


def fold_signals(X, y, meta, year, cfg, seed):
    #@ensures DataFrame of CNN signals for events in `year`
    split = time_split(meta, f"{year - 1}-01-01", f"{year}-01-01",
                       f"{year + 1}-01-01").to_numpy()
    tr, va, te = split == "train", split == "val", split == "test"
    tickers = meta["ticker"].to_numpy()
    model, _ = train(X[tr], y[tr], tickers[tr], X[va], y[va], tickers[va],
                     replace(cfg, seed=seed))
    proba = predict_proba(model, X[te], tickers[te])
    return pd.DataFrame({
        "ticker": tickers[te],
        "timestamp": meta["timestamp"].to_numpy()[te],
        "direction": direction_from_proba(proba),
    })


def backtest(name, signals, bars, atrs, cost, k, horizon):
    #@ensures (metrics dict, sized trade log)
    trades = simulate_trades(bars, atrs, signals, k, horizon, cost)
    log, equity = run_portfolio(trades, cost)
    m = metrics(log, equity)
    m["strategy"] = name
    return m, log


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--dataset", default="data/datasets/events_w64_k2_h10.npz")
    par.add_argument("--first-year", type=int, default=2010)
    par.add_argument("--last-year", type=int, default=2023)
    par.add_argument("--cost-mult", type=float, default=1.0)
    par.add_argument("--final", action="store_true",
                     help="include the held-out period (run once, at the end)")
    par.add_argument("--out", default="data/datasets/walk_forward")
    args = par.parse_args()

    end = "2100-01-01" if args.final else config.DEV_END
    if not args.final and args.last_year >= int(config.DEV_END[:4]):
        raise SystemExit("last-year is in the held-out period; use --final")

    base = CostModel()
    cost = CostModel(base.commission_per_share * args.cost_mult,
                     base.commission_min * args.cost_mult,
                     base.slippage_bps * args.cost_mult)
    cfg = TrainConfig(emb_dim=0, dropout=0.5)   # selected in Test B (val loss)

    X, y, meta = load_dataset(Path(args.dataset))
    bars, atrs = load_market(config.RAW_DIR, end)
    rng = np.random.default_rng(0)

    signals = {"cnn": [], "always_long": [], "coin_flip": []}
    for year in range(args.first_year, args.last_year + 1):
        s = fold_signals(X, y, meta, year, cfg, seed=0)
        signals["cnn"].append(s)
        signals["always_long"].append(s.assign(direction=1))
        signals["coin_flip"].append(s.assign(direction=rng.choice([-1, 1], len(s))))
        print(f"fold {year}: {len(s)} gated events", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, per_year = [], {}
    for name, parts in signals.items():
        m, log = backtest(name, pd.concat(parts), bars, atrs, cost,
                          config.BARRIER_K, config.HORIZON)
        results.append(m)
        log.to_csv(out / f"trades_{name}_cost{args.cost_mult:g}.csv", index=False)
        per_year[name] = log.groupby(pd.to_datetime(log["entry_time"]).dt.year)["r_net"].mean()

    fmt = lambda x: f"{x:.3f}"
    print(f"\nTest C walk-forward {args.first_year}-{args.last_year}, costs x{args.cost_mult:g}")
    print(pd.DataFrame(results).set_index("strategy").to_string(float_format=fmt))
    print("\nmean net R per trade by year:")
    print(pd.DataFrame(per_year).to_string(float_format=fmt))


if __name__ == "__main__":
    main()
