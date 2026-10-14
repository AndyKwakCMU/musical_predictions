##############################################################################

#           long_filter_eval
#           Andy Kwak

##############################################################################

#       Test D: long-only, filtered by the CNN's P(move).
#       The CNN can tell *whether* a breakout comes but not which way, and
#       gated events drift up. So: go long only on gated events whose
#       P(UP) + P(DOWN) is in the top q of the validation distribution.
#       q is picked on val net expectancy; the val cutoff is then applied
#       unchanged to the test split, which is scored once, through the real
#       backtester (costs, fills, sizing), against always-long and against
#       random subsets of the same size.


#       @requires a Test B-selected model and its event dataset + raw bars
#       @ensures val selection table, test table, permutation p-value

##############################################################################

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from musical import config
from musical.backtest import CostModel, metrics, run_portfolio, simulate_trades
from musical.cnn import load_model, predict_proba
from musical.dataset import time_split
from musical.experiment import (
    load_dataset, load_market, move_proba, perm_pvalue, quantile_cutoff,
)

KEEP_FRACS = [1.0, 0.5, 0.3, 0.2, 0.1]
MIN_VAL_TRADES = 200


def long_signals(meta, keep):
    #@ensures long signals for the rows of meta where keep is True
    return pd.DataFrame({
        "ticker": meta["ticker"].to_numpy()[keep],
        "timestamp": meta["timestamp"].to_numpy()[keep],
        "direction": 1,
    })


def score(signals, bars, atrs, cost):
    #@ensures backtest metrics dict for the given signals
    trades = simulate_trades(bars, atrs, signals, config.BARRIER_K,
                             config.HORIZON, cost)
    log, equity = run_portfolio(trades, cost)
    return metrics(log, equity)


def main():
    par = argparse.ArgumentParser()
    par.add_argument("model")
    par.add_argument("--interval", default="1d", help="1d (yfinance) or 15m (IBKR)")
    par.add_argument("--dataset", default=None)
    par.add_argument("--val-start", default=config.VAL_START)
    par.add_argument("--test-start", default=config.TEST_START)
    par.add_argument("--test-end", default=config.DEV_END)
    par.add_argument("--perms", type=int, default=200)
    args = par.parse_args()
    if pd.Timestamp(args.test_end) > pd.Timestamp(config.DEV_END):
        raise SystemExit("test-end is in the held-out period")

    suffix = "" if args.interval == "1d" else f"_{args.interval}"
    dataset = args.dataset or f"data/datasets/events_w64_k2_h10{suffix}.npz"
    X, y, meta = load_dataset(Path(dataset))
    split = time_split(meta, args.val_start, args.test_start, args.test_end).to_numpy()
    tickers = meta["ticker"].to_numpy()
    model = load_model(args.model)
    bars, atrs = load_market(config.raw_dir(args.interval), config.DEV_END, args.interval)
    cost = CostModel()
    cost2 = CostModel(cost.commission_per_share * 2, cost.commission_min * 2,
                      cost.slippage_bps * 2)

    va, te = split == "val", split == "test"
    m_va = meta[va].reset_index(drop=True)
    m_te = meta[te].reset_index(drop=True)
    pm_va = move_proba(predict_proba(model, X[va], tickers[va]))
    pm_te = move_proba(predict_proba(model, X[te], tickers[te]))

    # --- choose q on val only
    rows = []
    for q in KEEP_FRACS:
        c = quantile_cutoff(pm_va, q)
        m = score(long_signals(m_va, pm_va >= c), bars, atrs, cost)
        rows.append({"keep_frac": q, "cutoff": c, **m})
    val = pd.DataFrame(rows).set_index("keep_frac")
    ok = val[val["trades"] >= MIN_VAL_TRADES]
    best_q = float(ok["expectancy_r"].idxmax())
    cutoff = float(val.loc[best_q, "cutoff"])

    fmt = lambda x: f"{x:.3f}"
    cols = ["cutoff", "trades", "hit_rate", "expectancy_r", "cagr", "sharpe", "max_drawdown"]
    print(f"Test D {args.interval}  val [{args.val_start}, {args.test_start})  net of costs")
    print(val[cols].to_string(float_format=fmt))
    print(f"selected keep_frac={best_q}  cutoff P(move)>={cutoff:.4f}")
    if best_q >= 1.0:
        print("\nno filter won on val: the P(move) filter adds nothing -> FAIL (test not scored)")
        return

    # --- score test once
    keep = pm_te >= cutoff
    results = {
        "always_long": score(long_signals(m_te, np.ones(len(m_te), bool)), bars, atrs, cost),
        "filtered_long": score(long_signals(m_te, keep), bars, atrs, cost),
        "filtered_long_2x_cost": score(long_signals(m_te, keep), bars, atrs, cost2),
    }
    rng = np.random.default_rng(0)
    null = []
    for _ in range(args.perms):
        rand = np.zeros(len(m_te), bool)
        rand[rng.choice(len(m_te), keep.sum(), replace=False)] = True
        null.append(score(long_signals(m_te, rand), bars, atrs, cost)["expectancy_r"])
    null = np.array(null)

    test = pd.DataFrame(results).T
    obs = results["filtered_long"]["expectancy_r"]
    print(f"\ntest [{args.test_start}, {args.test_end})  kept {keep.sum()}/{len(m_te)} events "
          f"({keep.mean():.1%})")
    print(test[cols[1:]].astype(float).to_string(float_format=fmt))
    print(f"\nrandom same-size subsets ({args.perms}): mean {null.mean():.3f}, "
          f"95th pct {np.quantile(null, 0.95):.3f};  p = {perm_pvalue(obs, null):.3f}")

    checks = {
        "1 filtered > 0R": obs > 0,
        "2 filtered > always-long": obs > results["always_long"]["expectancy_r"],
        "3 beats 95% of random subsets": perm_pvalue(obs, null) <= 0.05,
        "4 > 0R at 2x costs": results["filtered_long_2x_cost"]["expectancy_r"] > 0,
    }
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print("Test D:", "PASS" if all(checks.values()) else "FAIL")


if __name__ == "__main__":
    main()
