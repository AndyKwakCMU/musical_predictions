##############################################################################

#           cnn_eval
#           Andy Kwak

##############################################################################

#       Backtest Test B: does the CNN know which way the cadence resolves?
#       Scores a trained model on the Test B test years (TSA-gated events
#       only) against coin flip, always-long, momentum, and logistic
#       regression on the same events.


#       @requires a model from model_train.py and its dataset
#       @ensures printed comparison table + per-year R expectancy

##############################################################################

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from musical import config
from musical.cnn import load_model, predict_proba
from musical.dataset import time_split
from musical.experiment import (
    direction_from_proba, fit_logistic, load_dataset, log_loss,
    momentum_direction, trade_r,
)


def evaluate(model_path, dataset, val_start, test_start, test_end, horizon, seed=0):
    #@ensures (table, per_year) DataFrames for the test split
    X, y, meta = load_dataset(Path(dataset))
    split = time_split(meta, val_start, test_start, test_end).to_numpy()
    tr, te = split == "train", split == "test"
    tickers = meta["ticker"].to_numpy()
    m_te = meta[te].reset_index(drop=True)

    model = load_model(model_path)
    p_cnn = predict_proba(model, X[te], tickers[te])
    logit = fit_logistic(X[tr], y[tr])
    p_lr = logit.predict_proba(X[te].reshape(te.sum(), -1))
    prior = np.bincount(y[tr], minlength=3) / tr.sum()
    p_prior = np.tile(prior, (te.sum(), 1))

    rng = np.random.default_rng(seed)
    directions = {
        "coin_flip": rng.choice([-1, 1], size=te.sum()),
        "always_long": np.ones(te.sum(), dtype=int),
        "momentum": momentum_direction(X[te]),
        "logistic": direction_from_proba(p_lr),
        "cnn": direction_from_proba(p_cnn),
    }
    probas = {"logistic": p_lr, "cnn": p_cnn, "always_long": p_prior}

    rows, per_year = [], {}
    decided = (m_te["label"] != "NONE").to_numpy()
    truth = np.where(m_te["label"] == "UP", 1, -1)
    for name, d in directions.items():
        r = trade_r(m_te, d, horizon)
        rows.append({
            "model": name,
            "events": len(r),
            "dir_acc_on_UP/DOWN": (d[decided] == truth[decided]).mean(),
            "mean_R": r.mean(),
            "log_loss": log_loss(probas[name], y[te]) if name in probas else np.nan,
        })
        per_year[name] = pd.Series(r).groupby(m_te["timestamp"].dt.year).mean()
    return pd.DataFrame(rows).set_index("model"), pd.DataFrame(per_year)


def main():
    par = argparse.ArgumentParser()
    par.add_argument("model")
    par.add_argument("--dataset", default="data/datasets/events_w64_k2_h10.npz")
    par.add_argument("--val-start", default=config.VAL_START)
    par.add_argument("--test-start", default=config.TEST_START)
    par.add_argument("--test-end", default=config.DEV_END)
    par.add_argument("--horizon", type=int, default=config.HORIZON)
    args = par.parse_args()

    table, per_year = evaluate(args.model, args.dataset, args.val_start,
                               args.test_start, args.test_end, args.horizon)
    fmt = lambda x: f"{x:.3f}"
    print(f"Test B  test period [{args.test_start}, {args.test_end})  (pre-cost R per trade)")
    print(table.to_string(float_format=fmt))
    print("\nmean R by year:")
    print(per_year.to_string(float_format=fmt))


if __name__ == "__main__":
    main()
