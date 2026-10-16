##############################################################################

#           experiment
#           Andy Kwak

##############################################################################

#       Shared helpers for Test B / Test C: loading the event dataset,
#       scoring directions in R units, and the baselines the CNN must beat.


#       @requires dataset built by scripts/build_dataset.py
#       @ensures consistent scoring across every experiment

##############################################################################

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from musical import config
from musical.data import load_bars
from musical.dataset import CHANNELS, LABEL_TO_INT
from musical.labels import NONE
from musical.tsa import analyze


def load_dataset(path):
    #@requires path is an events_*.npz with a matching _meta.csv
    #@ensures (X, y, meta) aligned row for row
    blob = np.load(path)
    meta = pd.read_csv(path.with_name(path.stem + "_meta.csv"),
                       parse_dates=["timestamp", "horizon_end"])
    return blob["X"], blob["y"], meta


def trade_r(meta, direction, horizon):
    #@requires direction[i] in {+1, -1}
    #@ensures \result[i] is the R multiple of taking event i in that
    #         direction (pre-cost); same-bar stop/target ambiguity = -1R
    risk = (meta["upper"] - meta["entry"]).to_numpy(float)
    move = (meta["exit_price"] - meta["entry"]).to_numpy(float)
    r = direction * move / risk
    ambiguous = ((meta["label"] == NONE) & (meta["bars_held"] < horizon)).to_numpy()
    r[ambiguous] = -1.0
    return r


def direction_from_proba(proba):
    #@ensures +1 where P(UP) >= P(DOWN), else -1
    up, down = LABEL_TO_INT["UP"], LABEL_TO_INT["DOWN"]
    return np.where(proba[:, up] >= proba[:, down], 1, -1)


def momentum_direction(X):
    #@ensures sign of close(t) - close(t-W+1), from the pos_atr channel
    pos = X[:, CHANNELS.index("pos_atr"), 0]   # close(t-W+1) - close(t)
    return np.where(pos <= 0, 1, -1)


def fit_logistic(X, y):
    #@ensures sklearn pipeline on flattened windows (predict_proba ready)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.05, max_iter=2000))
    return model.fit(X.reshape(len(X), -1), y)


def log_loss(proba, y):
    #@ensures mean negative log-likelihood of the true class
    p = np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


def move_proba(proba):
    #@ensures \result[i] = P(UP) + P(DOWN): "a breakout happens either way"
    return proba[:, LABEL_TO_INT["UP"]] + proba[:, LABEL_TO_INT["DOWN"]]


def quantile_cutoff(values, keep_frac):
    #@requires 0 < keep_frac <= 1
    #@ensures values >= \result keeps about the top keep_frac of values;
    #         keep_frac == 1 keeps everything
    if keep_frac >= 1.0:
        return -np.inf
    return float(np.quantile(values, 1.0 - keep_frac))


def perm_pvalue(observed, null):
    #@ensures fraction of null draws scoring >= observed
    return float((np.asarray(null) >= observed).mean())


def load_market(raw_dir, end, interval="1d"):
    #@ensures (bars_by_ticker, atr_by_ticker) truncated before `end`
    bars, atrs = {}, {}
    cfg = config.tsa_config(interval)
    for path in sorted(Path(raw_dir).glob(f"*_{interval}.csv")):
        ticker = path.name.split("_")[0]
        b = load_bars(path)
        b = b[b.index < end]
        bars[ticker] = b
        atrs[ticker] = analyze(b, cfg)["atr"]
    return bars, atrs
