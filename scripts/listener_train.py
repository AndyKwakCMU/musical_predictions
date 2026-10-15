##############################################################################

#           listener_train
#           Andy Kwak

##############################################################################

#       Trains the self-supervised "listener" on every ticker's 15m bars
#       (train < 2021, early-stopped on 2021-2022). Two variants:
#         full  - hears price, volume, the clock, SPY/QQQ, and onsets
#         clock - hears only the clock (time of day, weekday, session
#                 position): the baseline the Rhythm gate must beat, so
#                 we know how much the listener hears beyond the schedule
#       Loads nothing at or after config.LISTENER_VAL_END.


#       @requires data/raw_ib/*_15m.csv from data_fetch.py --source ib
#       @ensures models/listener_{variant}_{time}.pt + trained_params.csv row

##############################################################################

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from musical import config
from musical.features import CLOCK_COLS, PRICE_COLS
from musical.listener import (
    CONTEXT_TICKERS, ListenerConfig, load_series, make_windows, save_listener,
    spaced, train_listener,
)


def input_cols(variant):
    #@ensures the listener's input columns for this variant
    if variant == "clock":
        return list(CLOCK_COLS)
    context = [f"{t}_{f}" for t in CONTEXT_TICKERS for f in ("ret_atr", "vol_surprise")]
    return PRICE_COLS + CLOCK_COLS + context + ["onset"]


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--variant", choices=["full", "clock"], required=True)
    par.add_argument("--epochs", type=int, default=ListenerConfig.epochs)
    par.add_argument("--seed", type=int, default=0)
    par.add_argument("--tag", default="listener")
    par.add_argument("--raw-dir", default=config.INTRADAY_RAW_DIR)
    args = par.parse_args()

    series = load_series(args.raw_dir, config.LISTENER_VAL_END)
    latest = max(df.index.max() for df in series.values())
    assert latest < pd.Timestamp(config.LISTENER_VAL_END), latest
    first = min(df.index.min() for df in series.values())

    cols = input_cols(args.variant)
    cfg = ListenerConfig(epochs=args.epochs, seed=args.seed)
    train_range = (first, config.LISTENER_TRAIN_END)
    val_range = (config.LISTENER_TRAIN_END, config.LISTENER_VAL_END)
    print(f"{args.variant}: {len(series)} tickers, {len(cols)} inputs, device {cfg.device}",
          flush=True)
    model, hist = train_listener(series, cols, cfg, train_range, val_range)

    model_dir = Path(config.MODEL_DIR)
    model_dir.mkdir(exist_ok=True)
    run_id = f"{args.tag}_{args.variant}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    path = model_dir / f"listener_{args.variant}_{run_id}.pt"
    save_listener(model, path)

    n_train = len(make_windows(series, cols, *train_range, cfg.window))
    n_val = len(spaced(make_windows(series, cols, *val_range, cfg.window), cfg.window // 2))
    row = {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": "raw_ib_15m",
        "train_end": config.LISTENER_TRAIN_END,
        "val_start": config.LISTENER_TRAIN_END,
        "val_end": config.LISTENER_VAL_END,
        "n_train": n_train,
        "n_val": n_val,
        "epochs_run": len(hist["val_loss"]),
        "best_val_loss": round(float(np.min(hist["val_loss"])), 5),
        "model_path": str(path),
        "hparams": json.dumps({**asdict(cfg), "variant": args.variant,
                               "input_cols": cols}),
    }
    params_csv = model_dir / "trained_params.csv"
    pd.DataFrame([row], columns=config.PARAMS_COLUMNS).to_csv(
        params_csv, mode="a", header=not params_csv.exists(), index=False)
    print(json.dumps({k: row[k] for k in ["run_id", "epochs_run", "best_val_loss",
                                           "model_path"]}, indent=2))


if __name__ == "__main__":
    main()
