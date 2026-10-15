##############################################################################

#           model_train
#           Andy Kwak

##############################################################################

#       Utilizes different formatted CSV data of a ticker, 
#       trains the heart CNN model of the trading algorithm 


#       @requires proper CSV format of data
#       @ensures proper CSV data of trained parameters

##############################################################################

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from musical import config
from musical.cnn import TrainConfig, save_model, train
from musical.dataset import time_split
from musical.experiment import load_dataset

PARAMS_COLUMNS = config.PARAMS_COLUMNS


def train_split(dataset, val_start, test_start, cfg, tag):
    #@requires dataset path built by build_dataset.py
    #@ensures model trained on events before val_start, early-stopped on
    #         [val_start, test_start); weights + params row saved
    X, y, meta = load_dataset(Path(dataset))
    split = time_split(meta, val_start, test_start).to_numpy()
    tr, va = split == "train", split == "val"
    tickers = meta["ticker"].to_numpy()

    model, hist = train(X[tr], y[tr], tickers[tr], X[va], y[va], tickers[va], cfg)

    model_dir = Path(config.MODEL_DIR)
    model_dir.mkdir(exist_ok=True)
    run_id = f"{tag}_{datetime.now():%Y%m%d_%H%M%S}"
    model_path = model_dir / f"cadence_{run_id}.pt"
    save_model(model, model_path)

    row = {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "train_end": val_start,
        "val_start": val_start,
        "val_end": test_start,
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "epochs_run": len(hist["val_loss"]),
        "best_val_loss": round(float(np.min(hist["val_loss"])), 5),
        "model_path": str(model_path),
        "hparams": json.dumps(model.train_config),
    }
    params_csv = model_dir / "trained_params.csv"
    pd.DataFrame([row], columns=PARAMS_COLUMNS).to_csv(
        params_csv, mode="a", header=not params_csv.exists(), index=False)
    return model, row


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--dataset", default="data/datasets/events_w64_k2_h10.npz")
    par.add_argument("--val-start", default=config.VAL_START)
    par.add_argument("--test-start", default=config.TEST_START)
    par.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    par.add_argument("--tag", default="testB")
    args = par.parse_args()

    cfg = TrainConfig(epochs=args.epochs)
    _, row = train_split(args.dataset, args.val_start, args.test_start, cfg, args.tag)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
