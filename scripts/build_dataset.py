##############################################################################

#           build_dataset
#           Andy Kwak

##############################################################################

#       Builds the CNN event dataset (one sample per COMPRESSED bar) for
#       every ticker in data/raw, across ALL years. Splits are applied
#       later by time, so the held-out period is just rows we don't touch.


#       @requires protocol daily CSVs in data/raw
#       @ensures data/datasets/events_w{W}_k{k}_h{H}.npz (+ _meta.csv)

##############################################################################

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from musical import config
from musical.data import load_bars
from musical.dataset import build_events


def dataset_path(width, k, horizon, directory=config.DATASET_DIR):
    #@ensures \result is the .npz path for these settings
    return Path(directory) / f"events_w{width}_k{k:g}_h{horizon}.npz"


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--raw-dir", default=config.RAW_DIR)
    par.add_argument("--width", type=int, default=config.WINDOW)
    par.add_argument("--k", type=float, default=config.BARRIER_K)
    par.add_argument("--horizon", type=int, default=config.HORIZON)
    args = par.parse_args()

    Xs, ys, metas = [], [], []
    for path in sorted(Path(args.raw_dir).glob("*_1d.csv")):
        ticker = path.name.split("_")[0]
        X, y, meta = build_events(load_bars(path), ticker,
                                  args.width, args.k, args.horizon)
        if len(X):
            Xs.append(X); ys.append(y); metas.append(meta)

    X, y = np.concatenate(Xs), np.concatenate(ys)
    meta = pd.concat(metas, ignore_index=True)

    out = dataset_path(args.width, args.k, args.horizon)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y)
    meta.to_csv(out.with_name(out.stem + "_meta.csv"), index=False)

    print(f"wrote {out}: X{X.shape}")
    print("label balance:", meta["label"].value_counts(normalize=True).round(3).to_dict())
    years = pd.to_datetime(meta["timestamp"]).dt.year
    print("events per year:", years.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
