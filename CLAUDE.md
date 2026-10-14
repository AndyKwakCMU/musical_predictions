# CLAUDE.md

## What this project is

An experiment in trading-signal prediction inspired by music theory (see the two blog posts linked in `README.md`). The core analogy: a listener recognizes a cadence before it resolves; a model should recognize a "breakout setup" before the price resolves.

Planned two-stage pipeline:

1. **Time-Series Analysis (TSA) — the regime gate.** Detects volatility compression and outputs a state ("tradeable" vs "chop/noise"), plus extra metrics for the next stage. Musical framing: knowing we are *in the V function* without yet knowing how it cadences. It only does math, so it stays cheap and runs first.
2. **CNN — the cadence recognizer.** Runs only when TSA says the market is tradeable. Trained on historical data to learn a library of per-ticker breakout "cadences" and predict breakout *direction*.

Ideas raised in the blog posts but not decided: HMM / Bayesian regime detection, SQL storage for historical data, paper trading through Interactive Brokers' IB Gateway.

## Current state (as of 2026-10-13)

All of the code in `docs/plan.md` is built. Results so far are in `docs/experiments.md`:

- **Test A (TSA gate): PASS.** COMPRESSED bars are followed by larger moves than noise bars and than a low-vol control.
- **Test B (CNN direction): FAIL.** The CNN has no directional edge over coin flip, always-long, momentum, or logistic regression on 2021–2023.
- **Test D (long-only, filtered by CNN P(move), with costs): FAIL** on daily and 15m. See `scripts/long_filter_eval.py`.
- **Test C (walk-forward 2010–2023, with costs): FAIL.** The CNN is about 88% long and underperforms always-long on the same gated events.
- **IBKR:** verified live against IB Gateway paper (port 4002). Historical data works. Account endpoints (positions, account summary) time out on this Gateway, so `ib_check.py` treats them as optional. `TRADES` intraday bars come back already split-adjusted. A dry-run signal runner exists. Order placement is intentionally **not written**.
- **Intraday round (15m, 30 names, IBKR):** the gate is time-of-day-aware (`TSAConfig(seasonal_days=40)`) and Test A passes. Test B status is in `docs/experiments.md`. Raw data lives in `data/raw_ib/`, and the dataset is `data/datasets/events_w64_k2_h10_15m.npz`.

**The held-out period 2024-01-01+ (`config.DEV_END`) is untouched.** Only `walk_forward.py --final` may use it, once, after a model passes Test B.

Pre-declared intraday settings and splits (train 2016–2020 / val 2021–2022 / test 2023) are in `docs/experiments.md`. `config.INTRADAY_*` holds them.

**yfinance limits intraday history.** 1m data only goes back about 7 days, and other sub-daily intervals about 60 days. Use `data_fetch.py --source ib` for real intraday history.

## Commands

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # also installs `musical` editable
.venv/bin/pytest                                                       # 64 tests, no network

# data
.venv/bin/python scripts/data_fetch.py AAPL 2000-01-01 2026-01-01 1d data/raw          # yfinance
.venv/bin/python scripts/data_fetch.py AAPL 2026-01-01 2026-10-01 5m data/raw --source ib  # IBKR (Gateway on 4002)
# (universe: data/universe.txt; refetch all with a loop over it)

# experiments (dev period only); add --interval 15m for the intraday round (reads data/raw_ib/)
.venv/bin/python scripts/tsa_eval.py                 # Test A
.venv/bin/python scripts/build_dataset.py            # -> data/datasets/events_w64_k2_h10.npz
.venv/bin/python scripts/model_train.py              # -> models/*.pt + models/trained_params.csv
.venv/bin/python scripts/cnn_eval.py models/<run>.pt # Test B
.venv/bin/python scripts/walk_forward.py             # Test C (add --cost-mult 2 for stress)
.venv/bin/python scripts/long_filter_eval.py models/<run>.pt  # Test D (long-only, P(move) filter)

# IBKR
.venv/bin/python scripts/ib_check.py                       # connectivity check
.venv/bin/python scripts/paper_trade.py models/<run>.pt    # DRY RUN: logs would-trade signals to data/signals.csv
```

Dates are `YYYY-MM-DD`, the end date is exclusive, and `out` defaults to `data/`. A `STOP` file in the repo root halts `paper_trade.py`.

## Layout

- `musical/` holds the importable library:
  - `data`: the CSV protocol
  - `tsa`: the compression gate
  - `labels`: triple-barrier labeling
  - `dataset`: CNN windows and time splits with embargo
  - `cnn`: the model
  - `backtest`: the engine
  - `experiment`: scoring and baselines
  - `broker`: ib_async wrapper
  - `live`: per-bar signal
  - `config`: shared settings and the holdout boundary
- `scripts/` holds the CLI entry points listed above.
- `tests/` holds pytest tests. Network calls (yfinance, IBKR) are always faked.
- `data/` holds the committed sample and `universe.txt`. `data/raw/`, `data/datasets/`, and `data/signals.csv` are gitignored.
- `models/` holds weights and `trained_params.csv`, and is gitignored.

## Invariants (don't break these)

- **No lookahead.** TSA values at bar t use bars ≤ t, and there are tests that truncate the future to check this.
- **Labels and backtest trade the same thing.** Both enter at the next bar's open, use ±k·ATR(t) barriers, and time-stop at t+H.
- **Pessimistic fills.** A same-bar stop and target counts as the stop. A gap through a level fills at the open.
- **Model selection uses validation data only.** Test splits are scored once per selected model, and every attempt is logged in `docs/experiments.md`.

## Conventions

- Each script starts with a `#####` banner block giving the name, author (Andy Kwak), a purpose paragraph, and `@requires` / `@ensures` contracts.
- Functions document their contracts as `#@requires` / `#@ensures` comments in CMU 15-122 style. Keep this style in new code.
- Scripts are CLI entry points that use `argparse` and `if __name__ == "__main__": main()`.
- Data lives in `data/` as CSV, named `{ticker}_{start}_{end}_{interval}.csv`.
- The trained-parameters CSV protocol (`models/trained_params.csv`) is defined in the README.

## Roadmap

See `docs/plan.md` for the revival plan (phases with exit checks), backtesting plan, and IBKR (`ib_async`, IB Gateway paper port 4002) integration steps.
