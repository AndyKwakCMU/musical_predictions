# CLAUDE.md

## What this project is

An experiment in trading-signal prediction inspired by music theory (see the two blog posts linked in `README.md`). The core analogy: a listener recognizes a cadence before it resolves; a model should recognize a "breakout setup" before the price resolves.

Planned two-stage pipeline:

1. **Time-Series Analysis (TSA) — the regime gate.** Detects volatility compression and outputs a state ("tradeable" vs "chop/noise"), plus extra metrics for the next stage. Musical framing: knowing we are *in the V function* without yet knowing how it cadences. It only does math, so it stays cheap and runs first.
2. **CNN — the cadence recognizer.** Runs only when TSA says the market is tradeable. Trained on historical data to learn a library of per-ticker breakout "cadences" and predict breakout *direction*.

Ideas raised in the blog posts but not decided: HMM / Bayesian regime detection, SQL storage for historical data, paper trading through Interactive Brokers' IB Gateway.

## Current state (as of 2026-10-13)

Phases 0–1 of `docs/plan.md` are done. Phase 2 (the TSA gate) is next.

- `musical/data.py` is the single home of the raw CSV protocol. It has `to_protocol` (yfinance → protocol), `save_bars`, and `load_bars` (validates the header and returns a timestamp-indexed DataFrame). Intraday times are stored in exchange-local wall-clock time, and daily bars get `00:00:00`.
- `scripts/data_fetch.py` is a thin CLI over yfinance + `musical.data`, and handles both daily and intraday intervals.
- `scripts/model_train.py` is still just a header.
- `data/raw/` (gitignored) holds 10 years of daily bars (2016–2025) for 20 liquid tickers.

**yfinance limits intraday history.** 1m data only goes back about 7 days, and other sub-daily intervals about 60 days. Use daily bars until IBKR historical data exists (plan Part 3, step 2).

## Commands

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # also installs `musical` editable
.venv/bin/pytest                                                       # run tests
.venv/bin/python scripts/data_fetch.py AAPL 2016-01-01 2026-01-01 1d data/raw
```

Dates are `YYYY-MM-DD`, the end date is exclusive, and `out` defaults to `data/`.

## Layout

- `musical/` is importable library code (data, and later tsa / cnn / backtest / broker).
- `scripts/` holds the CLI entry points.
- `tests/` holds pytest tests. Mock network calls (yfinance, IBKR) in them.
- `data/` holds small committed sample CSVs. Bulk data goes in `data/raw/`, which is gitignored.
- `models/` holds trained weights, and is gitignored.

## Conventions

- Each script starts with a `#####` banner block giving the name, author (Andy Kwak), a purpose paragraph, and `@requires` / `@ensures` contracts.
- Functions document their contracts as `#@requires` / `#@ensures` comments in CMU 15-122 style. Keep this style in new code.
- Scripts are CLI entry points that use `argparse` and `if __name__ == "__main__": main()`.
- Data lives in `data/` as CSV, named `{ticker}_{start}_{end}_{interval}.csv`.
- The "trained parameters" CSV protocol is still TBD in the README. Define it there when the model exists.

## Roadmap

See `docs/plan.md` for the revival plan (phases with exit checks), backtesting plan, and IBKR (`ib_async`, IB Gateway paper port 4002) integration steps.
