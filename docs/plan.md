# musical_predictions — Revival Plan

**Status (2026-10-13):** all code phases are built. Test A passes, and Tests B and C fail. IBKR is built up to the dry run (step 3) but not yet run against a real Gateway. See `docs/experiments.md` and `CLAUDE.md`.

Written 2026-10-13. The project went dormant in 2026-01 with only `scripts/data_fetch.py` working. See `CLAUDE.md` for the current state and known bugs.

The plan has three parts:

1. **Revival.** Get the repo healthy and build the two-stage model (TSA gate → CNN direction).
2. **Backtesting.** Prove each stage earns its place before any money is involved.
3. **IBKR connection.** Pull better historical data, then paper trade.

Each phase ends with an **exit check**. Don't start the next phase until the current one passes. That keeps this from turning into a pile of half-built parts again.

---

## Part 1 — Revival

### Phase 0: Repo hygiene (an evening)

- Add `requirements.txt` with `pandas numpy yfinance torch scikit-learn matplotlib ib_async pytest`, plus `arch` and `hmmlearn` if TSA uses them.
- Add a `.gitignore` covering `.venv/`, `__pycache__/`, `*.pt`, large data files, and `data/raw/` once data gets big.
- Lay out the directories like this:
  ```
  scripts/      CLI entry points (data_fetch, train, backtest, paper_trade)
  musical/      importable code: data.py, tsa.py, cnn.py, backtest.py, broker.py
  data/         CSVs (small samples committed, bulk data gitignored)
  models/       trained weights / parameter CSVs (gitignored)
  tests/
  ```
- Keep the existing banner and `@requires`/`@ensures` style in every new file.

**Exit check:** `pip install -r requirements.txt` works in a fresh venv, and `pytest` runs, even with zero tests.

### Phase 1: Fix the data layer

Fix the bugs listed in `CLAUDE.md`:

- Flatten yfinance's MultiIndex columns, which removes the junk `,AAPL,AAPL` row.
- Handle the `Datetime` column that intraday intervals produce.
- Emit the README protocol exactly, as `date,time,open,high,low,close,volume`. Use exchange-local time, and fill `time` with `00:00:00` for daily bars.
- Fix the date format in the argparse help to `YYYY-MM-DD`, and make `out` optional with `nargs="?"`.
- Add `musical/data.py: load_bars(path) -> DataFrame` so the rest of the code reads the protocol in one place.
- Write tests for round-tripping a CSV and for daily and intraday column handling. Mock yfinance in these.

**Data reality check:** yfinance only keeps about 60 days of intraday history, and 1m bars only about 7 days. That's not enough to train a pattern library on intraday bars. Two options:

- (a) Start on **daily bars across many tickers**, which yfinance has for decades.
- (b) Pull intraday history from IBKR (Part 3, step 1).

**Recommendation:** do (a) now, then switch to (b) once the IBKR connection works.

**Exit check:** fetching 10 years of daily bars for about 20 liquid tickers gives clean protocol CSVs, and `load_bars` reads every one of them.

### Phase 2: TSA, the "are we in the V function?" gate

Goal: output a per-bar state, either `COMPRESSED` (setup forming, tradeable soon) or `NOISE`, plus a few continuous metrics that the CNN can use.

Start with transparent measures and add sophistication only if they fall short:

1. **Baseline features.** Each is computed with only past data:
   - Realized volatility over N bars, as a percentile of its own trailing 1-year distribution.
   - Bollinger bandwidth percentile.
   - ATR(short) / ATR(long) ratio.
   - Range contraction, meaning the high-low range shrinking over k bars.
2. **State model.** Pick one of these, in order of complexity:
   - Rule-based: compressed when vol percentile < 20 *and* ATR ratio < 1.
   - A 2–3 state Gaussian **HMM** (`hmmlearn`) on log-returns and log-vol. This matches blog post 1's "hidden states" idea.
   - **GARCH** (`arch`) conditional vol, compared against realized vol.
3. Output a DataFrame with one row per bar: `state`, `vol_pct`, `bw_pct`, `atr_ratio`, `p_compressed` (if HMM).

**Lookahead rule:** every TSA value at bar *t* may only use data up to and including bar *t*. Write a test for this: truncating the future must not change past outputs.

**Exit check:** this is Part 2 Test A, below. Compressed states must be followed by larger absolute moves than unconditional bars.

### Phase 3: Build the CNN "cadence" dataset

- **Events.** Each bar where TSA flips into `COMPRESSED` (or the bar where compression breaks) is one sample.
- **Input.** The previous W bars (start with W = 64) of OHLCV, normalized within the window: log-returns, range / ATR, and volume / its mean. Add the TSA metrics as extra channels. The result is a `(channels, W)` tensor.
- **Label.** Use **triple-barrier** labeling. From the event bar, set an upper barrier at +k·ATR, a lower barrier at −k·ATR, and a time limit of H bars. The label is `UP`, `DOWN`, or `NONE` depending on which barrier is hit first. This defines "breakout direction" in a way that is honest about risk.
- Store the samples in a dataset file with ticker, timestamp, tensor, and label.

**Exit check:** you have a few thousand events, the class balance is reported, and spot-checking a dozen events on a chart looks right.

### Phase 4: Train the CNN

- **Model.** Start with a small 1D CNN: 3–4 `Conv1d` blocks, then global pooling, then a linear layer with 3 outputs. 1D is the natural first choice for multichannel time series. Revisit 2D options, such as Gramian Angular Field images, only if 1D falls short.
- **Per-ticker behavior.** The README stresses that each ticker has its own "composer." Start with one shared model plus a ticker embedding or per-ticker fine-tuning, rather than one model per ticker, which would be data-starved.
- **Splits.** Split strictly by time: train on early years, validate on middle years, test on the latest years. Never shuffle across time. Add an **embargo** of H bars between splits so labels don't leak across the boundary.
- `scripts/model_train.py` writes weights to `models/` and a summary row to the "trained parameters" CSV. Define that protocol in the README once its columns settle.

**Exit check:** this is Part 2 Test B, below.

---

## Part 2 — Backtesting Plan

The question is whether each stage adds value on data it has never seen. Test the stages separately first, then together.

### Engine

Write a small **event-driven** backtester in `musical/backtest.py`. Off-the-shelf libraries like `backtesting.py` and `vectorbt` don't fit the gate-then-classify flow cleanly, and a ~200-line engine you understand is worth more here.

- **Fills.** Signal at bar *t* close, fill at bar *t+1* open. Never fill at the same bar's close.
- **Costs.** IBKR-like commission (about $0.0035/share, $0.35 minimum) plus slippage of 1–2 bps, or 1 tick, per side. Rerun everything at 2× costs. If the edge disappears, it wasn't real.
- **Position rules.** Fixed fractional sizing at 1% risk per trade. The stop and target are the triple-barrier levels and the time stop is H bars, so the backtest trades exactly what the model was labeled on.
- **Outputs.** A trade log CSV, an equity curve, and these metrics: trade count, hit rate, average win/loss, expectancy per trade (in R), Sharpe, max drawdown, exposure %, and turnover.

### Test A: Does TSA find V chords?

There is no CNN in this test.

- Compare the distribution of |forward return over H bars| after `COMPRESSED` bars against all bars, and against random bars matched for volatility.
- **Pass:** compressed bars lead to meaningfully larger absolute moves, measured as a higher median |move| / ATR. The result must hold across most tickers and years, not just on average.
- If it fails, iterate on Phase 2 *before* touching the CNN. Without a working gate, the CNN is guessing in noise.

### Test B: Does the CNN know which way the cadence resolves?

Only events that TSA has gated are used in this test.

- Measure out-of-sample accuracy, log-loss, and a confusion matrix on the held-out test years.
- Compare against these baselines on the **same events**:
  1. Random direction, i.e. coin flip.
  2. Always-long. Stocks drift up, so this baseline is tougher than it sounds.
  3. Simple momentum: the sign of the return over the last W bars.
  4. A logistic regression on the same features. If the CNN can't beat this, the convolutions aren't finding cadences.
- **Pass:** the CNN beats all four baselines on the test years, and stays ahead in most individual years.

### Test C: Full system, walk-forward

- **Walk-forward.** Train on years [1..n] and test on year n+1, then roll forward. Report every fold, not just the average.
- Compare the CNN running on TSA-gated events against the CNN running on all bars. This measures the cost and value of the gate, which is the README's central claim.
- Robustness:
  - Vary W, k, and H by ±25%, and check that results don't collapse.
  - Run on tickers that were never used during development.
  - Run the 2× cost scenario.
- **Pass:** positive expectancy after costs in most folds, with no single parameter setting doing all the work.

### Overfitting guardrails

- Pick the final test period now, for example 2024-01 onward. Don't look at it until Test C is final.
- Keep a `docs/experiments.md` log listing every configuration tried. The number of attempts matters when judging a good result.

---

## Part 3 — IBKR Connection

**Library:** use [`ib_async`](https://github.com/ib-api-reloaded/ib_async), the maintained community successor to `ib_insync`. It wraps IBKR's TWS API with a simple synchronous or async interface.

### Setup (one time)

1. Enable a **paper trading account** in IBKR Client Portal.
2. Install **IB Gateway**, which is lighter than TWS and works well headless on Linux. Log in with the *paper* credentials.
3. In the Gateway API settings:
   - Enable the socket API.
   - Use port **4002**. The ports are 4002 for paper and 4001 for live on Gateway, and 7497 / 7496 on TWS.
   - Add `127.0.0.1` to trusted IPs.
   - Leave "Read-Only API" **on** until order code exists.
4. Subscribe to market data. US equity snapshot/streaming bundles are a few dollars a month. Historical bars generally work without a subscription for many US stocks, but live quotes need one.
5. Gateway logs out daily. For unattended running later, look at IBC (IB Controller) for auto-login and restarts.

### Code: `musical/broker.py`

```python
from ib_async import IB, Stock, util

def connect(host="127.0.0.1", port=4002, client_id=1):
    #@requires IB Gateway running with API enabled on port
    #@ensures \result is a connected IB instance
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib

def fetch_bars(ib, ticker, end="", duration="1 Y", bar_size="5 mins"):
    #@ensures \result is a DataFrame in the project CSV protocol
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)
    bars = ib.reqHistoricalData(
        contract, endDateTime=end, durationStr=duration,
        barSizeSetting=bar_size, whatToShow="TRADES",
        useRTH=True, formatDate=1,
    )
    df = util.df(bars)
    # convert df to date,time,open,high,low,close,volume here
    return df
```

### Rollout, one step at a time

1. **Connectivity.** Write `scripts/ib_check.py`, which connects, prints account summary, and disconnects. Exit check: it runs against paper.
2. **Historical data.**
   - Add an `--source ib` option to `data_fetch.py` that pages backward with `endDateTime` to pull years of 5-min or 1-min bars, and writes protocol CSVs.
   - Respect IBKR pacing limits. Stay at or under about 60 requests per 10 minutes, avoid identical requests within 15 s, and sleep between pages.
   - This removes the yfinance intraday limit and unlocks intraday versions of Phases 3–4.
3. **Live bars, no orders.**
   - Write `scripts/paper_trade.py --dry-run`. It subscribes to `reqRealTimeBars` or `keepUpToDate=True` historical bars, runs TSA → CNN on each new bar, and logs the signals it *would* trade.
   - Run it for 2+ weeks and compare its signals to what the backtester produces on the same days. They must match. A mismatch means there is lookahead or a data difference somewhere.
4. **Paper orders.**
   - Turn Read-Only API off. Place **bracket orders** with `ib.bracketOrder(...)`: entry, stop at the lower barrier, and target at the upper barrier. This mirrors the backtest exactly.
   - Add a time-stop exit after H bars.
   - Log every fill to CSV.
5. **Paper evaluation.**
   - Run for at least 2–3 months.
   - Compare paper fills and slippage against the backtest's cost assumptions.
   - Going live is a separate decision, made only after paper results line up with the backtest.

### Safety rails (in code, from day one)

- Default to port 4002 (paper). Any live port requires an explicit `--live` flag *and* a confirmation prompt.
- Enforce a maximum position size, a maximum number of open positions, and a daily loss limit that flattens all positions and stops the script.
- Add a kill switch: a file such as `STOP` in the repo root halts new orders.

---

## Suggested order and rough timing

| Step | What | Rough effort |
|---|---|---|
| 0 | Repo hygiene | 1 evening |
| 1 | Data fixes + tests | 1–2 evenings |
| 3.1 | IBKR connectivity check | 1 evening (can run parallel to 2) |
| 2 + Test A | TSA gate | 1–2 weeks |
| 3.2 | IBKR historical data | 2–3 evenings |
| 3–4 + Test B | CNN dataset + training | 2–3 weeks |
| Test C | Walk-forward full system | 1 week |
| 3.3–3.5 | Dry run → paper orders → evaluation | 2 weeks + 2–3 months running |

The right first commit is Phase 0 + Phase 1, which is small, concrete, and unblocks everything else.
