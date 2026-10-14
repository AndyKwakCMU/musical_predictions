# Live Listening and IBKR Paper Trading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** provisional. Tasks 1–3 (live state + dry run) may start once plan 3's Cadence gate passes. Tasks 4–6 (orders) are **blocked until plan 3's holdout run passes and Andy says go.**

**Goal:** Run the frozen pipeline on live 15m bars from IB Gateway (paper, port 4002). First log would-trade signals and prove they match the backtest exactly. Then place bracket orders on the paper account with safety rails, and measure live vs backtest.

**Architecture:**
- `musical/live_state.py` keeps a rolling bar buffer per ticker and recomputes the causal pipeline on the buffer's tail each new bar: features → onsets/beats/meter → measures → keys → chords → cadence.
- `scripts/harmony_live.py` replaces the daily `paper_trade.py` for this project. It is a long-running loop that wakes after each 15m bar closes, appends signals to a log, and (when enabled) places orders through `musical/orders.py`.
- `scripts/paper_report.py` compares paper fills with backtest expectations.

**Tech Stack:** as before, plus ib_async 2.1 (`IB.bracketOrder`, `placeOrder`, `openTrades`, `reqPositions`).

**Spec:** `docs/superpowers/specs/2026-10-13-market-harmony-design.md` (gate 6: paper trading), plus the safety rails in `docs/plan.md` Part 3.

## Global Constraints

- **Paper only.** `broker.connect` already refuses live ports unless `live=True`. This plan never passes `live=True`.
- **Kill switch:** a `STOP` file in the repo root makes the loop exit before fetching or ordering, every cycle.
- **Frozen models:** the live runner loads exactly the paths recorded in `docs/holdout_run.json`, and refuses others unless `--dry-run`.
- **Warm-up:** buffers hold ≥ 90 sessions of 15m bars. That covers the 40-session seasonal norm, the 520-bar tempo window, the 32-measure key lookback, the 16-measure cadence window and the 256-bar listener context.
- **Only completed bars:** a bar is used only after its end time + 30 s. Never feed a forming bar.
- **Same trade rule as the backtest** (plan 3 Global Constraints). Live signals and orders use identical targets, stops and deadlines.
- **Account endpoints timed out on this Gateway before** (CLAUDE.md). Order state must not depend on `accountSummary`.

## Review Focus

1. **Gateway disconnect mid-session:** reconnect with backoff, re-sync open trades from `openTrades()`, never duplicate an order. Test in Task 4.
2. **Restart with open positions:** state is rebuilt from the order log + `reqPositions`, and positions past their deadline are closed. Test in Task 5.
3. **A missing or late bar** (IBKR pacing, a halt): the cycle skips that ticker, logs it, and never fills gaps with guesses. Test in Task 1.
4. **Half days and holidays:** no signal on a bar whose deadline would fall after a scheduled early close without the next session counted correctly. Test in Task 2.
5. **Signal ≠ backtest:** any divergence on replayed data fails the parity check loudly. Test in Task 2.

---

### Task 1: Rolling live state

**Files:**
- Create: `musical/live_state.py`
- Test: `tests/test_live_state.py`

**Interfaces:**
- Consumes:
  - `broker.connect`, `broker.fetch_history`;
  - all pipeline functions from plans 1–3;
  - `load_listener`, `ChordVocab.load`, the names table CSV, the CadenceNet `load` (same pattern as `musical/cnn.load_model`).
- Produces:
  - `Pipeline.from_holdout_record(path="docs/holdout_run.json")` loads every frozen artifact.
  - `LiveState(pipeline, buffer_sessions=90)`:
    - `.seed(ticker, bars)` loads history;
    - `.append(ticker, bar_row) -> bool`, which is False and logs when the bar is out of order or duplicated;
    - `.signal(ticker) -> dict | None` returns `ticker, timestamp, dest, p_dest, direction, entry_ref, target, stop, deadline_bar_time, chord_name, key_list`.
  - `is_complete(bar_end, now, grace_s=30) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
def test_append_rejects_duplicate_and_out_of_order_bars():
def test_forming_bar_is_not_complete_until_grace_passes():
def test_missing_bar_skips_signal_and_logs():
```

- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `git commit -m "Add rolling live pipeline state"`

### Task 2: Parity check: live signals equal backtest signals

**Files:**
- Create: `scripts/parity_check.py`
- Test: `tests/test_live_state.py`

**Interfaces:**
- Produces: `replay(pipeline, bars_by_ticker, start, end) -> pd.DataFrame`. It feeds bars one at a time through `LiveState` and collects signals.

- [ ] **Step 1: Write the failing tests**

```python
def test_replay_signals_match_batch_signals_exactly():     # small fixture: replay vs batch pipeline -> identical rows
def test_no_signal_when_deadline_crosses_early_close_incorrectly():
```

- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass.
- [ ] **Step 5:** Run `scripts/parity_check.py --start 2023-10-01 --end 2023-12-31` on real data. Expected: 0 mismatched signals. Commit: `git commit -m "Add live/backtest parity check"`

### Task 3: Dry-run listening loop

**Files:**
- Create: `scripts/harmony_live.py`
- Modify: `CLAUDE.md` (commands)

**Interfaces:**
- Produces: `scripts/harmony_live.py [--dry-run] [--port 4002] [--universe data/universe_intraday.txt]`.
  - It seeds buffers via `fetch_history` (≥ 90 sessions), then each cycle sleeps until the next 15m bar end + 30 s.
  - Each cycle fetches the last 2 hours per ticker, appends new completed bars, and writes signals to `data/live_signals.csv` (gitignored), including chord name and key list.
  - It checks `STOP` at the top of every cycle and only runs during RTH (09:30–16:00 America/New_York, exchange calendar from the bars).

- [ ] **Step 1:** Implement with the existing `broker` helpers. Write unit tests with the fake IB object pattern from `tests/test_broker.py`:
  - `test_stop_file_exits_before_fetch`
  - `test_cycle_writes_signal_rows`
- [ ] **Step 2:** Run `.venv/bin/pytest -q`. Expected: all pass.
- [ ] **Step 3:** Run against the Gateway for one session with `--dry-run`. Compare the logged signals with `parity_check.py` over the same day. Expected: identical.
- [ ] **Step 4:** Commit: `git commit -m "Add dry-run live harmony loop"`. **Run the dry run for ≥ 2 weeks**, then report to Andy.

### Task 4: Bracket orders with idempotency (blocked until holdout pass + Andy's go)

**Files:**
- Create: `musical/orders.py`
- Test: `tests/test_orders.py`

**Interfaces:**
- Produces:
  - `RiskLimits(max_positions=10, risk_frac=0.01, max_daily_loss_frac=0.02, max_orders_per_day=40)`
  - `size_shares(equity, entry, stop, limits) -> int`, using the same formula as `run_portfolio`.
  - `place_bracket(ib, contract, signal, shares, order_ref) -> list[Trade]`, built from `IB.bracketOrder`:
    - the parent is a MARKET order at the next bar's open, i.e. sent right after the signal;
    - the take-profit is a LIMIT at the target;
    - the stop-loss is a STOP at the stop.
  - `order_ref = f"{ticker}-{timestamp:%Y%m%d%H%M}"`, which is unique per signal. If `openTrades()` already has that ref, nothing is placed.
  - `close_expired(ib, positions, now)` sends a market close at the deadline and cancels the bracket children.

- [ ] **Step 1: Write the failing tests** (fake IB)

```python
def test_same_signal_never_places_twice_after_reconnect():
def test_bracket_prices_match_signal_target_and_stop():
def test_daily_loss_limit_blocks_new_orders():
def test_max_positions_blocks_new_orders():
def test_expired_position_is_closed_and_children_cancelled():
```

- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `git commit -m "Add bracket order placement with safety rails (paper)"`

### Task 5: Order loop, restart recovery, order log

**Files:**
- Modify: `scripts/harmony_live.py` (add `--enable-orders`)
- Test: `tests/test_orders.py`

**Interfaces:**
- `--enable-orders` requires all of: port 4002, `docs/holdout_run.json` present with a PASS, and no `STOP` file.
- Every placement, fill, cancel and close appends to `data/order_log.csv`.
- On start, state is rebuilt from `data/order_log.csv` plus `reqPositions()` (wrapped in a 10 s timeout and retried; if it times out, start in **no-new-orders** mode and log loudly).

- [ ] **Step 1: Write the failing tests:**
  - `test_restart_rebuilds_open_positions_from_log`
  - `test_positions_timeout_starts_in_no_new_orders_mode`
  - `test_enable_orders_refused_on_live_port`
- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass.
- [ ] **Step 5:** Probe the Gateway: run `reqPositions`/`openTrades` against the paper account and record whether they still time out (CLAUDE.md says account endpoints did). If they do, fix the Gateway config with Andy before enabling orders. Commit: `git commit -m "Order loop with restart recovery and order log"`

### Task 6: Paper evaluation

**Files:**
- Create: `scripts/paper_report.py`
- Modify: `docs/experiments.md`

- [ ] **Step 1:** Implement `scripts/paper_report.py --since YYYY-MM-DD`. It joins `data/order_log.csv` with a replayed backtest over the same dates and reports, per trade and in total:
  - realized R vs backtest R;
  - fill slippage vs the 2 bps assumption;
  - missed or extra trades.
- [ ] **Step 2:** Pre-register in `docs/experiments.md` and commit. Run 2–3 months on paper. The success criteria are:
  1. live net R is within the backtest's bootstrap 90% interval;
  2. median slippage ≤ 2× the assumption;
  3. zero duplicate or orphaned orders.
- [ ] **Step 3:** Run monthly. Log each report. **Any decision about real money is Andy's**, and it is outside this plan.
