# Cadence Layer, Backtest and Holdout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** provisional.
- **Prerequisite:** plan 2 (`2026-10-13-harmony.md`) is built and its Vocabulary and Harmony gates **passed**.
- Re-read those results first. If chord roles came back thin (e.g. no applied chords survived), update the destination classes and baselines below and commit before writing code.

**Goal:** Recognize cadences, predict which key they land in (possibly a new one) and so their direction, then prove it:
1. on the 2023 test split, once;
2. in a walk-forward backtest with costs;
3. finally, once, on the 2024+ holdout.

**Architecture:**
- `musical/cadence.py` builds labels: which candidate key price reaches first within H measures, detours allowed.
- `musical/cadence_net.py` is the CNN head. It is a small conv over the last N measures (chords, per-key readings, listener states, ticker embedding = composer) and outputs P(destination).
- `musical/backtest.py` gains level-target trades and a non-alphabetical tie-break.
- Scripts run the Cadence gate, the walk-forward Backtest gate, and the one-time holdout.

**Tech Stack:** as plans 1–2.

**Spec:** `docs/superpowers/specs/2026-10-13-market-harmony-design.md` (Layer 3, trading rule, gates 4–6)

## Global Constraints

- Plans 1–2 constraints hold.
- **This plan is the first allowed to load 2023 data**, and only in the steps that say so.
- 2024+ is loaded **only** by `cadence_final.py`, and only behind `--final`.
- **Destination classes (relative, per the spec):**
  - `0` = current key1 (home);
  - `1..4` = key2..key5 at signal time;
  - `5` = "new key up" (price rises ≥ 3 ATR beyond every candidate above it);
  - `6` = "new key down" (the mirror);
  - `7` = NONE (nothing reached within H).
- **Reached** = a bar's high/low crosses within `0.25` ATR of the center.
- **Horizon H = 8 measures**, converted to bars at signal time as `8 × current measure length in bars`, so nothing after the signal is used to set it.
- **Trade rule (pre-registered here):**
  - enter at the next bar's open toward the predicted destination;
  - target = that destination's center (classes 5/6: entry ± 3 ATR);
  - protective stop = the mirror of the target distance, minimum 1.5 ATR;
  - time stop H measures (in bars as above);
  - only when P(destination) ≥ cutoff, with the cutoff chosen on val; never class 7 or a destination within 0.5 ATR of price.
- **Portfolio:** 1% risk per trade, max 10 positions, IBKR costs + 2 bps per side, pessimistic fills (as now).
- **Selection** uses val only. Test is scored once per selected model and every attempt is logged.

## Review Focus

1. **Destination key already at price** (the home key is where price sits): no trade (distance < 0.5 ATR), never a zero-risk division. Test in Task 3.
2. **Simultaneous signals across tickers:** take the highest P(destination) first, then random with a fixed seed. Never alphabetical (fixes the bias found in Tests C–D). Test in Task 3.
3. **A measure length that changes after the signal:** the time stop stays the bar count fixed at signal time. Test in Task 1.
4. **Key list shorter than 5** (NaN-padded): those classes are masked out of the softmax, not predicted. Test in Task 2.
5. **The holdout script run twice:** the second run refuses unless `--i-know` is passed, and any re-run is logged as such. Test in Task 6.

---

### Task 1: Cadence labels

**Files:**
- Create: `musical/cadence.py`
- Test: `tests/test_cadence.py`

**Interfaces:**
- Consumes: the harmony dataset (plan 2, Task 3), bars per ticker.
- Produces:
  - `DEST_CLASSES = 8`
  - `cadence_labels(bars, measures, keys, H=8, reach_atr=0.25, new_key_atr=3.0) -> pd.DataFrame`, one row per measure, with columns:
    - `dest` (int 0..7);
    - `dest_price` (float, NaN for 7);
    - `bars_to_dest`;
    - `deadline_bar` (position, fixed at signal time);
    - `entry` (next bar's open);
    - `atr`.
  - The first event wins. Price may wander anywhere before (detours count).

- [ ] **Step 1: Write the failing tests**

```python
def test_detour_then_home_is_labeled_home():           # dips 1.5 ATR, then reaches key1 -> dest 0
def test_reaching_second_key_first_is_class_1():
def test_breakout_beyond_all_keys_is_new_key_up():
def test_nothing_reached_is_none():
def test_deadline_is_fixed_at_signal_time():            # measure lengthens later -> deadline unchanged
def test_labels_use_only_bars_after_signal_and_before_deadline():
```

- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `git commit -m "Add cadence destination labels (relative keys, detours allowed)"`

### Task 2: Cadence network (the CNN recognizer)

**Files:**
- Create: `musical/cadence_net.py`, `scripts/cadence_train.py`
- Test: `tests/test_cadence_net.py`

**Interfaces:**
- Consumes: the harmony dataset + `cadence_labels`; `name_chords`/`read_measure` (plan 2, Task 4); the listener measure states (plan 2 descriptors).
- Produces:
  - `MEASURES_N = 16`
  - `cadence_inputs(df, names) -> (X: (n, C, 16) float32, mask: (n, 8) bool)`. Per measure in the window, the channels are:
    - a one-hot chord (k);
    - for each of the 5 keys: relative position (ATR), conf, degree index / 7;
    - descriptor PCA-16 (fit on train);
    - `top_age`, a `modulation` flag.

    `mask` marks which destination classes exist (keys present; 5/6/7 always present).
  - `CadenceNet(n_channels, ticker_vocab, emb_dim=8, dropout=0.3)`. Two conv blocks (kernel 3) over the 16 measures, then mean+max pooling, concatenated with the ticker embedding (slot 0 = unseen ticker). The head outputs 8 logits, with masked classes set to −inf. There is an auxiliary head for the next chord (k logits), with loss weight 0.3.
  - `train_cadence(...)` reuses the `musical/cnn.py` pattern: TrainConfig fields, early stopping on val destination loss.
  - `predict_dest(model, X, mask, tickers) -> (n, 8)` probabilities.
  - `save_cadence(model, path)` / `load_cadence(path) -> CadenceNet`, which also stores the PCA and input standardization (pattern: `musical/cnn.save_model`).
  - `scripts/cadence_train.py --listener <path> --emb {8,0} --dropout {0.3,0.5} --seed {0,1,2}` appends to `models/trained_params.csv`.

- [ ] **Step 1: Write the failing tests**

```python
def test_masked_classes_get_zero_probability():
def test_net_learns_planted_chord_to_destination_rule():  # chord 2 -> dest 1 always; val acc > 0.9 after few epochs (cpu)
def test_unseen_ticker_uses_slot_zero():
```

- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `git commit -m "Add CadenceNet destination model and training script"`

### Task 3: Backtester: level targets and fair tie-breaks

**Files:**
- Modify: `musical/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces:
  - `simulate_level_trades(bars_by_ticker, signals, cost) -> pd.DataFrame`. Signals carry `ticker, timestamp, direction, target, stop, deadline_bar, score`.
    - Same columns as `simulate_trades`.
    - `risk_per_share = |entry − stop|`.
    - It reuses `_exit_for`'s fill rules, generalized to asymmetric target/stop levels and a bar deadline.
    - Signals with |target − entry| < 0.5 ATR or risk ≤ 0 are skipped.
  - `run_portfolio(trades, cost, ..., order="score", seed=0)`. Same-timestamp entries are ordered by `score` desc, then a seeded shuffle. `order="legacy"` keeps the old behavior so Tests C–D stay reproducible.

- [ ] **Step 1: Write the failing tests**

```python
def test_level_trade_hits_asymmetric_target():
def test_level_trade_gap_through_stop_fills_at_open():
def test_level_trade_time_stop_at_deadline_bar():
def test_zero_distance_target_is_skipped():
def test_simultaneous_entries_take_highest_score_not_alphabetical():
def test_legacy_order_reproduces_old_results():   # same as current run_portfolio output on an existing fixture
```

- [ ] **Step 2:** Run. Expected: the new tests FAIL. **Step 3:** Implement. **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass, including all old backtest tests.
- [ ] **Step 5:** Commit: `git commit -m "Backtester: level targets, bar deadlines, score-ordered entries"`

### Task 4: Cadence gate (test 2023, once)

**Files:**
- Create: `scripts/cadence_gate.py`
- Modify: `docs/experiments.md`

- [ ] **Step 1: Pre-register and commit:**
  - **Selection:** the 12-run grid (emb {8, 0} × dropout {0.3, 0.5} × seed {0, 1, 2}), selected by val destination loss. The signal cutoff is chosen on val net expectancy from keep-fractions {0.5, 0.3, 0.2, 0.1} of P(chosen dest), with ≥ 200 val trades.
  - **Baselines** trade the same signal times:
    - coin flip;
    - always-long to key above / short to key below;
    - **home-reversion** (always target key1, i.e. plain mean reversion);
    - **naming-only** (the destination implied by the chord name's behavior table, no CNN);
    - logistic regression on the same inputs.
  - **Pass** (test 2023, pre-cost R, level-trade rule): CNN mean R > 0, and CNN beats **every** baseline on mean R, with a bootstrap 95% CI of (CNN − best baseline) > 0. Direction accuracy is reported too.
- [ ] **Step 2:** Implement `scripts/cadence_gate.py <model> --test-end 2024-01-01`. It loads data < 2024-01-01 only.
- [ ] **Step 3:** Run once. Log verbatim. Commit: `git commit -m "Run Cadence gate"`. If FAIL: **stop and report to Andy.**

### Task 5: Backtest gate (walk-forward with costs)

**Files:**
- Create: `scripts/cadence_walk_forward.py`
- Modify: `docs/experiments.md`

- [ ] **Step 1: Pre-register and commit.**
  - **Folds:** test years 2021, 2022, 2023. Each retrains CadenceNet on measures before Y−1, early-stopped on Y−1. The listener, vocabulary and names stay fixed: they were fit < 2021.
  - **Signals** go through `simulate_level_trades` + `run_portfolio(order="score")`.
  - **Pass** (all of the following):
    1. Net expectancy > 0 at 1× and 2× costs.
    2. It beats the home-reversion and naming-only baselines on net expectancy on the same signal times.
    3. Random same-size signal subsets (200) are beaten at p ≤ 0.05.
    4. Sharpe > always-long-to-key-above on the same times.
- [ ] **Step 2:** Implement (reuse `long_filter_eval.score`-style helpers; move shared helpers into `musical/experiment.py` if two scripts need them).
- [ ] **Step 3:** Run once. Log. Update CLAUDE.md. Commit. **Stop and report to Andy** whether we spend the holdout.

### Task 6: Holdout (2024-01-01 → latest), once

**Files:**
- Create: `scripts/cadence_final.py`
- Test: `tests/test_cadence_final.py`

**Interfaces:**
- Produces: `scripts/cadence_final.py --final`.
  - It refits CadenceNet on all data < 2024-01-01 (with the selected hparams and 2023 as val), freezes the listener, vocabulary and names, and trades 2024-01-01 → the last bar.
  - It writes `docs/holdout_run.json`: timestamp, git sha, model paths (listener, vocab, names CSV, CadenceNet), metrics, and `"result"`.
  - `"result"` is `"PASS"` only if:
    - net expectancy > 0 at 1× costs and at 2× costs; and
    - it beats the home-reversion baseline on the same signal times.

    It is `"FAIL"` otherwise. These rules are fixed here, before anything is run. If that file exists, it refuses unless `--i-know` is passed, and appends a "RERUN" note to `docs/experiments.md`.

- [ ] **Step 1: Write the failing test**

```python
def test_final_refuses_second_run_without_flag(tmp_path):   # existing holdout_run.json -> SystemExit
```

- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass.
- [ ] **Step 5:** Get Andy's explicit go-ahead, then run once and log verbatim, whatever the result. Commit: `git commit -m "Holdout run"`.
