# Listener + Rhythm Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the self-supervised "listener" on 15m bars, build the causal rhythm layer (onsets, tempo, beats, meter), and run the Rhythm gate.

**Architecture:**
- `musical/features.py` turns protocol bars into a causal per-bar feature frame: price, volume, clock and market context.
- `musical/rhythm.py` computes onset strength, tempo, an online beat grid and meter from the past only.
- `musical/listener.py` is a causal dilated-conv encoder (a TCN, the CNN family) trained to predict the next bars' features, including onset strength. A clock-only twin is trained as the baseline.
- `scripts/rhythm_gate.py` compares real vs clock-preserving surrogates and full vs clock-only listener, with pass rules pre-registered in `docs/experiments.md`.

**Tech Stack:** Python 3.14 venv (`.venv`), numpy 2.5, pandas 3.0, torch 2.14 (CUDA, Quadro RTX 5000), pytest.

**Spec:** `docs/superpowers/specs/2026-10-13-market-harmony-design.md`

**Deferred to the next plan (deliberately):**
- beat-synchronous resampling (spec Layer 1 step 5): only needed by the harmony layer;
- the masked-reconstruction objective: next-bars prediction alone is the expectation signal this gate needs;
- volume-at-price / key features.

## Global Constraints

- **Causal everywhere:** every feature, onset, tempo, beat and meter value at bar t uses bars ≤ t. Each module gets a truncation test: compute on the full series and on series[:n], and the first n rows must match.
- **Data:** IBKR 15m RTH bars in `data/raw_ib/*_15m.csv` (30 names; SPY and QQQ are both tradable names and the market-context source).
- **Splits for this slice:**
  - train < `2021-01-01`, val `2021-01-01` ≤ t < `2023-01-01`;
  - **no data ≥ `2023-01-01` is loaded by any script in this slice** (2023 test and 2024+ holdout stay untouched);
  - windows never straddle a split boundary.
- **Seasonal norm:** 40 sessions, same time of day (`config.INTRADAY_SEASONAL_DAYS`).
- **Code style:** `#####` banner header (name, Andy Kwak, purpose, @requires/@ensures); `#@requires`/`#@ensures` comments on every function; scripts use argparse + `if __name__ == "__main__": main()`.
- **Experiment discipline:** gate thresholds are written to `docs/experiments.md` and committed **before** `rhythm_gate.py` is first run on real data.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Half-day sessions (14 bars instead of 26):** features, onsets and the beat tracker must run without errors or NaN beyond warm-up. Test in Task 1 and Task 3.
2. **Missing bars / ragged histories** (PEP starts 2017-12-20, AVGO 2016-02-02, some files miss ~5 bars): market context must align on timestamp, and missing context is filled with 0, not dropped. Test in Task 1.
3. **Zero-volume or zero-range bars:** every feature stays finite (no ±inf from log/divide). Test in Task 1.
4. **Warm-up NaN** (first 40 sessions of seasonal stats, tempo window): training windows containing NaN are skipped, never fed to the model. Test in Task 4.
5. **Session-open gap bars** dominate onsets. That is allowed (the clock is learned), but the first-bar-of-session flag must be set correctly, including after holidays. Test in Task 1.

---

### Task 1: Per-bar features

**Files:**
- Create: `musical/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces:
  - `PRICE_COLS = ["ret_atr", "range_atr", "body_atr", "vol_surprise", "gap_atr"]`
  - `CLOCK_COLS = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "session_pos", "first_bar"]`
  - `bar_features(bars: pd.DataFrame, seasonal_days: int = 40, atr_n: int = 14) -> pd.DataFrame`, indexed like bars, columns `PRICE_COLS + CLOCK_COLS`.
  - `add_context(feats: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.DataFrame`, which adds `f"{name}_ret_atr"` and `f"{name}_vol_surprise"` per context name, aligned on timestamp, with missing values filled with 0.0.

Definitions:
- `ret_atr` = (close − prev close) / ATR(t)
- `range_atr` = (high − low) / ATR
- `body_atr` = (close − open) / ATR
- `gap_atr` = (open − prev close) / ATR on the first bar of a session, else 0
- `vol_surprise` = log(volume+1) − mean of log(volume+1) at the same time of day over the previous `seasonal_days` sessions (exclusive of today)
- `tod` = minutes since 09:30 over 390
- `session_pos` = bar index within session / 25
- `first_bar` = 1.0 on the first bar of each date
- ATR uses `musical.tsa.true_range`, rolling mean over `atr_n`

- [ ] **Step 1: Write the failing tests**

```python
def test_features_are_causal():            # full vs bars[:3000]: first 3000 rows equal (assert_frame_equal)
def test_first_bar_flag_marks_each_session_start_including_after_holiday():
def test_half_day_session_has_finite_features():   # synthetic 14-bar day after warm-up: no NaN/inf
def test_zero_volume_and_zero_range_bars_stay_finite():
def test_vol_surprise_is_zero_for_constant_volume_profile():  # same volume every day -> ~0 after warm-up
def test_context_aligns_on_timestamp_and_fills_missing_with_zero():
    # context missing one timestamp -> that row's SPY_ret_atr == 0.0, row count unchanged
```

Use the `intraday_bars` helper pattern from `tests/test_tsa.py` (26 bars/session from 09:30).

- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_features.py -v`. Expected: FAIL (module not found).
- [ ] **Step 3:** Implement `bar_features` and `add_context` in `musical/features.py`. The seasonal mean is grouped by `index.time`, then `shift(1).rolling(seasonal_days).mean()`.
- [ ] **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass (64 + 6).
- [ ] **Step 5:** Commit: `git add musical/features.py tests/test_features.py && git commit -m "Add causal per-bar features (price, volume, clock, context)"`

### Task 2: Onset strength and tempo

**Files:**
- Create: `musical/rhythm.py`
- Test: `tests/test_rhythm.py`

**Interfaces:**
- Consumes: `bar_features` columns `ret_atr`, `range_atr`, `vol_surprise`.
- Produces:
  - `onset_strength(feats: pd.DataFrame) -> pd.Series`. Takes the mean of three causal z-scores (each over the trailing 520 bars):
    - |ret_atr|;
    - max(vol_surprise, 0);
    - turn = 1.0 where sign(ret_atr) ≠ sign(previous nonzero ret_atr), else 0.0.
  - `tempo(onset: pd.Series, window: int = 520, min_lag: int = 2, max_lag: int = 24) -> pd.DataFrame` with columns `period` (bars, int) and `strength` (peak autocorrelation, float). Computed every bar from the trailing `window` onsets: the argmax over lags in [min_lag, max_lag] of the normalized autocorrelation. Before `window` bars: NaN.

- [ ] **Step 1: Write the failing tests**

```python
def test_tempo_recovers_period_of_synthetic_pulse():   # onset spikes every 5 bars + noise -> median period == 5, strength > 0.5
def test_tempo_strength_is_low_for_white_noise():      # median strength < 0.2
def test_tempo_tracks_a_slowdown():                    # period 4 for 2000 bars then 7 -> last-500 median == 7
def test_onset_and_tempo_are_causal():                 # truncation test on both
```

- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_rhythm.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Vectorize the autocorrelation with a strided view or rolling sums per lag; a Python loop over 70k bars × 23 lags is fine if under ~10 s per ticker.
- [ ] **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `git commit -m "Add causal onset strength and tempo estimation"`

### Task 3: Online beat grid and meter

**Files:**
- Modify: `musical/rhythm.py`
- Test: `tests/test_rhythm.py`

**Interfaces:**
- Produces:
  - `track_beats(onset: pd.Series, period: pd.Series, tol: float = 0.34, q: float = 0.6) -> pd.Series` (bool, True on beat bars)
  - `meter(onset: pd.Series, beats: pd.Series, n_beats: int = 32, choices=(2, 3, 4)) -> pd.DataFrame` with columns `beats_per_measure` (int) and `downbeat` (bool, on beat bars)

The beat algorithm (not determined by the tests): walk bars in order, keeping `last` = the last beat bar.
- At bar t with p = period[t] (skip while NaN), let d = t − last.
- Mark a beat if d ≥ round(p·(1−tol)) **and** onset[t] ≥ the q-quantile of the trailing 520 onsets.
- Also mark a beat if d ≥ round(p·(1+tol)), which forces a beat to keep the pulse.

Meter: over the last `n_beats` beats, accent = onset at each beat. For each m in `choices`, compute the autocorrelation of the accent sequence at lag m, and pick the argmax. `downbeat` is True where beat index mod m equals the phase with the highest mean accent over the last `n_beats` beats.

- [ ] **Step 1: Write the failing tests**

```python
def test_beats_follow_a_steady_pulse():          # spikes every 6 bars -> median beat interval == 6
def test_beats_keep_time_through_a_missing_onset():  # one spike removed -> still a beat within ±2 bars of it
def test_meter_finds_three_beat_accent_pattern():    # pulse every 4 bars, every 3rd spike 3x larger -> beats_per_measure mode == 3, downbeats on large spikes
def test_beats_and_meter_are_causal():
def test_half_day_sessions_do_not_break_tracking():  # real-shaped index with a 14-bar day -> runs, no exception
```

- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_rhythm.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement `track_beats` and `meter`.
- [ ] **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `git commit -m "Add online beat tracking and meter detection"`

### Task 4: The listener (causal TCN) and its training loop

**Files:**
- Create: `musical/listener.py`
- Test: `tests/test_listener.py`

**Interfaces:**
- Consumes: the feature frame from Task 1 plus an `onset` column from Task 2.
- Produces:
  - `TARGET_COLS = ["ret_atr", "range_atr", "vol_surprise", "onset"]`; `HORIZON_K = 4`.
  - `ListenerConfig` dataclass:
    - `window=256`, `channels=64`, `levels=7` (dilations 1..64, kernel 3), `emb_dim=8`, `dropout=0.1`;
    - `lr=1e-3`, `weight_decay=1e-4`, `batch_size=128`, `steps_per_epoch=500`, `epochs=40`, `patience=5`;
    - `device` = cuda if available, `seed=0`.
  - `Listener(nn.Module)`, built as `Listener(n_inputs, ticker_vocab, cfg)`:
    - `forward(x[B,C,T], ticker_idx[B]) -> (B, len(TARGET_COLS)*HORIZON_K, T)`. The output at t predicts targets at t+1..t+K and depends only on x[..., ≤t].
    - Ticker embedding slot 0 is reserved for unseen tickers, as in `musical/cnn.py`.
  - `make_windows(series_by_ticker: dict[str, pd.DataFrame], input_cols, start, end, window) -> list[tuple[str, int]]`: (ticker, end_position) pairs whose whole window plus K future bars lie in [start, end) with no NaN.
  - `train_listener(series_by_ticker, input_cols, cfg, train_range, val_range) -> (model, history)`:
    - MSE on standardized targets (train-set mean/std, stored on the model);
    - early stopping on val loss;
    - the history dict has `train_loss`, `val_loss`, `val_loss_onset`.
  - `save_listener(model, path)` / `load_listener(path) -> Listener`.

- [ ] **Step 1: Write the failing tests**

```python
def test_listener_output_at_t_ignores_future_inputs():   # perturb x[..., 100:] -> out[..., :100] unchanged (atol 1e-6), eval mode
def test_windows_skip_nan_and_respect_split_end():       # NaN at bar 300 -> no window covers it; no window+K crosses `end`
def test_listener_overfits_tiny_periodic_series():       # 1 ticker, period-5 onset, cpu, 3 epochs -> val_loss_onset falls below 50% of epoch-1
def test_save_load_round_trip_gives_same_predictions(tmp_path):
```

- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_listener.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Causal convs use left padding `(kernel−1)·dilation`. Each residual block is conv → ReLU → dropout → conv, plus a skip. The embedding is broadcast over T and concatenated to the input. The loss is computed on the last `window//2` positions only, so each prediction sees at least half a window of context.
- [ ] **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `git commit -m "Add causal TCN listener with next-bars self-supervised objective"`

### Task 5: Train the listener (full and clock-only)

**Files:**
- Create: `scripts/listener_train.py`
- Modify: `musical/listener.py` (add `load_series`), `musical/config.py` (add `LISTENER_TRAIN_END = "2021-01-01"`, `LISTENER_VAL_END = "2023-01-01"`)
- Test: `tests/test_listener.py`

**Interfaces:**
- Consumes: `bar_features`, `add_context`, `onset_strength`, `train_listener`, `save_listener`.
- Produces:
  - `musical.listener.load_series(raw_dir, end) -> dict[str, pd.DataFrame]`, truncated before `end`. Context is SPY and QQQ for every ticker, and onset is included. Test: `test_load_series_never_returns_bars_at_or_after_end` (two tiny protocol CSVs in tmp_path).
  - Saved models `models/listener_{variant}_{timestamp}.pt`, with a row appended to `models/trained_params.csv` (existing `PARAMS_COLUMNS` from `scripts/model_train.py`; `dataset` = `"raw_ib_15m"`).
  - CLI: `--variant full|clock`, `--epochs`, `--seed`, `--tag`.
    - `full` inputs = PRICE_COLS + CLOCK_COLS + context cols + `onset`.
    - `clock` inputs = CLOCK_COLS only (same targets, same ticker embedding).

- [ ] **Step 1:** Write `test_load_series_never_returns_bars_at_or_after_end`, see it fail, implement `load_series`, see it pass. Then implement the script. It loads only bars before `LISTENER_VAL_END` (assert the max timestamp is < 2023-01-01).
- [ ] **Step 2:** Smoke run: `.venv/bin/python scripts/listener_train.py --variant full --epochs 1 --tag smoke`. Expected: finishes, a model file is written, and a CSV row is appended. Then delete the smoke model and its CSV row.
- [ ] **Step 3:** Train both variants in the background:
  - `.venv/bin/python scripts/listener_train.py --variant full --tag listener`
  - `.venv/bin/python scripts/listener_train.py --variant clock --tag listener`

  Expected: early stop, and the history is printed per epoch.
- [ ] **Step 4:** Commit the script and config: `git commit -m "Add listener training script; train full and clock-only listeners"`

### Task 6: Clock-preserving surrogate

**Files:**
- Create: `musical/surrogates.py`
- Test: `tests/test_surrogates.py`

**Interfaces:**
- Produces: `clock_shuffle(feats: pd.DataFrame, cols, rng) -> pd.DataFrame`. For each time-of-day slot, it permutes the rows of `cols` across dates (each column's rows move together). It preserves every slot's exact distribution, i.e. the clock, and destroys sequential structure. Other columns are untouched.

- [ ] **Step 1: Write the failing tests**

```python
def test_clock_shuffle_preserves_each_slot_distribution():   # sorted values per time-of-day identical
def test_clock_shuffle_destroys_within_session_periodicity(): # period-5 synthetic -> tempo strength median drops below 0.2
def test_clock_shuffle_keeps_clock_driven_pattern():          # U-shaped volume by slot unchanged in mean per slot
```

- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run `.venv/bin/pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `git commit -m "Add clock-preserving surrogate"`

### Task 7: Pre-register and run the Rhythm gate

**Files:**
- Modify: `docs/experiments.md` (pre-registration first, results after)
- Create: `scripts/rhythm_gate.py`
- Modify: `CLAUDE.md` (current state, commands, test count)

**Interfaces:**
- Consumes: `load_series` (Task 5), `tempo`, `track_beats`, `meter`, `clock_shuffle`, `load_listener`, and the two trained listeners (paths passed as args).

- [ ] **Step 1:** Append a pre-registration to `docs/experiments.md` and commit it **before** running the script. The gate passes only if all three checks hold:

  | Check | Data | Compare against | Pass rule |
  |---|---|---|---|
  | Periodicity | 2016–2022, per ticker-year | Clock-shuffled surrogate (seed 0) | Real median tempo `strength` beats it in ≥ 70% of ticker-years |
  | Tempo stability | 2016–2022, per ticker-year | Same surrogate | Real median |Δ log period| between consecutive sessions is lower in ≥ 70% of ticker-years |
  | Beyond the clock | Val period 2021–2022 | Clock-only listener | Full listener's onset loss (val MSE on standardized onset, t+1..t+4) is lower by ≥ 2% relative overall, and lower in ≥ 70% of tickers |

  The script also reports (informational only):
  - the meter distribution;
  - median beat interval per ticker in bars;
  - which onset ingredient correlates most with beats.

  If the median beat interval is ≤ 2 bars, the next step is fetching 5m data per the spec.
- [ ] **Step 2:** Implement `scripts/rhythm_gate.py` (CLI: `full_model`, `clock_model`, `--seed`). It prints one table per check, then `Rhythm gate: PASS|FAIL`.
- [ ] **Step 3:** Run: `.venv/bin/python scripts/rhythm_gate.py models/listener_full_*.pt models/listener_clock_*.pt`.
- [ ] **Step 4:** Log the results verbatim in `docs/experiments.md`. Update `CLAUDE.md`: current state (new direction, listener + rhythm built, gate result), new commands, test count.
- [ ] **Step 5:** Commit: `git commit -m "Run Rhythm gate and log results"`. Stop and report to Andy, pass or fail (spec: checkpoint after the Rhythm gate).
