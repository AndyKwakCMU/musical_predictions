# Experiment log

This file records every configuration tried. The number of attempts matters when you judge a good-looking result (see `docs/plan.md`, overfitting guardrails).

The held-out final test period is **2024-01-01 onward** (`config.DEV_END`). As of the last entry below, it has **not** been touched.

Universe: 100 current large-cap US stocks and ETFs (`data/universe.txt`), daily bars from 2000 on. The universe has survivorship bias.

---

## 2026-10-13 — Test A (TSA gate), k=2, H=10, dev period 2000–2023

The gate is COMPRESSED when the realized-vol percentile and the Bollinger-bandwidth percentile are both ≤ 20%, ranked against the trailing 252 bars.

| group | bars | barrier hit rate | median max excursion (ATR) | median \|10-bar ret\| % |
|---|---|---|---|---|
| ALL | 548,789 | 0.642 | 2.387 | 2.93 |
| NOISE | 478,838 | 0.627 | 2.338 | 2.95 |
| LOW_VOL_CONTROL (vol pct ≤ 20%, not compressed) | 69,956 | 0.699 | 2.554 | 2.53 |
| **COMPRESSED** | 69,951 | **0.746** | **2.745** | 2.76 |
| EPISODE_START | 9,280 | 0.725 | 2.664 | 2.70 |

COMPRESSED beats NOISE in 82% of 2,082 ticker-years.

**Result: PASS.** Compression does precede larger moves in ATR terms, and it beats a volatility-matched control even in raw %. That means the bandwidth condition adds information beyond "vol is low."

An earlier run on 20 tickers, 2016–2023, gave the same picture: 0.783 vs 0.656 hit rate and 81% of ticker-years.

## 2026-10-13 — Test B (CNN direction), dataset w=64, k=2, H=10

Splits: train < 2018 (52,496 events), val 2018–2020 (4,712), test 2021–2023 (11,455). Embargoed at the boundaries.

Labels across all years: UP 40.3%, DOWN 34.6%, NONE 25.1%.

### Run 1: default CNN (ticker emb 4, dropout 0.3, seed 0)

| model | dir. acc. on UP/DOWN | mean R (pre-cost) | log loss |
|---|---|---|---|
| coin flip | 0.504 | 0.006 | |
| always long | 0.497 | 0.008 | 1.085 (prior) |
| momentum | 0.504 | 0.012 | |
| logistic | 0.514 | 0.023 | 1.080 |
| cnn | 0.494 | 0.000 | 1.086 |

### Run 2: grid selected on **validation loss only**

The grid was 12 runs: ticker emb {4, 0} × dropout {0.3, 0.5} × seed {0, 1, 2}.

Validation log loss:

| emb | dropout | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| 4 | 0.3 | 1.0551 | 1.0642 | 1.0694 |
| 4 | 0.5 | 1.0548 | 1.0558 | 1.0624 |
| 0 | 0.3 | 1.0567 | 1.0616 | 1.0667 |
| 0 | 0.5 | **1.0548** | 1.0578 | 1.0640 |

For reference, the prior scores 1.0647 on val and logistic regression scores 1.0766. Seed variance is about as large as the differences between configs.

The selected model (emb 0, dropout 0.5, seed 0) scored once on test:

| model | dir. acc. | mean R | log loss |
|---|---|---|---|
| logistic | 0.514 | 0.023 | 1.080 |
| cnn | 0.503 | 0.016 | 1.077 |

Mean R by year: 2021 +0.034, 2022 −0.030, 2023 +0.006.

**Result: FAIL.** The CNN does not beat the baselines on direction. Its log-loss gain over the prior suggests it learns *whether* a barrier is hit (a volatility question), not *which* barrier. No model has a directional edge on daily bars here.

Ideas for next iterations, not yet tried:

- Intraday bars via IBKR. Daily-bar breakout direction may simply be close to efficient.
- Predict "barrier hit vs NONE" and pair it with a non-directional structure, e.g. a straddle-like bracket, instead of direction.
- Longer windows, cross-sectional or market context (SPY state as a channel), and events at compression *break* instead of during compression.
- Stricter gate, e.g. episode starts only, at the cost of fewer events.

## 2026-10-13 — Test C (walk-forward, full system), dev years 2010–2023

Each fold works the same way. The CNN (emb 0, dropout 0.5, seed 0) trains on events before year Y−1 and early-stops on Y−1. It then trades Y's gated events through `musical/backtest.py`:

- next-open entry
- ±2 ATR bracket with a 10-bar time stop
- 1% risk per trade
- max 10 positions
- IBKR-like commission plus 2 bps slippage per side

| strategy | trades | hit rate | expectancy (net R) | CAGR | Sharpe | max DD |
|---|---|---|---|---|---|---|
| cnn | 3,629 | 0.530 | 0.027 | 1.9% | 0.31 | −21.3% |
| always long (same events) | 3,629 | 0.537 | 0.045 | 3.6% | 0.50 | −16.0% |
| coin flip | 3,629 | 0.498 | −0.018 | −2.7% | −0.44 | −37.4% |

The CNN went long on 88% of events. Its longs made +0.043R and its shorts lost 0.093R. In effect it is "always long" plus losing shorts.

**Result: FAIL** (it does not beat always-long). The 2× cost and parameter-robustness runs were skipped, since the system fails at 1× costs already.

The gated-vs-ungated CNN comparison is deferred until some model passes Test B. The holdout (2024+) is still untouched.

Takeaway: the gated events have a long drift edge (always-long +0.045R/trade), and the TSA gate is real (Test A). The missing piece is direction.

---

## 2026-10-13 — Intraday round: settings declared BEFORE seeing results

**Why:** daily-bar breakout direction showed no edge (Tests B and C). The next hypothesis is that direction is more predictable on intraday bars.

**Data:** IBKR 15-minute `TRADES` bars, regular trading hours, 2016-01-01 → 2026-10-12, for 30 liquid names (`data/universe_intraday.txt`). IBKR intraday bars are already split-adjusted. I verified this: AAPL closes around 125 on 2020-08-28, not around 500.

**Gate:** same rule (vol pct ≤ 20% and bandwidth pct ≤ 20%). The difference is that percentiles rank each bar only against the **same time of day over the previous 40 sessions**.

Without this, the gate is distorted badly. On synthetic U-shaped data, the naive gate flags 35% of 3:00pm bars versus 1% of 11:15am bars.

**Labels and backtest:** k = 2, H = 10 bars (2.5 h, may hold overnight), window 64 bars (about 2.5 sessions). Same CNN and training code.

**Splits (Test B):** train 2016–2020, val 2021–2022, test 2023. The holdout is still 2024-01-01 onward, untouched.

**Cost note:** 2 bps slippage per side is a much bigger share of a 15-minute ±2 ATR bracket than of a daily one. Expect costs to bite.

**Data notes (after download):** all 30 files cover 2016-01-04 → 2026-10-12 with about 26 bars per session. Two exceptions:
- PEP starts 2017-12-20. IBKR returns no data before PEP's NYSE→NASDAQ listing move.
- AVGO starts 2016-02-02, when the post-merger Broadcom began trading.

Both gaps fall in the train period, so both names stay in.

### Intraday Test A (15m, 30 names, dev period < 2024)

| group | bars | hit rate | median excursion (ATR) | median \|ret\| % |
|---|---|---|---|---|
| ALL | 1,519,178 | 0.588 | 2.374 | 0.429 |
| NOISE | 1,329,688 | 0.581 | 2.340 | 0.441 |
| LOW_VOL_CONTROL | 156,260 | 0.640 | 2.657 | 0.372 |
| COMPRESSED | 189,490 | 0.635 | 2.641 | 0.354 |
| EPISODE_START | 26,919 | 0.738 | 3.269 | 0.407 |

COMPRESSED beats NOISE on median excursion in 97% of 238 ticker-years, and every ticker wins at least 88% of its years.

**Result: PASS** by the plan's criterion: larger moves in ATR units than unconditional bars, consistent across tickers and years. Two caveats are stronger here than on daily bars:
- **Compression adds nothing beyond plain low volatility.** COMPRESSED (0.635) ≈ LOW_VOL_CONTROL (0.640).
- **In percent terms, compressed moves are smaller.** The median |ret| is 0.354% vs 0.441% for noise. The ATR "edge" partly reflects a small denominator, and per-trade costs are a fixed share of price. A ±2 ATR bracket on a compressed 15m bar is narrow, so costs eat a larger share of R.

### Intraday Test B (15m): train 2016–2020, val 2021–2022, test 2023

**Model selection.** I trained the same 12-run grid as on daily bars: ticker emb {4, 0} × dropout {0.3, 0.5} × seed {0, 1, 2}. Every run had 121,316 train and 44,047 val events and early-stopped after 7–11 epochs.

Validation log loss:

| emb | dropout | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| 4 | 0.3 | 0.9013 | 0.9008 | 0.8999 |
| 4 | 0.5 | 0.9012 | 0.9011 | **0.8995** |
| 0 | 0.3 | 0.9013 | 0.9010 | 0.9002 |
| 0 | 0.5 | 0.9002 | 0.9009 | 0.8996 |

The class prior scores 1.0946 on val. All 12 configs are within 0.002 of each other, so the choice is effectively noise.

**Selected:** emb 4, dropout 0.5, seed 2 (`cadence_i15_e4_d0.5_s2_20261013_225255.pt`). It was scored once on 2023 (23,086 events, pre-cost R):

| strategy | dir acc on UP/DOWN | mean R | log loss |
|---|---|---|---|
| coin flip | 0.496 | −0.016 | — |
| always long | 0.521 | **+0.035** | — |
| momentum | 0.508 | +0.004 | — |
| logistic | 0.514 | −0.003 | 0.926 |
| cnn | 0.519 | −0.006 | **0.916** |

On 2023 the class prior scores 1.0979.

**Result: FAIL.** This is the same pattern as on daily bars. The CNN beats the prior by a lot on log loss (0.916 vs 1.098), but almost all of that gain is in predicting **whether** price breaks out (NONE vs move), not **which way**:
- Its direction accuracy (0.519) does not beat always-long (0.521).
- Its mean R is negative before costs.

Per the plan, Test C is skipped because no model passed Test B. The holdout (2024+) is still untouched.

**Takeaway across both rounds.** The gate and the CNN agree on one thing: there is a learnable signal for "a big move is coming." Neither daily nor 15m bars contain a learnable signal for its direction with these inputs. Remaining directions worth trying, none of them pre-registered yet:
1. **Trade the volatility, not the direction.** For example, a long straddle or strangle at compression, using the CNN's P(move) as the filter. This needs options data.
2. **Add directional context that price windows lack.** Candidates are market or sector state (SPY/QQQ trend at the event), order-flow or imbalance data, and news or earnings calendars.
3. **Long-only with P(move) filtering.** The drift edge exists (always-long +0.035R pre-cost intraday, +0.045R net on daily). Test whether conditioning on the CNN's P(move) raises long expectancy enough to survive costs.

---

## 2026-10-13 — Test D (long-only, P(move) filter): settings declared BEFORE seeing results

**Hypothesis.** The CNN knows *whether* a move is coming, not which way. Gated events have a long drift. So going long only on gated events where the CNN's P(move) = P(UP) + P(DOWN) is high should beat going long on every gated event, after costs.

**Models.** The already-selected Test B models. Nothing is retrained.
- Daily: `cadence_grid_e0_d0.5_s0_*`. Val 2018–2020, test 2021–2023.
- 15m: `cadence_i15_e4_d0.5_s2_*`. Val 2021–2022, test 2023.

**Filter.** Keep an event if P(move) ≥ c. The cutoff c is the (1−q) quantile of P(move) over the **val** events, with q ∈ {1.0, 0.5, 0.3, 0.2, 0.1}. q = 1.0 means no filter, i.e. always-long. q is chosen as the one with the highest val net expectancy among the options with ≥ 200 val trades. If q = 1.0 wins, the filter adds nothing and the test fails.

**Scoring.** Use the real backtester (`simulate_trades` + `run_portfolio`):
- next-open entry, ±2 ATR, H = 10;
- 1% risk per trade, max 10 positions;
- commission + 2 bps slippage per side.

The cutoff from val is applied unchanged to test, which is scored once.

**Pass (all of the following on test):**
1. Filtered-long net expectancy > 0 R.
2. Filtered-long net expectancy > always-long (all gated events) net expectancy.
3. A permutation check: keep a random subset of the same fraction of the same test events, 200 times. Filtered-long must beat ≥ 95% of these random subsets on net expectancy.
4. Net expectancy stays > 0 at 2× costs.

The holdout (2024+) stays untouched. A pass on either resolution would justify a one-time `--final` holdout run of that exact configuration.

### Test D results (net of costs, real backtester)

**Daily.** Val 2018–2020 picked keep_frac 0.2 (P(move) ≥ 0.805), which scored +0.118R on val vs +0.050R for no filter. On test 2021–2023:

| strategy | trades | hit rate | net R | CAGR | Sharpe | max DD |
|---|---|---|---|---|---|---|
| always long | 841 | 0.521 | +0.010 | 4.1% | 0.47 | −12.5% |
| filtered long | 425 | 0.518 | −0.012 | 0.2% | 0.06 | −9.6% |
| filtered long, 2× cost | 425 | 0.508 | −0.028 | −0.5% | −0.08 | −9.9% |

Random same-size subsets averaged −0.003R, with a 95th percentile of +0.037R. The permutation p-value is 0.665.

**15m.** Val 2021–2022 picked keep_frac 0.1 (P(move) ≥ 0.963). Every option lost money on val, but the filter lost less: −0.047R vs −0.103R with no filter. On test 2023:

| strategy | trades | hit rate | net R | CAGR | Sharpe | max DD |
|---|---|---|---|---|---|---|
| always long | 4,651 | 0.490 | −0.084 | −16.1% | −2.59 | −17.1% |
| filtered long | 756 | 0.528 | −0.060 | −3.5% | −1.23 | −3.9% |
| filtered long, 2× cost | 756 | 0.526 | −0.189 | −7.1% | −2.53 | −7.2% |

Random same-size subsets averaged −0.065R, with a 95th percentile of −0.025R. The permutation p-value is 0.430.

**Result: FAIL on both resolutions.**
- Daily fails all 4 checks. The val improvement did not carry over to test.
- 15m passes only check 2 (it loses less than always-long). A random subset of the same size loses just as little, so this is a portfolio-crowding effect (fewer trades means fewer skipped and overlapping positions), not a P(move) signal.
- On 15m bars, costs flip the pre-cost long drift (+0.035R in Test B) to about −0.08R net. A ±2 ATR bracket on compressed 15m bars is too narrow for 2 bps per side plus commission.

Note: on daily bars, always-long takes only 841 of 11,455 test events because of the one-position-per-ticker and 10-position caps. Gated events cluster in calm markets.

The holdout (2024+) is still untouched.

---

## 2026-10-14 — Rhythm gate (plan 1, market-harmony design): settings declared BEFORE seeing results

**Context.** The project now follows `docs/superpowers/specs/2026-10-13-market-harmony-design.md`. Tests A–D above tested the old volatility-compression framing. This is the first gate of the new design: is there a market pulse, inferred from the 15m bars, that is more than the trading-day clock?

**Data.** IBKR 15m RTH bars, 30 names, 2016-01-04 → 2022-12-30. Nothing ≥ 2023-01-01 is loaded.

**Method** (`musical/rhythm.py`, all causal):
- **Onset strength:** mean of trailing-520-bar z-scores of |ret/ATR|, positive volume surprise vs the same time of day over 40 sessions, and direction turns.
- **Tempo:** the most prominent local peak of the trailing-520-bar onset autocorrelation over lags 2–24 bars. Prominence is the strength; no peak means strength 0.
- **Beats:** an online tracker.
- **Meter:** the accent autocorrelation over the last 32 beats, choosing 2, 3 or 4.

**Surrogate.** `clock_shuffle` with seed 0. Within each time-of-day slot, the rows of the onset ingredients (`ret_atr`, `vol_surprise`) are permuted across dates. Onset, tempo and beats are then recomputed on the shuffled data. This keeps the clock exactly and destroys any other rhythm.

**Listeners.** `scripts/listener_train.py`: a causal TCN predicting the next 4 bars of ret/ATR, range/ATR, volume surprise and onset. Train is < 2021, early-stopped on 2021–2022.
- **full:** price + clock + SPY/QQQ + onset.
- **clock:** clock features only.

**Pass:** all three checks below.
1. **Periodicity.** Per ticker-year (2016–2022), the real median tempo strength > the surrogate's, in ≥ 70% of ticker-years.
2. **Tempo stability.** Per ticker-year, the median |Δ log period| between consecutive sessions is lower for real than for the surrogate, in ≥ 70% of ticker-years. The session's period is taken at its last bar, and only sessions where both have a period count.
3. **Beyond the clock.** On val 2021–2022, the full listener's onset loss (MSE on the standardized onset, t+1..t+4, scored half of non-overlapping windows) is ≥ 2% lower (relative) than the clock-only listener's overall, **and** lower in ≥ 70% of tickers.

**Reported, not gated:**
- the meter distribution;
- the median beat interval per ticker, in bars;
- the correlation of each onset ingredient with beats.

If the median beat interval is ≤ 2 bars, the next step is 5m data (per the spec), not the harmony layer.

**Known caveat.** Before this pre-registration, real AAPL < 2023 tempo was looked at once, during development. It showed that a naive argmax picked lag 2 everywhere, which is why tempo uses peak prominence. No gate statistic was computed. Listener training logs (pooled val losses per epoch) were visible while these rules were written; the per-ticker check and the 2% threshold were not tuned to them.

### Rhythm gate results

**Listeners.**
- Full: `listener_full_listener_full_s0_20261014_212748.pt`, best val loss 0.6445, early-stopped after 7 epochs (best at epoch 2–3).
- Clock-only: `listener_clock_listener_clock_s0_20261014_212839.pt`, best val loss 0.8008, 10 epochs.

**Run 1 was invalid** and is logged here for the record. `clock_shuffle` moved warm-up NaN rows into the middle of the series, so every rolling window in the surrogate was NaN and checks 1–2 had no data. The fix keeps NaN rows in place (`test_clock_shuffle_keeps_nan_rows_in_place`). The rules were unchanged for run 2, and check 3 was identical in both runs.

**Run 2:**

| year | tempo strength, real | tempo strength, surrogate | drift, real | drift, surrogate |
|---|---|---|---|---|
| 2016 | 0.068 | 0.094 | 0.000 | 0.000 |
| 2017 | 0.069 | 0.093 | 0.000 | 0.000 |
| 2018 | 0.068 | 0.097 | 0.000 | 0.000 |
| 2019 | 0.068 | 0.093 | 0.000 | 0.000 |
| 2020 | 0.068 | 0.097 | 0.000 | 0.000 |
| 2021 | 0.070 | 0.096 | 0.000 | 0.000 |
| 2022 | 0.070 | 0.095 | 0.000 | 0.000 |

(Medians over tickers.)

**Checks:**
1. **Periodicity: FAIL.** Real tempo strength beats the clock-shuffled surrogate in **0%** of 208 ticker-years.
2. **Stability: FAIL.** Real drifts less in 0% of ticker-years. The median session-to-session change is 0 for both, so every comparison is a tie. With a discrete period, this metric was degenerate as specified.
3. **Beyond the clock: PASS.** Pooled val onset MSE is 0.7468 for the full listener vs 0.8792 for clock-only, a 15.1% gain. The full listener is better in **100%** of the 30 tickers.

**Reported only:**
- Median beat interval: 7 bars across tickers (range 5–8). The tracker mostly forces beats at the carried tempo, because the pulse is weak.
- Meter shares at beats (median): 2 → 0.40, 3 → 0.31, 4 → 0.29. This is close to uniform, i.e. no clear meter.
- Correlation of each ingredient with beat bars: |move| ≈ 0.18, volume burst ≈ 0.11, turns ≈ 0.08.

**Result: FAIL** (checks 1–2). The 5m branch does not apply, since beats are not ≤ 2 bars apart.

**What it means:**
- **No steady pulse beyond the clock.** Measured as a periodic autocorrelation peak at 2–24 bars, the real market has **less** pulse than a fake market that keeps only the trading-day schedule. Whatever periodicity the tracker finds is clock-locked. Volatility clustering in real data smooths the autocorrelation and hides even those clock peaks.
- **The listener does hear structure beyond the clock** (check 3, every ticker). But this check rewards any predictability of upcoming activity, including plain volatility clustering. It does not show a beat.
- In musical terms: on 15m bars, the market's "time" looks like the schedule of the trading day plus rubato bursts of activity, not a steady tempo.

Per the spec, the build stops here for Andy's decision. There is no automatic fallback.
