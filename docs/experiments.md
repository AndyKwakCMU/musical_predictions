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
