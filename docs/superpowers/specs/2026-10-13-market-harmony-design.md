# Market Harmony: design

Written 2026-10-13. Status: **draft, awaiting Andy's approval.** Nothing in here is built yet.

## Intent and why

This project tests one idea from Andy's two blog posts (linked in `README.md`). The idea: a market can be *heard* the way a musician hears a piece. A trained listener recognizes a cadence before it resolves. They know both **that** the music will resolve and **where** it will land, because they have the context (meter, key, the progression so far) and know the composer's habits.

The first build reduced this to "volatility compression → breakout." That is loudness only, i.e. dynamics. It has no rhythm, no key and no harmony. Tests A–D (`docs/experiments.md`) show it predicts *that* a move comes but never *which way*. This design replaces that reduction with the actual musical model:

- **Rhythm.** The market has its own beats, tempo and meter. They are found in the data, not assumed from the bar size.
- **Harmony.** At every moment the market is playing some chord in some local key. The harmony is **jazz, not classical**:
  - key centers move fast;
  - applied (secondary, tertiary, …) dominants chain;
  - substitutions and deceptive resolutions are normal vocabulary.
- **Cadence.** A recognized progression (e.g. ii–V, or a chain V/V/V → V/V → V) tells us where it resolves, i.e. its local I. That gives both the move and its direction. Detours still count: if the resolution "trails off into another riff" but still reaches the I, the prediction was right.

Andy's framing of the parts: TSA is the fast math that runs first, and the CNN is the pattern recognizer that learns each ticker's "composer" style.

## Goals

1. **Rhythm.** Infer the market's beats, tempo and meter from the bar data, per ticker, causally (only past data at each point).
2. **Chord vocabulary.**
   - Learn a vocabulary of market "chords" from data.
   - **Categorize** them in musical terms: degrees I–VII relative to the local key center, plus applied chords of any depth (V/x, V/V/x, …), and substitutions and deceptive resolutions where the behavior fits.
   - Andy wants **both** the learned vocabulary and the musical categorization. The predictions come from the categorized progression.
3. **Prediction.** From the progression so far plus the ticker's habits, predict the next harmony and, when a cadence is forming, its destination (the local I) and therefore its direction.
4. **Trading.** Trade only recognized cadences: enter toward the predicted I, target the I, and invalidate on a key change. Paper trade through IBKR only after every gate below passes.

## Non-goals

- Classical-only harmony (a single home key with a fixed T–PD–D grammar).
- Hand-tuned indicator rules standing in for musical structure.
- Live-money trading in this phase.
- Order placement code before a model passes the holdout.

## Anti-goals (failure even if it "works")

- **Names without behavior.** Calling a cluster "V" must mean it behaves like a V: tension that resolves to a predictable I. Labels that look musical but predict nothing are a failure.
- **Collapsing back to volatility thresholds** or other indicator stacks.
- **Lookahead.** Beat tracking, key finding and HMM smoothing are usually done offline with future data. Every quantity used at time t must be computable from data ≤ t. Offline (two-sided) versions may be used for analysis plots only, never as inputs or labels' inputs.
- **Unfalsified structure.** Any rhythm or harmony we "find" must beat the same procedure run on shuffled or surrogate data.
- **Touching the 2024+ holdout** before a model has passed everything else.

## The model, layer by layer

### Layer 1: Rhythm (TSA): beats, tempo, meter

The 15-minute bars are raw samples, like audio samples, **not** beats. TSA "wraps over" them to find the beats, following the standard audio beat-tracking pipeline, run causally.

1. **Onset strength per bar.** How much "something new starts here." *Claude's call* for the starting ingredients:
   - volume surprise vs the same time of day (the seasonal profile already in `musical/tsa.py`);
   - absolute move relative to recent range;
   - price direction changes.

   Which ingredients carry the pulse is an empirical question; the builder may weight or learn them.
2. **Tempo.** Rolling periodicity of the onset envelope (tempogram / autocorrelation), smoothed so tempo can drift but not jump.
3. **Beat grid.** Place beats on strong onsets at roughly the current tempo. This is an online beat tracker that predicts the next beat from past beats, causal by construction.
4. **Meter.** Accent pattern across beats (which beats are stronger) gives downbeats and measure length. Meter may differ per ticker and over time.
5. **Beat-synchronous resampling.** Re-express each ticker's series per beat/measure, not per bar. Everything downstream works in beats and measures.

**The clock is something to learn, not to strip out (Andy's decision).** Timestamps go in as inputs: time of day, day of week, calendar position, and the session open/close. The model learns whatever the clock means for each ticker, the way a lifelong listener picks up nuances no textbook teaches (Andy's example: hearing that a violin recording is old from its tuning). The Rhythm gate still compares against a **clock-only baseline**, so we can see how much the model hears beyond the clock.

### The "listener": learning from lots of listening

Andy's principle: the model should listen like someone who grew up with this music. No one starts as a perfect listener, but long exposure teaches subtle cues. A trained model can get that exposure from data instead of 20 years.

*Claude's call* for how:
- Pretrain one shared encoder, the "listener", **self-supervised** on all the bar data from all tickers: predict what comes next, and fill in masked stretches. It uses no musical labels, so it learns from all the history.
- Feed it **as many metrics as we can**: OHLCV, timestamp features, the volume-at-price profile, the seasonal volume norm, TSA metrics, and market context (SPY/QQQ as "the rest of the orchestra").
- The musical layers become **heads** on that encoder: onsets/beats/meter, key list with confidences, chord vocabulary, cadence and destination. The TSA math stays as the cheap front end and as inputs. The CNN family (convolutional or a temporal-convolution encoder) stays the recognizer, per the README.

So the listener supplies intuition and the heads supply the musical names. Every head still has to pass its gate against surrogates and simple baselines.

### Layer 2: Harmony: key centers and chords

1. **A flexible list of keys, each with a confidence (Andy's decision).** There is no single strict key.
   - At every measure the system keeps a ranked list of candidate key centers, each with a confidence: a causal posterior over keys, like a Bayesian key-finder that keeps all hypotheses alive instead of committing.
   - Decisions use the most confident key, but the others stay in play.
   - Secondary keys that were only hinted at earlier can grow into the main key. This is how real key changes often arrive, through a nuanced pivot. A chord that is ii in one candidate key can be V/V in another; that dual reading is the pivot.
   - *Claude's call:* candidate key centers come from where volume trades (volume-at-price), the market analog of inferring key from how often each pitch sounds (Temperley). Each candidate's confidence comes from how well the recent harmony fits it.
   - A **tonicization** (brief visit) vs a **modulation** (home really moved) is a matter of how long and how strongly a candidate's confidence stays on top. A modulation is the analog of a reversal.
2. **Chord vocabulary (learned).** Cut the beat-synchronous series into measure-sized chunks and describe each relative to the local key center: where price sits, how it moves, volume and range character. Then learn a discrete vocabulary from data (e.g. encoder + clustering or vector quantization). The vocabulary size is chosen on validation data.
3. **Categorization (musical names).** Name each learned chord by how it **behaves**, using transition statistics in the train period:
   - **I (tonic):** stable; progressions end there.
   - **V (dominant):** high-probability resolution into a tonic chord.
   - **Predominants (ii, IV):** lead into V.
   - **Applied chords (V/x):** behave as a dominant whose resolution target is a non-tonic chord x. Chains (V/V/V → V/V → V → I) are applied chords resolving into applied chords.
   - **Deceptive resolutions and substitutions:** chords that occupy a dominant's or tonic's role but land somewhere else in a consistent way.
   - **Degrees I–VII:** also position relative to the key center.

   A chord is named **relative to each candidate key**, so it can carry several readings with the key's confidence attached. The working name is the reading under the most confident key. Where position and behavior disagree, behavior wins. The full jazz vocabulary is a hypothesis: a name the data doesn't support stays empty rather than being forced. Names must be consistent across retrains (see the Vocabulary gate).

### Layer 3: Phrase and cadence (CNN)

- **Input:** a window of the last N **measures** (not bars), with chord labels, key-center position, and beat-synchronous features as channels. Plus a per-ticker embedding: the composer.
- **Output:** a distribution over the next chord(s), and when a cadential progression is forming, P(cadence) and a distribution over its **destination key**.
- **Destination (Andy's decision: option c).** A cadence can always land in a new key.
  - The destination is expressed **relative** to the current key (e.g. "up a step", "to the relative key"), not as an absolute price.
  - The ticker's composer habits inform which key it tends to pick: the per-ticker embedding plus each ticker's history of key moves.
  - A plain V resolving home is just the case where the destination equals the current most-confident key.
  - Direction = which side of current price the predicted destination lies on; the target price is that key's center.
- **Cadence label:** did price reach the predicted destination key's center before an invalidation (the confident key list moving decisively elsewhere, or a protective stop) within a horizon measured in **measures**? Detours in between are allowed (evaded cadence / "one more time").

### Trading rule (once gates pass)

Signal when P(cadence) is high and the destination is clear:
- enter toward the I;
- target the I;
- stop on invalidation (modulation) or a protective price stop;
- time limit in measures.

Costs and fills use the existing pessimistic backtester, extended to support level targets and measure-based time stops.

## Gates (each must pass before the next layer is trusted)

All gates use only the dev period (< 2024-01-01). The settings for each gate are written into `docs/experiments.md` **before** running it, as in the Test A–D rounds.

1. **Rhythm gate.** The inferred pulse is real.
   - Tempo estimates are more stable, and beat-synchronous onsets more periodic, than the same procedure on surrogate data (block-shuffled bars / phase-randomized onset envelope).
   - It beats a **clock-only baseline** (time-of-day/calendar alone), so the rhythm is more than the trading-day schedule.
   - It holds across most tickers and years.
   - If this fails, stop and rethink before building harmony.
2. **Vocabulary gate.** The learned chords are a real vocabulary.
   - The chord sequence predicts the next chord better than a frequency baseline and better than the same pipeline on surrogates.
   - Chord identities and their musical names are reproducible across seeds/retrains.
3. **Harmony gate.** The names mean something.
   - Chords named V resolve into their I more often than matched non-dominant chords.
   - Applied chords resolve into their named targets more than chance.
   - Out of sample (val period).
4. **Cadence gate.** Recognized cadences predict direction on the test split, beating coin flip, always-long, momentum and a logistic baseline, in R terms and pre-cost.
5. **Backtest gate.** Walk-forward with costs (and 2× costs) beats always-long on the same signal times.
6. **Holdout.** One run on 2024+, then paper trading via IBKR (dry run first, as now).

## Data and splits

- **Data.** IBKR 15-minute RTH bars for the 30 names in `data/universe_intraday.txt`, 2016 → 2026-10, already downloaded (`data/raw_ib/`). If the Rhythm gate shows beats only 1–3 bars apart, 15m is too coarse: fetch 5-minute bars (IBKR has them back to at least 2010) and re-run the Rhythm gate before going on. That decision is made from the Rhythm gate's results.
- **Splits.** Unchanged from the intraday round: train 2016–2020, val 2021–2022, test 2023, holdout ≥ 2024-01-01. An embargo applies wherever a label horizon crosses a boundary. Note that 2021–2023 test-period reuse is already high from Tests B–D, so the holdout matters more than ever.
- **Survivorship.** The universe is today's large caps. This is a known bias that makes long results look better. It is flagged in every report and not fixed in this phase.

## What carries over, what is retired

- **Carries over:** CSV protocol and loaders, IBKR connection and history fetch, the backtester (extend it, don't replace it), splits with embargo, experiment logging discipline, test-first development, code conventions (CLAUDE.md).
- **Retired as the gate:** the volatility-compression gate. Its metrics may remain as a "dynamics" input channel. The old triple-barrier labels and Test A–D scripts stay in the repo as history.

## Decided by Andy

- Beats, tempo and meter are **inferred** by TSA + CNN from the bars; bars are not beats.
- Harmony is jazz: all degrees I–VII plus secondary, tertiary and deeper applied harmonies.
- The system learns a chord vocabulary **and** categorizes it musically; predictions come from that.
- Cadence = confident in both the move and the direction; detours that still reach the I count as correct.
- A cadence can always land in a new key, expressed relative to the current one; the ticker's habits inform which key.
- No single strict key: keep a flexible list of candidate keys with confidences and act on the most confident. Secondary keys hinted at earlier often become the new key.
- The model learns the clock from timestamps instead of having it stripped out, and trains on as many metrics as possible: listen like someone who grew up with the music.

## Claude's calls (Andy: check these)

- Onset ingredients: volume surprise, move vs range, direction changes.
- Candidate key centers come from volume-at-price; confidence comes from how well recent harmony fits each.
- A self-supervised "listener" encoder is pretrained on all bars and metrics; musical layers are heads on it.
- Market context (SPY/QQQ) is an input channel: the rest of the orchestra.
- Everything is causal (online beat tracking and filtering only).
- Chord names are assigned from behavior (transition statistics), not from hand rules alone.
- Gates run in order, each against shuffled/surrogate controls. At the first failure, stop and bring the results to Andy; no automatic fallback (e.g. no silently swapping in clock-time beats).
- Stay on 15m unless the Rhythm gate says it's too coarse.
- Beats and measures run continuously across sessions. The overnight gap is treated as a rest, not a reset, and the onset envelope still sees the open.
- Andy asked for plans covering the whole project (2026-10-13). Plans 2–4 are marked provisional: each starts by re-reading the previous gate's results and updating its defaults before any code. Every gate is still a stop-and-report checkpoint.

## Left to the builder

- Exact algorithms: tempogram method, beat tracker, meter detection, key filter, clustering/VQ method.
- Vocabulary size and window lengths, chosen on validation.
- Libraries (numpy/pandas/torch already present; librosa-style tools may be reimplemented or added).
- Surrogate construction details.

## References

- Andy's blog posts: see `README.md`.
- D. Ellis, "Beat Tracking by Dynamic Programming" (2007).
- D. Temperley, *Music and Probability* (2007): Bayesian key-finding and meter.
- W. Caplin, "The Classical Cadence" and "Beyond the Classical Cadence".
- J. Schmalfeldt, evaded cadence / "one more time" technique.
- D. Huron, *Sweet Anticipation* (2006): expectation.
