##############################################################################

#           labels
#           Andy Kwak

##############################################################################

#       Triple-barrier labels: how did the "cadence" resolve?
#       For an event at bar t we enter at bar t+1's open, put barriers at
#       entry +/- k * ATR(t), and watch bars t+1..t+horizon.
#           UP    upper barrier touched first
#           DOWN  lower barrier touched first
#           NONE  neither touched, or both inside one bar (order unknown)
#       This is exactly the trade the backtester simulates.


#       @requires protocol bars and an ATR series on the same index
#       @ensures one labeled row per event with a full horizon ahead

##############################################################################

import numpy as np
import pandas as pd

UP = "UP"
DOWN = "DOWN"
NONE = "NONE"


def triple_barrier(bars, atr, k, horizon, events=None):
    #@requires k > 0, horizon >= 1, atr indexed like bars
    #@ensures \result indexed by event timestamp with columns
    #         label, bars_held, entry, upper, lower, exit_price,
    #         max_excursion_atr, fwd_ret
    opens = bars["open"].to_numpy(float)
    highs = bars["high"].to_numpy(float)
    lows = bars["low"].to_numpy(float)
    closes = bars["close"].to_numpy(float)
    atrs = atr.reindex(bars.index).to_numpy(float)
    n = len(bars)

    if events is None:
        positions = np.arange(n)
    else:
        positions = bars.index.get_indexer(events)
        positions = positions[positions >= 0]

    rows = []
    for t in positions:
        if t + horizon >= n or not np.isfinite(atrs[t]) or atrs[t] <= 0:
            continue
        entry = opens[t + 1]
        upper = entry + k * atrs[t]
        lower = entry - k * atrs[t]

        label, held, exit_price = NONE, horizon, closes[t + horizon]
        for j in range(1, horizon + 1):
            hit_up = highs[t + j] >= upper
            hit_down = lows[t + j] <= lower
            if hit_up or hit_down:
                held = j
                if hit_up and hit_down:
                    exit_price = closes[t + j]
                elif hit_up:
                    label, exit_price = UP, upper
                else:
                    label, exit_price = DOWN, lower
                break

        window = slice(t + 1, t + horizon + 1)
        excursion = max(highs[window].max() - entry, entry - lows[window].min())
        rows.append({
            "timestamp": bars.index[t],
            "label": label,
            "bars_held": held,
            "entry": entry,
            "upper": upper,
            "lower": lower,
            "exit_price": exit_price,
            "max_excursion_atr": excursion / atrs[t],
            "fwd_ret": closes[t + horizon] / entry - 1,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.set_index("timestamp")
