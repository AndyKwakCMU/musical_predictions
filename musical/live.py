##############################################################################

#           live
#           Andy Kwak

##############################################################################

#       Turns the most recent bar into a trading decision using exactly
#       the same TSA gate + CNN window code the backtest used, so live
#       (or dry-run) signals can be checked against the backtest.


#       @requires protocol bars ending at the bar we are deciding on
#       @ensures a signal dict for that bar

##############################################################################

import numpy as np

from musical.cnn import predict_proba
from musical.dataset import make_window
from musical.experiment import direction_from_proba
from musical.tsa import COMPRESSED, analyze


def latest_signal(bars, ticker, model, width):
    #@requires bars time-ordered, last row is the decision bar
    #@ensures dict with timestamp, ticker, state, atr, direction
    #         (+1 / -1, or 0 = no trade) and proba (None if no trade)
    tsa = analyze(bars)
    t = len(bars) - 1
    signal = {
        "timestamp": bars.index[t],
        "ticker": ticker,
        "state": tsa["state"].iloc[t],
        "close": float(bars["close"].iloc[t]),
        "atr": float(tsa["atr"].iloc[t]),
        "direction": 0,
        "proba": None,
    }
    if signal["state"] != COMPRESSED:
        return signal
    x = make_window(bars, tsa, t, width)
    if not np.isfinite(x).all():
        return signal
    proba = predict_proba(model, x[None], [ticker])[0]
    signal["proba"] = proba
    signal["direction"] = int(direction_from_proba(proba[None])[0])
    return signal


def drop_incomplete_daily(bars, now, close_hour=16):
    #@requires bars indexed by date (daily), now tz-aware
    #@ensures today's bar removed if the session hasn't closed yet,
    #         so we never decide on a still-forming candle
    local = now.tz_convert("America/New_York")
    today = local.normalize().tz_localize(None)
    if local.hour < close_hour and len(bars) and bars.index[-1] >= today:
        return bars[bars.index < today]
    return bars
