##############################################################################

#           broker
#           Andy Kwak

##############################################################################

#       Thin layer over ib_async (Interactive Brokers TWS API).
#       Safety first: paper ports and read-only by default; a live port
#       is refused unless the caller explicitly passes live=True.
#       Historical bars come back in the project CSV protocol so IBKR
#       data drops into the same pipeline as yfinance data.


#       @requires IB Gateway / TWS running with the socket API enabled
#       @ensures connected IB instances and protocol DataFrames

##############################################################################

import datetime as dt
import time

import pandas as pd

from musical.data import PROTOCOL_COLUMNS

PAPER_PORTS = {4002, 7497}     # IB Gateway paper, TWS paper
LIVE_PORTS = {4001, 7496}      # IB Gateway live, TWS live
EXCHANGE_TZ = "America/New_York"

BAR_SIZES = {
    "1m": "1 min", "2m": "2 mins", "5m": "5 mins", "15m": "15 mins",
    "30m": "30 mins", "1h": "1 hour", "60m": "1 hour", "1d": "1 day",
}
# How much history to ask for per request, sized to stay well under
# IBKR's per-request limits (measured 2026-10-13: 6 M of 5m bars ~13s,
# 2 Y of 15m bars ~20s on a paper account)
DEFAULT_CHUNK = {
    "1 min": "1 M", "2 mins": "2 M", "5 mins": "6 M", "15 mins": "2 Y",
    "30 mins": "2 Y", "1 hour": "2 Y", "1 day": "10 Y",
}


def _default_ib():
    from ib_async import IB
    return IB()


def connect(host="127.0.0.1", port=4002, client_id=1, readonly=True,
            live=False, timeout=10, ib_factory=_default_ib):
    #@requires port is a paper port, or live=True
    #@ensures \result is a connected IB instance
    if port not in PAPER_PORTS and not live:
        raise ValueError(f"port {port} is not a paper port; pass live=True "
                         f"only if you really mean to touch a live account")
    ib = ib_factory()
    # timeout bounds ib_async's startup sync (positions/account), which
    # can stall on some Gateway setups even though market data works
    ib.connect(host, port, clientId=client_id, readonly=readonly, timeout=timeout)
    return ib


def to_ib_bar_size(interval):
    #@requires interval in yfinance style (1m, 5m, 1h, 1d, ...)
    #@ensures \result is the matching IBKR barSizeSetting
    if interval not in BAR_SIZES:
        raise ValueError(f"unsupported interval {interval}; use one of {list(BAR_SIZES)}")
    return BAR_SIZES[interval]


def bars_to_protocol(bars):
    #@requires bars is a list of ib_async BarData (formatDate=2 for intraday)
    #@ensures \result has PROTOCOL_COLUMNS, intraday times in exchange time
    if not bars:
        return pd.DataFrame(columns=PROTOCOL_COLUMNS)
    df = pd.DataFrame([{
        "when": b.date, "open": b.open, "high": b.high, "low": b.low,
        "close": b.close, "volume": b.volume,
    } for b in bars])

    if isinstance(bars[0].date, dt.datetime):
        stamps = pd.DatetimeIndex(pd.to_datetime(df["when"], utc=True))
        stamps = stamps.tz_convert(EXCHANGE_TZ).tz_localize(None)
    else:
        stamps = pd.DatetimeIndex(pd.to_datetime(df["when"]))

    out = pd.DataFrame({
        "date": stamps.strftime("%Y-%m-%d"),
        "time": stamps.strftime("%H:%M:%S"),
        "open": df["open"], "high": df["high"], "low": df["low"],
        "close": df["close"], "volume": df["volume"].astype("int64"),
    })
    return out[PROTOCOL_COLUMNS]


def fetch_history(ib, contract, start, end, bar_size, chunk=None,
                  pace_seconds=10, what="TRADES", use_rth=True,
                  sleep=time.sleep):
    #@requires start < end (tz-aware datetimes), contract qualified
    #@ensures protocol DataFrame of bars in [start, end], oldest first,
    #         fetched newest-to-oldest one chunk per request.
    #         pace_seconds between requests keeps us under IBKR's pacing
    #         limit (~60 historical requests per 10 minutes).
    chunk = chunk or DEFAULT_CHUNK[bar_size]
    collected = []
    page_end = end
    first = True
    while True:
        if not first:
            sleep(pace_seconds)
        first = False
        bars = ib.reqHistoricalData(
            contract, endDateTime=page_end, durationStr=chunk,
            barSizeSetting=bar_size, whatToShow=what, useRTH=use_rth,
            formatDate=2,
        )
        if not bars:
            break
        collected = list(bars) + collected
        earliest = bars[0].date
        if _as_utc(earliest) <= _as_utc(start):
            break
        page_end = earliest

    out = bars_to_protocol(collected)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["date", "time"], keep="last")
    stamp = pd.to_datetime(out["date"] + " " + out["time"])
    lo = pd.Timestamp(_as_utc(start)).tz_convert(EXCHANGE_TZ).tz_localize(None)
    lo = lo.normalize() if bar_size == "1 day" else lo
    out = out[stamp >= lo]
    order = pd.to_datetime(out["date"] + " " + out["time"]).argsort()
    return out.iloc[order].reset_index(drop=True)


def _as_utc(when):
    #@ensures tz-aware UTC datetime (dates become midnight UTC)
    if isinstance(when, dt.datetime):
        return when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)
    return dt.datetime(when.year, when.month, when.day, tzinfo=dt.timezone.utc)
