##############################################################################

#           paper_trade
#           Andy Kwak

##############################################################################

#       Daily signal runner (plan Part 3, step 3: DRY RUN ONLY).
#       Run once after the close. For each ticker it pulls recent daily
#       bars, runs the TSA gate + CNN exactly like the backtest, and
#       appends what it WOULD trade to a signal log. It never places
#       orders; order placement (step 4) is intentionally not written
#       until a model passes Test B/C and dry-run signals match the
#       backtest.
#
#       Kill switch: if a file named STOP exists in the repo root,
#       the script exits immediately.


#       @requires a model from model_train.py; IB Gateway for --source ib
#       @ensures rows appended to the signal log CSV

##############################################################################

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from musical import broker, config
from musical.cnn import load_model
from musical.data import load_bars, to_protocol
from musical.live import drop_incomplete_daily, latest_signal

HISTORY_BARS = 400      # > TSA lookback + ATR + CNN window


def read_universe(path):
    #@ensures list of tickers (comments and blank lines skipped)
    tickers = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        tickers += line.split()
    return tickers


def bars_from_protocol(df):
    #@ensures timestamp-indexed bars like load_bars produces
    df = df.copy()
    df.index = pd.to_datetime(df["date"] + " " + df["time"])
    return df


def history_yf(ticker):
    data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True,
                       progress=False)
    return bars_from_protocol(to_protocol(data))


def history_ib(ib, ticker):
    from ib_async import Stock
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)
    bars = ib.reqHistoricalData(contract, endDateTime="", durationStr="2 Y",
                                barSizeSetting="1 day", whatToShow="ADJUSTED_LAST",
                                useRTH=True, formatDate=1)
    return bars_from_protocol(broker.bars_to_protocol(bars))


def main():
    par = argparse.ArgumentParser()
    par.add_argument("model", help="path to a .pt from model_train.py")
    par.add_argument("--source", choices=["yf", "ib"], default="ib")
    par.add_argument("--port", type=int, default=4002)
    par.add_argument("--universe", default="data/universe.txt")
    par.add_argument("--log", default="data/signals.csv")
    par.add_argument("--width", type=int, default=config.WINDOW)
    args = par.parse_args()

    if Path("STOP").exists():
        raise SystemExit("STOP file present: kill switch engaged, exiting")

    model = load_model(args.model)
    ib = broker.connect(port=args.port, client_id=21) if args.source == "ib" else None
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    try:
        for ticker in read_universe(args.universe):
            bars = history_ib(ib, ticker) if ib else history_yf(ticker)
            bars = drop_incomplete_daily(bars, now).iloc[-HISTORY_BARS:]
            sig = latest_signal(bars, ticker, model, args.width)
            p = sig.pop("proba")
            sig.update(run_time=now.isoformat(timespec="seconds"),
                       source=args.source, model=Path(args.model).name,
                       p_down=None if p is None else round(float(p[0]), 4),
                       p_none=None if p is None else round(float(p[1]), 4),
                       p_up=None if p is None else round(float(p[2]), 4),
                       risk_per_share=round(config.BARRIER_K * sig["atr"], 4))
            rows.append(sig)
    finally:
        if ib:
            ib.disconnect()

    df = pd.DataFrame(rows)
    log = Path(args.log)
    df.to_csv(log, mode="a", header=not log.exists(), index=False)
    trades = df[df["direction"] != 0]
    print(f"{len(df)} tickers checked, {(df['state'] == 'COMPRESSED').sum()} compressed, "
          f"{len(trades)} would-trade signals -> {log}")
    if len(trades):
        print(trades[["timestamp", "ticker", "direction", "p_down", "p_up",
                      "close", "risk_per_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
