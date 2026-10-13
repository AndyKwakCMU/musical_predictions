##############################################################################

#           data_fetcher
#           Andy Kwak

##############################################################################

#       Takes data from yfinance, convert it into my own CSV protocol
#       for easy data management.


#       @requires proper args when ran
#       @ensures proper CSV data in data directory

##############################################################################

import argparse
import yfinance as yf
from pathlib import Path

from musical.data import save_bars, to_protocol

def fetch_data(ticker, start, end, interval):
    #@requires the params are all str
    #@ensures \result type is DataFrame
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    return data

def csv_save(t, s, e, i, data, directory):
    #@requires data type is DataFrame from fetch_data
    #@requires directory type is str
    #@ensures protocol CSV named {t}_{s}_{e}_{i}.csv in directory

    out_path = Path(directory) / f"{t}_{s}_{e}_{i}.csv"
    save_bars(to_protocol(data), out_path)
    return out_path

def main():
    par = argparse.ArgumentParser()
    par.add_argument("ticker", help="ticker we want the data of", type=str)
    par.add_argument("start", help="YYYY-MM-DD start date of data", type=str)
    par.add_argument("end", help="YYYY-MM-DD end date of data (exclusive)", type=str)
    par.add_argument("interval", help="time interval of data\n1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo", type=str)
    par.add_argument("out", help="directory to save the CSV data", nargs="?", default="data")

    args = par.parse_args()

    data = fetch_data(args.ticker, args.start, args.end, args.interval)
    if data.empty:
        raise SystemExit(f"no data returned for {args.ticker} {args.interval}")
    out_path = csv_save(args.ticker, args.start, args.end, args.interval, data, args.out)
    print(f"wrote {out_path}")



if __name__ == "__main__":
    main()

