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
import pandas as pd
import csv
from pathlib import Path

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
    #@requires data type is DataFrame
    #@requires directory type is str

    out_path = Path(directory) / f"{t}_{s}_{e}_{i}.csv"
    
    df = data[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_csv(out_path, index=False)

def main():
    par = argparse.ArgumentParser()
    par.add_argument("ticker", help="ticker we want the data of", type=str)
    par.add_argument("start", help="YYYYMMDD start date of data", type=str)
    par.add_argument("end", help="YYYYMMDD end of date of data", type=str)
    par.add_argument("interval", help="time interval of data\n1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo", type=str)
    par.add_argument("out", help="directory to save the CSV data", default="data")
    
    args = par.parse_args()

    data = fetch_data(args.ticker, args.start, args.end, args.interval)
    data = data.reset_index()
    #print(data.columns.tolist())
    #print(type(data.index))
    #print(data.index[:5])
    csv_save(args.ticker, args.start, args.end, args.interval, data, args.out)



if __name__ == "__main__": 
    main()
    
