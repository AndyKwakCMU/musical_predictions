##############################################################################

#           data
#           Andy Kwak

##############################################################################

#       Single home for the project's raw-data CSV protocol:
#           date,time,open,high,low,close,volume
#       Converts yfinance output into the protocol, and reads/writes
#       protocol CSVs so the rest of the code never touches raw formats.


#       @requires DataFrames from yfinance (daily or intraday)
#       @ensures protocol-shaped DataFrames and CSVs

##############################################################################

import pandas as pd

PROTOCOL_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def to_protocol(data):
    #@requires data is a yfinance DataFrame indexed by Date or Datetime
    #@ensures \result has exactly PROTOCOL_COLUMNS, no NaN rows,
    #         times in the exchange's local time
    df = data.copy()

    # yfinance>=1.0 returns (Price, Ticker) MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]

    # intraday indexes are tz-aware in exchange time; keep that wall clock
    stamps = pd.DatetimeIndex(df.index)
    if stamps.tz is not None:
        stamps = stamps.tz_localize(None)

    df = df[PRICE_COLUMNS].copy()
    df.insert(0, "date", stamps.strftime("%Y-%m-%d"))
    df.insert(1, "time", stamps.strftime("%H:%M:%S"))
    df = df.dropna().reset_index(drop=True)
    df["volume"] = df["volume"].astype("int64")
    return df


def save_bars(df, path):
    #@requires df has exactly PROTOCOL_COLUMNS
    #@ensures CSV at path with the protocol header
    df[PROTOCOL_COLUMNS].to_csv(path, index=False)


def load_bars(path):
    #@requires path is a protocol CSV
    #@ensures \result has PROTOCOL_COLUMNS, indexed by bar timestamp
    df = pd.read_csv(path, dtype={"date": str, "time": str})
    if list(df.columns) != PROTOCOL_COLUMNS:
        raise ValueError(
            f"{path} is not a protocol CSV; got columns {list(df.columns)}"
        )
    df.index = pd.to_datetime(df["date"] + " " + df["time"])
    df.index.name = "timestamp"
    return df
