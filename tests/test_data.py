import pandas as pd
import pytest

from musical.data import PROTOCOL_COLUMNS, load_bars, save_bars, to_protocol


def yf_frame(index, ticker="AAPL"):
    # Mimics yfinance>=1.0 output: MultiIndex (Price, Ticker) columns
    n = len(index)
    cols = pd.MultiIndex.from_product(
        [["Close", "High", "Low", "Open", "Volume"], [ticker]],
        names=["Price", "Ticker"],
    )
    values = [[10.0 + i, 11.0 + i, 9.0 + i, 9.5 + i, 1000 + i] for i in range(n)]
    return pd.DataFrame(values, index=index, columns=cols)


def daily_frame():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    return yf_frame(idx)


def intraday_frame():
    idx = pd.DatetimeIndex(
        ["2026-10-12 09:30", "2026-10-12 09:35"], name="Datetime"
    ).tz_localize("America/New_York")
    return yf_frame(idx)


def test_protocol_columns_match_readme():
    assert PROTOCOL_COLUMNS == [
        "date", "time", "open", "high", "low", "close", "volume"
    ]


def test_daily_frame_converts_to_protocol_without_ticker_row():
    out = to_protocol(daily_frame())
    assert list(out.columns) == PROTOCOL_COLUMNS
    assert len(out) == 2
    assert out.iloc[0].to_dict() == {
        "date": "2024-01-02", "time": "00:00:00",
        "open": 9.5, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000,
    }


def test_intraday_frame_keeps_exchange_local_time():
    out = to_protocol(intraday_frame())
    assert out["date"].tolist() == ["2026-10-12", "2026-10-12"]
    assert out["time"].tolist() == ["09:30:00", "09:35:00"]


def test_rows_with_missing_prices_are_dropped():
    df = daily_frame()
    df.iloc[1, 0] = float("nan")
    assert len(to_protocol(df)) == 1


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "bars.csv"
    save_bars(to_protocol(intraday_frame()), path)

    first_line = path.read_text().splitlines()[0]
    assert first_line == "date,time,open,high,low,close,volume"

    bars = load_bars(path)
    assert list(bars.columns) == PROTOCOL_COLUMNS
    assert bars.index[0] == pd.Timestamp("2026-10-12 09:30")
    assert bars["close"].tolist() == [10.0, 11.0]


def test_load_rejects_non_protocol_csv(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Date,Open,High,Low,Close,Volume\n,AAPL,AAPL,AAPL,AAPL,AAPL\n")
    with pytest.raises(ValueError):
        load_bars(path)
