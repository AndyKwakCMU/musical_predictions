import datetime as dt

import pandas as pd
import pytest
from ib_async import BarData

from musical.broker import (
    PAPER_PORTS, bars_to_protocol, connect, fetch_history, to_ib_bar_size,
)

UTC = dt.timezone.utc


class FakeIB:
    def __init__(self, pages=None):
        self.connected_with = None
        self.pages = list(pages or [])
        self.requests = []

    def connect(self, host, port, clientId, readonly, timeout):
        self.connected_with = dict(host=host, port=port, clientId=clientId,
                                   readonly=readonly)

    def reqHistoricalData(self, contract, endDateTime, durationStr,
                          barSizeSetting, whatToShow, useRTH, formatDate):
        self.requests.append(endDateTime)
        return self.pages.pop(0) if self.pages else []


def bar(when, px=100.0):
    return BarData(date=when, open=px, high=px + 1, low=px - 1, close=px,
                   volume=10, average=px, barCount=1)


def test_connect_defaults_to_paper_and_read_only():
    ib = connect(ib_factory=FakeIB)
    assert ib.connected_with["port"] in PAPER_PORTS
    assert ib.connected_with["readonly"] is True


def test_connect_refuses_live_port_without_live_flag():
    with pytest.raises(ValueError):
        connect(port=4001, ib_factory=FakeIB)
    ib = connect(port=4001, live=True, ib_factory=FakeIB)
    assert ib.connected_with["port"] == 4001


def test_intraday_bars_convert_to_new_york_wall_clock():
    bars = [bar(dt.datetime(2026, 10, 12, 13, 30, tzinfo=UTC))]   # 09:30 EDT
    out = bars_to_protocol(bars)
    assert list(out.columns) == ["date", "time", "open", "high", "low", "close", "volume"]
    assert out.iloc[0]["date"] == "2026-10-12"
    assert out.iloc[0]["time"] == "09:30:00"


def test_daily_bars_get_midnight_time():
    out = bars_to_protocol([bar(dt.date(2026, 10, 12))])
    assert out.iloc[0]["time"] == "00:00:00"


def test_bar_size_mapping_from_yfinance_style_intervals():
    assert to_ib_bar_size("1m") == "1 min"
    assert to_ib_bar_size("5m") == "5 mins"
    assert to_ib_bar_size("1h") == "1 hour"
    assert to_ib_bar_size("1d") == "1 day"
    with pytest.raises(ValueError):
        to_ib_bar_size("7m")


def test_fetch_history_pages_backward_until_start_and_paces():
    t0 = dt.datetime(2026, 10, 9, 13, 30, tzinfo=UTC)
    later = [bar(t0 + dt.timedelta(minutes=5 * i)) for i in range(3, 6)]
    earlier = [bar(t0 + dt.timedelta(minutes=5 * i)) for i in range(0, 4)]  # overlaps
    ib = FakeIB(pages=[later, earlier])
    slept = []

    out = fetch_history(ib, contract=None, start=t0, end=t0 + dt.timedelta(hours=1),
                        bar_size="5 mins", chunk="1 D", pace_seconds=10,
                        sleep=slept.append)

    assert len(out) == 6                         # overlap de-duplicated
    assert out["time"].tolist()[0] == "09:30:00"
    assert out["time"].is_monotonic_increasing
    assert ib.requests[1] == later[0].date       # next page ends at earliest bar
    assert slept == [10]                         # paced between requests


def test_fetch_history_stops_on_empty_page():
    t0 = dt.datetime(2026, 10, 9, 13, 30, tzinfo=UTC)
    ib = FakeIB(pages=[[bar(t0 + dt.timedelta(days=1))], []])
    out = fetch_history(ib, contract=None, start=t0, end=t0 + dt.timedelta(days=2),
                        bar_size="5 mins", chunk="1 D", pace_seconds=0,
                        sleep=lambda s: None)
    assert len(out) == 1
    assert len(ib.requests) == 2
