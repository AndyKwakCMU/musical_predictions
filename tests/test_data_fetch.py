import importlib.util
import sys
from pathlib import Path

import pandas as pd

from tests.test_data import daily_frame

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data_fetch.py"


def load_script():
    spec = importlib.util.spec_from_file_location("data_fetch", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_writes_protocol_csv_to_default_data_dir(tmp_path, monkeypatch):
    mod = load_script()
    monkeypatch.setattr(mod, "fetch_data", lambda *a: daily_frame())
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(
        sys, "argv", ["data_fetch.py", "AAPL", "2024-01-01", "2024-01-05", "1d"]
    )

    mod.main()

    out = tmp_path / "data" / "AAPL_2024-01-01_2024-01-05_1d.csv"
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "date", "time", "open", "high", "low", "close", "volume"
    ]
    assert len(df) == 2


def test_cli_ib_source_routes_to_ibkr_fetcher(tmp_path, monkeypatch):
    mod = load_script()
    protocol = pd.DataFrame({
        "date": ["2026-10-12"], "time": ["09:30:00"], "open": [1.0],
        "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1],
    })
    calls = []
    monkeypatch.setattr(mod, "fetch_data", lambda *a: (_ for _ in ()).throw(AssertionError("yfinance used")))
    monkeypatch.setattr(mod, "fetch_ib", lambda *a, **k: calls.append(a) or protocol)
    monkeypatch.setattr(sys, "argv", [
        "data_fetch.py", "AAPL", "2026-10-01", "2026-10-13", "5m", str(tmp_path),
        "--source", "ib",
    ])

    mod.main()

    assert calls and calls[0][0] == "AAPL"
    out = tmp_path / "AAPL_2026-10-01_2026-10-13_5m.csv"
    assert pd.read_csv(out)["time"].tolist() == ["09:30:00"]
