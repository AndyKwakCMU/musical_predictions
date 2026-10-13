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
