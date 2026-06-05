import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import data_intake as di


def test_load_tickers_hygiene(tmp_path):
    # one file with a BOM, whitespace, a nan, a blank line, and a duplicate
    f1 = tmp_path / "a.csv"
    f1.write_bytes("﻿AAPL\n MSFT \nnan\n\nAAPL\n".encode("utf-8"))
    f2 = tmp_path / "b.csv"
    f2.write_text("US1912161007\n")
    tickers = di.load_tickers(str(tmp_path / "*.csv"))
    assert set(tickers) == {"AAPL", "MSFT", "US1912161007"}   # BOM/space stripped, nan/blank/dup gone
