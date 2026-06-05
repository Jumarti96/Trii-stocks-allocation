"""
Data-intake helpers for pipeline step 1: load tickers, download Close+Volume in parallel batches,
and prune the universe by a currency-robust relative-liquidity filter (with conservative
small/degenerate-group handling and a grouping-health diagnostic).

Lives in src/ (importable) so pipeline/01_download.py (digit-prefixed, not importable) stays a thin
orchestrator and the experiment/test suite can import these functions directly.

See docs/superpowers/specs/2026-06-05-step1-scaling-liquidity-filter-design.md.
"""

import glob as _glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def load_tickers(csv_glob):
    """Load and clean tickers/ISINs from every CSV matching csv_glob.

    Reads utf-8-sig (strips BOM), coerces to str, strips whitespace, drops empty and 'nan',
    de-duplicates. Returns a list of any size.
    """
    out = set()
    for path in _glob.glob(csv_glob):
        col = pd.read_csv(path, header=None, encoding="utf-8-sig")[0]
        for raw in col.astype(str).tolist():
            t = raw.strip()
            if t and t.lower() != "nan":
                out.add(t)
    return sorted(out)
