"""
Shared configuration loader for the stock allocation pipeline.

Every pipeline script imports from here:
    from config import load_config, PATHS, BASE_DIR, DATA_DIR
"""

import os
import yaml
import pandas as pd

# config.py lives in pipeline/; BASE_DIR is the project root one level up.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'data')

# Intermediate file paths
PATHS = {
    # Step 1 outputs
    '01_prices':           os.path.join(DATA_DIR, '01_prices.csv'),
    '01_returns':          os.path.join(DATA_DIR, '01_returns.csv'),
    # Step 2 outputs
    '02_selected_returns': os.path.join(DATA_DIR, '02_selected_returns.csv'),
    '02_selected_prices':  os.path.join(DATA_DIR, '02_selected_prices.csv'),
    '02_signals':          os.path.join(DATA_DIR, '02_signals.csv'),
    # Step 3 outputs
    '03_expected_returns': os.path.join(DATA_DIR, '03_expected_returns.csv'),
    '03_covmat':           os.path.join(DATA_DIR, '03_covmat.csv'),
    '03_predictions':      os.path.join(DATA_DIR, '03_predictions.csv'),
    '03_metadata':         os.path.join(DATA_DIR, '03_metadata.json'),
    # Step 4 outputs
    '04_weights':          os.path.join(DATA_DIR, '04_weights.csv'),
    # Step 5 output — set dynamically by load_config() from cfg['output_path']
    '05_report':           os.path.join(BASE_DIR, 'results', 'allocation_output.csv'),
}


def load_config(config_path=None):
    """Load params.yaml and return a config dict with derived values added."""
    if config_path is None:
        config_path = os.path.join(BASE_DIR, 'params.yaml')

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    interval = cfg['interval']
    cfg['period_freq']  = 'W'  if interval == '1wk' else 'M'
    cfg['date_offset']  = pd.Timedelta(days=7) if interval == '1wk' else pd.DateOffset(months=1)
    cfg['future_freq']  = 'W-SUN' if interval == '1wk' else 'MS'
    cfg['time_window']  = cfg['time_window'] or cfg['periods_per_year']

    PATHS['05_report'] = os.path.join(BASE_DIR, cfg['output_path'])

    return cfg
