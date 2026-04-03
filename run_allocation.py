"""
Trii Stocks Allocation Pipeline
================================
Runs the full pipeline:
  1. Stock pre-selection via technical signals (SMA, EMA, MACD)
  2. Future return estimation using a Transformer Neural Network
  3. Sharpe Ratio maximising allocation (with min/max weight constraints)
  4. CPPI strategy backtest

Outputs a CSV with the final portfolio weights and COP allocations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import risk_kit as rk
import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PERIODS_PER_YEAR   = 54          # 54 for weekly data
INTERVAL           = '1wk'
DAYS_OF_DATA       = 365 * 10
MA_TERMS           = 10          # Moving-average window (weeks)
PERIODS_TO_FORECAST = 4          # Periods ahead predicted by the Transformer
TIME_WINDOW        = 54          # Transformer input sequence length (weeks)
RF_RATE            = 0.11        # Risk-free rate (10-Y Colombian bond yield ≈ 11.2 %)
MAX_WEIGHT         = 0.15        # Max portfolio weight per stock
MIN_WEIGHT         = 0.05        # Min portfolio weight per stock
INVESTMENT_COP     = 105_000_000 # Total capital available (COP)
DRAWDOWN_FLOOR     = 0.20        # CPPI max drawdown floor
CPPI_MULTIPLIER    = 5           # CPPI cushion multiplier
N_TRANSFORMER_RUNS = 10         # Independent training runs; predictions are averaged
OUTPUT_PATH        = os.path.join(os.path.dirname(__file__), 'results', 'allocation_output.csv')

BASE_DIR           = os.path.dirname(__file__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DOWNLOAD & PRE-PROCESS STOCK DATA
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Step 1: Downloading stock data ===")

col_stock_list_path    = os.path.join(BASE_DIR, 'stock_tickers', 'colombia_stocks_trii.csv')
global_stock_list_path = os.path.join(BASE_DIR, 'stock_tickers', 'global_stocks_trii.csv')

ticker_list_col  = list(pd.read_csv(col_stock_list_path,    header=None)[0])
ticker_list_glob = list(pd.read_csv(global_stock_list_path, header=None)[0])

end_date   = datetime.date.today()
start_date = end_date - datetime.timedelta(days=DAYS_OF_DATA)

col_stocks_raw    = yf.download(ticker_list_col,  interval=INTERVAL, start=start_date, end=end_date, auto_adjust=True)['Close']
global_stocks_raw = yf.download(ticker_list_glob, interval=INTERVAL, start=start_date, end=end_date, auto_adjust=True)['Close']

col_stocks_raw.index    = col_stocks_raw.index.to_period(freq='W')
global_stocks_raw.index = global_stocks_raw.index.to_period(freq='W')


def combine_duplicate_rows(df):
    """Keep first non-null value when YFinance returns duplicate rows for the latest week."""
    def first_non_null(series):
        non_null = series.dropna()
        return non_null.iloc[0] if len(non_null) > 0 else np.nan
    return df.groupby(df.index).agg(first_non_null)


col_stocks_raw    = combine_duplicate_rows(col_stocks_raw)
global_stocks_raw = combine_duplicate_rows(global_stocks_raw)

stocks = pd.concat([col_stocks_raw, global_stocks_raw], axis='columns').sort_index()

# Drop tickers with more than 15 % missing data
stocks_not_missing = stocks.columns[stocks.isna().sum() < stocks.shape[0] * 0.15]
stocks = stocks[stocks_not_missing].bfill()

rets = stocks.pct_change().iloc[1:]

# Trim to the analysis window
analysis_end   = str(datetime.date.today())
analysis_start = str(datetime.date.today() - datetime.timedelta(days=DAYS_OF_DATA))
rets   = rets.loc[analysis_start:analysis_end]
stocks = stocks.loc[analysis_start:analysis_end]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TECHNICAL SIGNAL FILTERING
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Step 2: Computing technical signals ===")

signals = []
for ticker in stocks.columns:
    sig = rk.technical_indicators(
        stocks[ticker],
        indicators=['SMA', 'EMA', 'MACD'],
        time_window=MA_TERMS,
        macd_params=[12, 26, 9],
        return_df=True,
        plot=False,
        signal_tolerance=0.975
    ).iloc[-1]
    sig_df = pd.DataFrame(sig).T
    sig_df.index = [ticker]
    sig_df.rename(columns={ticker: 'Price'}, inplace=True)
    signals.append(sig_df)

signals = pd.concat(signals, axis=0)

# Keep stocks where at least 2 out of 3 signals are positive
signals_filtered = signals[
    np.int64(signals['MACD Signal']) +
    np.int64(signals[f'SMA{MA_TERMS} Signal']) +
    np.int64(signals[f'EMA{MA_TERMS} Signal']) >= 2
]

selected_stocks_rets   = rets[signals_filtered.index]
selected_stocks_stocks = stocks[signals_filtered.index]

# Convert PeriodIndex to end-of-week date strings
selected_stocks_rets.index   = selected_stocks_rets.index.astype('str').str.split('/').str[1]
selected_stocks_stocks.index = selected_stocks_stocks.index.astype('str').str.split('/').str[1]
rets.index   = rets.index.astype('str').str.split('/').str[1]
stocks.index = stocks.index.astype('str').str.split('/').str[1]

selected_stocks = selected_stocks_rets.columns
print(f"Selected {len(selected_stocks)} stocks: {list(selected_stocks)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRANSFORMER NEURAL NETWORK: FUTURE RETURN ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Step 3: Training Transformer Neural Network ===")

data = rets.values  # shape: (n_periods, n_stocks)


def create_dataset(data, time_window=1):
    X, Y = [], []
    for i in range(len(data) - time_window):
        X.append(data[i:(i + time_window)])
        Y.append(data[i + time_window])
    return np.array(X), np.array(Y)


X, Y = create_dataset(data, TIME_WINDOW)
print(f"Training shapes — X: {X.shape}, Y: {Y.shape}")


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model

    def call(self, inputs):
        positions = np.arange(self.sequence_length)[:, np.newaxis]
        num_dims  = (self.d_model + 1) // 2
        div_term  = np.exp(np.arange(0, num_dims) * (-np.log(10000.0) / self.d_model))
        pos_enc   = np.zeros((self.sequence_length, self.d_model))
        pos_enc[:, 0::2] = np.sin(positions * div_term)
        pos_enc[:, 1::2] = np.cos(positions * div_term[:self.d_model // 2])
        return inputs + tf.convert_to_tensor(pos_enc, dtype=tf.float32)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size, dropout=dropout
    )(inputs, inputs)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = tf.keras.layers.Dense(inputs.shape[-1])(ff)
    x  = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x


def build_transformer_model(input_shape, head_size=128, num_heads=8,
                             ff_dim=512, num_blocks=6, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = PositionalEncoding(sequence_length=input_shape[0], d_model=input_shape[1])(inputs)
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(input_shape[1])(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError]
    )
    return model


# Prepare prediction input once — constant across runs
data_preds = np.concatenate((data, np.expand_dims(np.zeros_like(data[-1]), axis=0)))
X_pred, _  = create_dataset(data_preds, TIME_WINDOW)

# Train N_TRANSFORMER_RUNS independent models and average their predictions
all_preds_runs = []
for run in range(N_TRANSFORMER_RUNS):
    print(f"  Training run {run + 1}/{N_TRANSFORMER_RUNS}...")
    model = build_transformer_model(input_shape=(TIME_WINDOW, X.shape[2]))
    model.fit(X, Y, epochs=50, batch_size=32, verbose=0)

    run_preds   = []
    pred_inputs = np.expand_dims(X_pred[-1], axis=0)
    for _ in range(PERIODS_TO_FORECAST):
        pred        = model.predict(pred_inputs, verbose=0)
        run_preds.append(pred[0])
        pred_inputs = np.concatenate((pred_inputs[0][1:], np.expand_dims(pred[0], axis=0)))
        pred_inputs = np.expand_dims(pred_inputs, axis=0)

    all_preds_runs.append(np.array(run_preds))

preds = np.mean(all_preds_runs, axis=0)  # shape: (PERIODS_TO_FORECAST, n_stocks)
print(f"Predictions averaged across {N_TRANSFORMER_RUNS} runs.")

last_date    = pd.to_datetime(rets.index).max()
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=7),
    periods=PERIODS_TO_FORECAST, freq='W-SUN'
).to_period('W')
preds_df = pd.DataFrame(preds, columns=rets.columns, index=future_dates)

# Forecasted price at end of horizon: compound the predicted returns on the last known price
current_prices    = stocks.iloc[-1]                          # last available price per stock
forecasted_prices = current_prices * (preds_df + 1).prod()  # cumulative product over forecast horizon

# Annualise with exponential decay weighting (most recent period weighted highest)
weight_indexes        = np.arange(1, PERIODS_TO_FORECAST + 1)
lambda_               = 0.2
exponential_decay_w   = np.exp(-lambda_ * weight_indexes) / np.exp(-lambda_ * weight_indexes).sum()

expected_annualized_rets = {
    symbol: (1 + np.sum(preds_df[symbol] * exponential_decay_w)) ** PERIODS_PER_YEAR - 1
    for symbol in preds_df.columns
}
expected_returns = pd.Series(expected_annualized_rets)

# Ledoit-Wolf shrinkage covariance (3 years of weekly data)
# Analytically optimal shrinkage estimator — reduces estimation error vs. raw
# sample covariance, especially effective when n_stocks > n_observations.
from sklearn.covariance import LedoitWolf
ret_sample = rets[selected_stocks].iloc[-PERIODS_PER_YEAR * 3:]
covmat = pd.DataFrame(
    LedoitWolf().fit(ret_sample).covariance_,
    index=selected_stocks, columns=selected_stocks
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SHARPE RATIO MAXIMISING ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Step 4: Sharpe Ratio Maximising Allocation ===")

returns = expected_returns[selected_stocks]

initial_weights = rk.msr_tuned(
    riskfree_rate=RF_RATE, returns=returns, covmat=covmat,
    max_weight=MAX_WEIGHT, periods_per_year=PERIODS_PER_YEAR
)
optimal_allocation = (
    pd.DataFrame(initial_weights, index=returns.index, columns=['Weights'])
    .sort_values(by='Weights')
)

# Iteratively remove the lowest-weight stock until all weights meet the minimum
while optimal_allocation['Weights'].min() < MIN_WEIGHT and len(optimal_allocation) > 2:
    optimal_allocation = optimal_allocation.iloc[1:]
    w = rk.msr_tuned(
        riskfree_rate=RF_RATE,
        returns=returns[optimal_allocation.index],
        covmat=covmat.loc[optimal_allocation.index, optimal_allocation.index],
        max_weight=MAX_WEIGHT,
        periods_per_year=PERIODS_PER_YEAR
    )
    optimal_allocation = (
        pd.DataFrame(w, index=optimal_allocation.index, columns=['Weights'])
        .sort_values(by='Weights')
    )

chosen_allocation = optimal_allocation
print(f"Final portfolio: {len(chosen_allocation)} stocks")
print(chosen_allocation.sort_values('Weights', ascending=False).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — BUILD ALLOCATED INDEX
# ─────────────────────────────────────────────────────────────────────────────

weights_series = chosen_allocation['Weights']

allocated_index = (selected_stocks_rets[weights_series.index] + 1).cumprod() * weights_series
allocated_index['index'] = allocated_index.sum(axis='columns')
allocated_index.index = pd.to_datetime(allocated_index.index.str.split('/').str[0])

# Prepend a starting row where the index equals 1
first_period    = allocated_index.index[0]
starting_period = first_period - pd.offsets.Week(weekday=6)
start_row       = pd.DataFrame(index=[starting_period], columns=allocated_index.columns)
start_row.iloc[:, :-1] = 1 * weights_series
start_row['index']     = 1
allocated_index = pd.concat([start_row, allocated_index])

allocated_index_returns = (allocated_index / allocated_index.shift(1) - 1).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — CPPI STRATEGY BACKTEST & FINAL ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Step 5: Running CPPI Strategy ===")

btr = rk.run_cppi(
    allocated_index_returns,
    m=CPPI_MULTIPLIER,
    start=1000,
    riskfree_rate=RF_RATE,
    drawdown=DRAWDOWN_FLOOR
)

risky_pct = btr['Risky Allocation']['index'].iloc[-1]
safe_pct  = 1 - risky_pct

risky_cop_per_stock = risky_pct * INVESTMENT_COP * weights_series
safe_cop_total      = safe_pct  * INVESTMENT_COP


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CSV
# ─────────────────────────────────────────────────────────────────────────────

output = pd.DataFrame({
    'Portfolio Weight':            weights_series.round(4),
    'Expected Annual Return':      expected_returns[weights_series.index].round(4),
    'Current Price':               current_prices[weights_series.index].round(4),
    f'Forecasted Price ({future_dates[-1]})': forecasted_prices[weights_series.index].round(4),
    'CPPI Risky Allocation (%)':   round(risky_pct, 4),
    'Risky Investment (COP k)':    (risky_cop_per_stock / 1_000).round(2),
})

safe_row = pd.DataFrame({
    'Portfolio Weight':          [None],
    'Expected Annual Return':    [None],
    'Current Price':             [None],
    f'Forecasted Price ({future_dates[-1]})': [None],
    'CPPI Risky Allocation (%)': [round(safe_pct, 4)],
    'Risky Investment (COP k)':  [round(safe_cop_total / 1_000, 2)],
}, index=['Safe Assets'])

output = pd.concat([output.sort_values('Portfolio Weight', ascending=False), safe_row])

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
output.to_csv(OUTPUT_PATH)

print(f"\nAllocation saved to: {OUTPUT_PATH}")
print(f"\n{'─'*70}")
print(output.to_string())
print(f"{'─'*70}")
print(f"\nTotal CPPI risky allocation:  {risky_pct*100:.1f}%  →  COP {risky_cop_per_stock.sum()/1e6:.2f}M")
print(f"Total CPPI safe  allocation:  {safe_pct*100:.1f}%  →  COP {safe_cop_total/1e6:.2f}M")
