"""
Trii Stocks Allocation Pipeline
================================
Runs the full pipeline:
  1. Stock pre-selection via technical signals (SMA, EMA, MACD)
  2. Future return estimation using a Transformer Neural Network
       - Trained N_TRANSFORMER_RUNS times; predictions are averaged to reduce
         the effect of random weight initialisation
  3. Covariance matrix estimation using Ledoit-Wolf shrinkage
       - Analytically optimal estimator for portfolios where n_stocks > n_observations
       - DCC-GARCH is available as an alternative (see comments in the code)
  4. Sharpe Ratio maximising allocation (with min/max weight constraints)
       - Iteratively removes the lowest-weight stock until all weights meet MIN_WEIGHT

Outputs a CSV with the final portfolio weights, forecasted prices, and COP allocations.
The full budget is deployed into the risky (equity) portfolio.
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH GPU & TENSOR CORES SETUP
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"\n[GPU CONFIG] GPU detected: {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision('high')
    print("[GPU CONFIG] Tensor Cores enabled (TF32 + AMP acceleration).\n")
else:
    print("\n[GPU CONFIG] No GPU found. Running on CPU.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PERIODS_PER_YEAR   = 54          # 54 for weekly data
INTERVAL           = '1wk'       # 1wk: 1 week, 1mth: 1 month; 1yr: 1 year
DAYS_OF_DATA       = 365 * 10
MA_TERMS           = 10          # Moving-average window (weeks)
PERIODS_TO_FORECAST = 4          # Periods ahead predicted by the Transformer
TIME_WINDOW        = 54          # Transformer input sequence length (weeks)
RF_RATE            = 0.11        # Risk-free rate (10-Y Colombian bond yield ≈ 11.2 %)
MAX_WEIGHT         = 0.15        # Max portfolio weight per stock
MIN_WEIGHT         = 0.05        # Min portfolio weight per stock
INVESTMENT_COP     = 105_000_000 # Total capital available (COP)
N_TRANSFORMER_RUNS = 50         # Independent training runs; predictions are averaged
OUTPUT_PATH        = os.path.join(os.path.dirname(__file__), 'results', 'allocation_output.csv')

BASE_DIR           = os.path.dirname(__file__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DOWNLOAD & PRE-PROCESS STOCK DATA
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Step 1: Downloading stock data ===")

col_stock_list_path    = os.path.join(BASE_DIR, 'stock_tickers', 'colombia_stocks_trii.csv')
global_stock_list_path = os.path.join(BASE_DIR, 'stock_tickers', 'global_stocks_trii.csv')

ticker_list_col  = list(pd.read_csv(col_stock_list_path,    header=None)[0].dropna().astype(str).str.strip())
ticker_list_glob = list(pd.read_csv(global_stock_list_path, header=None)[0].dropna().astype(str).str.strip())

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
        indicators=['SMA', 'EMA', 'MACD', 'PRC'],
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


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        positions = np.arange(sequence_length)[:, np.newaxis]
        num_dims  = (d_model + 1) // 2
        div_term  = np.exp(np.arange(0, num_dims) * (-np.log(10000.0) / d_model))
        pos_enc   = np.zeros((sequence_length, d_model))
        pos_enc[:, 0::2] = np.sin(positions * div_term)
        pos_enc[:, 1::2] = np.cos(positions * div_term[:d_model // 2])
        self.register_buffer('pos_enc', torch.tensor(pos_enc, dtype=torch.float32).unsqueeze(0))

    def forward(self, x):
        return x + self.pos_enc


class TransformerModel(nn.Module):
    def __init__(self, input_shape, head_size=128, num_heads=8, ff_dim=512, num_blocks=6, dropout=0.1):
        super().__init__()
        seq_len, num_features = input_shape
        # Proyectar variables al d_model múltiplo de attention heads
        d_model = 128
        
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.output_proj = nn.Linear(d_model, num_features)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_proj(x)
        return x


# Prepare prediction input once — constant across runs
data_preds = np.concatenate((data, np.expand_dims(np.zeros_like(data[-1]), axis=0)))
X_pred, _  = create_dataset(data_preds, TIME_WINDOW)

# PyTorch Training Preparation
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
dataset = TensorDataset(X_tensor, Y_tensor)

all_preds_runs = []
for run in range(N_TRANSFORMER_RUNS):
    print(f"  Training run {run + 1}/{N_TRANSFORMER_RUNS}...")
    model = TransformerModel(input_shape=(TIME_WINDOW, X.shape[2])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Objecto para escalar gradientes y usar Precisión Mixta en Tensor Cores
    scaler = torch.cuda.amp.GradScaler()
    
    model.train()
    for target_epoch in range(50):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            # Activar Precisión Mixta (FP16 para mayor velocidad)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(batch_x)
                loss = criterion(output, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    run_preds = []
    pred_inputs = torch.tensor(X_pred[-1], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(PERIODS_TO_FORECAST):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred = model(pred_inputs)
            run_preds.append(pred[0].cpu().numpy())
            pred_inputs = torch.cat((pred_inputs[:, 1:, :], pred.unsqueeze(1)), dim=1)

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
    max_weight=MAX_WEIGHT, periods_per_year=PERIODS_PER_YEAR,
    debug=False
)
optimal_allocation = (
    pd.DataFrame(initial_weights, index=returns.index, columns=['Weights'])
    .sort_values(by='Weights')
)

# Batch elimination: drop ALL stocks whose CUMULATIVE weight (ascending) is below MIN_WEIGHT.
# Stocks are sorted by weight ascending; we accumulate and cut where the running total < MIN_WEIGHT.
iteration = 0
while len(optimal_allocation) > 2:
    iteration += 1

    # Cumulative sum of weights (already sorted ascending)
    cum_weights = optimal_allocation['Weights'].cumsum()

    # A stock is dropped if its cumulative weight is still below the threshold
    failing_mask = cum_weights < MIN_WEIGHT

    if not failing_mask.any():
        break  # all cumulative weights meet the threshold — done

    n_dropped = failing_mask.sum()
    print(f"  [pass {iteration}] Dropping {n_dropped} stock(s) with cumulative weight < {MIN_WEIGHT:.0%}: "
          f"{list(optimal_allocation[failing_mask].index)}")

    # Drop the entire sub-threshold batch at once
    optimal_allocation = optimal_allocation[~failing_mask]
    if len(optimal_allocation) <= 2:
        break

    w = rk.msr_tuned(
        riskfree_rate=RF_RATE,
        returns=returns[optimal_allocation.index],
        covmat=covmat.loc[optimal_allocation.index, optimal_allocation.index],
        max_weight=MAX_WEIGHT,
        periods_per_year=PERIODS_PER_YEAR,
        debug=True
    )
    optimal_allocation = (
        pd.DataFrame(w, index=optimal_allocation.index, columns=['Weights'])
        .sort_values(by='Weights')
    )

chosen_allocation = optimal_allocation
print(f"Final portfolio: {len(chosen_allocation)} stocks")
print(chosen_allocation.sort_values('Weights', ascending=False).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CSV
# ─────────────────────────────────────────────────────────────────────────────

weights_series = chosen_allocation['Weights']
cop_per_stock  = (weights_series * INVESTMENT_COP / 1_000).round(2)

output = pd.DataFrame({
    'Portfolio Weight':                        weights_series.round(4),
    'Expected Annual Return':                  expected_returns[weights_series.index].round(4),
    'Current Price':                           current_prices[weights_series.index].round(4),
    f'Forecasted Price ({future_dates[-1]})':  forecasted_prices[weights_series.index].round(4),
    'Investment (COP k)':                      cop_per_stock,
}).sort_values('Portfolio Weight', ascending=False)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
output.to_csv(OUTPUT_PATH)

print(f"\nAllocation saved to: {OUTPUT_PATH}")
print(f"\n{'─'*70}")
print(output.to_string())
print(f"{'─'*70}")
print(f"\nTotal invested:  COP {cop_per_stock.sum()/1_000:.2f}M  across {len(weights_series)} stocks")
