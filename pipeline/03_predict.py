"""
Step 3 - Transformer Prediction and Covariance Estimation

Trains the Transformer model N times (predictions are averaged to reduce the
effect of random weight initialisation), then:
  - Winsorises predictions at the 1st-99th percentile of historical returns
  - Annualises expected returns with exponential-decay weighting
  - Estimates the covariance matrix with Ledoit-Wolf shrinkage (full history)

Reads  (data/): 01_prices.csv, 01_returns.csv, 02_selected_returns.csv,
                02_selected_prices.csv
Outputs (data/):
    03_expected_returns.csv - annualised expected return per selected stock
    03_covmat.csv           - Ledoit-Wolf covariance matrix
    03_predictions.csv      - raw period-by-period predicted returns
    03_metadata.json        - current prices, forecasted prices, future dates,
                              winsorisation bounds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.covariance import LedoitWolf

from config import load_config, PATHS

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"\n[GPU] {torch.cuda.get_device_name(0)} detected.")
    torch.set_float32_matmul_precision('high')
    print("[GPU] Tensor Cores enabled (TF32 + AMP acceleration).\n")
else:
    print("\n[GPU] No GPU found - running on CPU.\n")


# Model definition

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
    def __init__(self, input_shape, num_heads=8, ff_dim=512, num_blocks=6, dropout=0.1):
        super().__init__()
        seq_len, num_features = input_shape
        d_model = 128
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True
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


def create_dataset(data, time_window):
    X, Y = [], []
    for i in range(len(data) - time_window):
        X.append(data[i:(i + time_window)])
        Y.append(data[i + time_window])
    return np.array(X), np.array(Y)


# Main

def main():
    cfg = load_config()

    print("\n=== Step 3: Training Transformer Neural Network ===")

    selected_rets   = pd.read_csv(PATHS['02_selected_returns'], index_col=0)
    selected_prices = pd.read_csv(PATHS['02_selected_prices'],  index_col=0)
    rets_full       = pd.read_csv(PATHS['01_returns'],          index_col=0)

    time_window         = cfg['time_window']
    periods_to_forecast = cfg['periods_to_forecast']
    n_runs              = cfg['n_transformer_runs']
    periods_per_year    = cfg['periods_per_year']

    data = selected_rets.values
    X, Y = create_dataset(data, time_window)
    print(f"Training shapes - X: {X.shape}, Y: {Y.shape}")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    dataset  = TensorDataset(X_tensor, Y_tensor)

    # Prediction input: append a dummy row so create_dataset captures the last window
    data_preds = np.concatenate((data, np.expand_dims(np.zeros_like(data[-1]), axis=0)))
    X_pred, _  = create_dataset(data_preds, time_window)

    all_preds_runs = []
    for run in range(n_runs):
        print(f"  Training run {run + 1}/{n_runs}...")
        model      = TransformerModel(input_shape=(time_window, X.shape[2])).to(device)
        optimizer  = optim.Adam(model.parameters(), lr=1e-4)
        criterion  = nn.MSELoss()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        scaler     = torch.cuda.amp.GradScaler()

        model.train()
        for _ in range(50):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = model(batch_x)
                    loss   = criterion(output, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        model.eval()
        run_preds   = []
        pred_inputs = torch.tensor(X_pred[-1], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            for _ in range(periods_to_forecast):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = model(pred_inputs)
                run_preds.append(pred[0].cpu().numpy())
                pred_inputs = torch.cat((pred_inputs[:, 1:, :], pred.unsqueeze(1)), dim=1)

        all_preds_runs.append(np.array(run_preds))

    preds = np.mean(all_preds_runs, axis=0)
    print(f"Predictions averaged across {n_runs} runs.")

    # Build future date range
    last_date    = pd.to_datetime(selected_rets.index).max()
    future_dates = pd.date_range(
        start=last_date + cfg['date_offset'],
        periods=periods_to_forecast,
        freq=cfg['future_freq']
    ).to_period(cfg['period_freq'])

    preds_df = pd.DataFrame(preds, columns=selected_rets.columns, index=future_dates)

    # Winsorise at 1st-99th percentile of historical returns
    lower_w = np.percentile(selected_rets.values, 1)
    upper_w = np.percentile(selected_rets.values, 99)
    preds_df = preds_df.clip(lower=lower_w, upper=upper_w)
    print(f"Predictions winsorised to [{lower_w:.4f}, {upper_w:.4f}].")

    # Annualised expected returns with exponential-decay weighting
    weight_indexes = np.arange(1, periods_to_forecast + 1)
    lambda_        = 0.2
    exp_decay_w    = np.exp(-lambda_ * weight_indexes) / np.exp(-lambda_ * weight_indexes).sum()

    expected_returns = pd.Series({
        sym: (1 + np.sum(preds_df[sym] * exp_decay_w)) ** periods_per_year - 1
        for sym in preds_df.columns
    })

    # Ledoit-Wolf covariance (full historical window)
    selected_cols = selected_rets.columns
    covmat = pd.DataFrame(
        LedoitWolf().fit(rets_full[selected_cols]).covariance_,
        index=selected_cols, columns=selected_cols
    )

    # Current and forecasted prices
    current_prices    = selected_prices.iloc[-1]
    forecasted_prices = current_prices * (preds_df + 1).prod()

    # Write outputs
    expected_returns.to_csv(PATHS['03_expected_returns'], header=['Expected Return'])
    covmat.to_csv(PATHS['03_covmat'])

    preds_out = preds_df.copy()
    preds_out.index = preds_out.index.astype('str')
    preds_out.to_csv(PATHS['03_predictions'])

    metadata = {
        'future_dates':        [str(d) for d in future_dates],
        'last_date':           str(last_date.date()),
        'winsorization_lower': float(lower_w),
        'winsorization_upper': float(upper_w),
        'current_prices':      current_prices.to_dict(),
        'forecasted_prices':   forecasted_prices.to_dict(),
    }
    with open(PATHS['03_metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {PATHS['03_expected_returns']}")
    print(f"       {PATHS['03_covmat']}")
    print(f"       {PATHS['03_predictions']}")
    print(f"       {PATHS['03_metadata']}")


if __name__ == '__main__':
    main()
