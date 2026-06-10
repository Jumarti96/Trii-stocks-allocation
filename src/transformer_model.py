"""
Transformer model for multivariate return forecasting.

Used by pipeline/02_predict.py. Lives in src/ (rather than next to the pipeline
step) because step scripts have digit-leading filenames that cannot be imported
directly.

Public API:
    train_and_predict(returns_df, cfg, n_runs=None) -> preds_df
        Trains the Transformer n_runs times, autoregressively forecasts
        cfg['periods_to_forecast'] periods ahead, averages across runs to damp
        random-initialisation noise, and winsorises predictions at the 1st-99th
        percentile of returns_df. Returns a DataFrame of predicted returns with the
        same columns as returns_df and a default RangeIndex (the caller assigns dates).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# GPU setup (shared by every caller)
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = torch.cuda.is_available()


def describe_device():
    """Return a one-line description of the compute device (callers may print it)."""
    if use_amp:
        torch.set_float32_matmul_precision('high')
        return f"[GPU] {torch.cuda.get_device_name(0)} detected. Tensor Cores enabled (TF32 + AMP)."
    return "[GPU] No GPU found - running on CPU."


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


def weighted_mean_return(preds_df, lambda_=0.2):
    """Exponential-decay-weighted mean of the per-period predicted returns, per column.

    This is the 'raw' forecast level *before* annualisation - nearer-term forecasts get more
    weight (decay rate lambda_). It is far less noise-amplified than the annualised figure,
    so it is the cleaner quantity for comparing forecast optimism across regimes.
    """
    periods        = preds_df.shape[0]
    weight_indexes = np.arange(1, periods + 1)
    exp_decay_w    = np.exp(-lambda_ * weight_indexes)
    exp_decay_w    = exp_decay_w / exp_decay_w.sum()
    return pd.Series({
        sym: float(np.sum(preds_df[sym].values * exp_decay_w))
        for sym in preds_df.columns
    })


def annualize_period_return(period_returns, periods_per_year):
    """Compound-annualise a per-period return (scalar or Series): (1+r)^ppy - 1.

    Display-only helper: used by the report to present per-period forecasts as
    annual figures. It is NOT used to build the vector fed to the optimiser.
    """
    return (1 + period_returns) ** periods_per_year - 1


def annualize_expected_returns(preds_df, periods_per_year, lambda_=0.2):
    """Exponential-decay-weighted annualised expected return per column of preds_df.

    Used by the train-universe comparison experiment and the report's display path.
    """
    wmr = weighted_mean_return(preds_df, lambda_=lambda_)
    return annualize_period_return(wmr, periods_per_year)


def train_runs(returns_df, cfg, n_runs=None, verbose=True):
    """Train n_runs Transformers and return the raw per-run forecasts.

    Returns an np.ndarray of shape (n_runs, periods_to_forecast, n_stocks) — no
    averaging, no winsorisation. train_and_predict composes averaging + winsorisation
    on top; the convergence sweep averages prefixes of these runs.
    """
    time_window         = cfg['time_window']
    periods_to_forecast = cfg['periods_to_forecast']
    n_epochs   = cfg.get('transformer_epochs', 50)
    n_warmup   = cfg.get('transformer_warmup_epochs', 5)
    lr         = cfg.get('transformer_lr', 1e-4)
    batch_size = cfg.get('transformer_batch_size', 32)
    if n_runs is None:
        n_runs = cfg['n_transformer_runs']

    data, mu, sigma = _normalise(returns_df)
    X, Y = create_dataset(data, time_window)
    if verbose:
        print(f"Training shapes - X: {X.shape}, Y: {Y.shape}")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    dataset  = TensorDataset(X_tensor, Y_tensor)

    # Prediction input: append a dummy row so create_dataset captures the last window
    data_preds = np.concatenate((data, np.expand_dims(np.zeros_like(data[-1]), axis=0)))
    X_pred, _  = create_dataset(data_preds, time_window)

    all_preds_runs = []
    for run in range(n_runs):
        if verbose:
            print(f"  Training run {run + 1}/{n_runs}...")
        model      = TransformerModel(input_shape=(time_window, X.shape[2])).to(device)
        optimizer  = optim.Adam(model.parameters(), lr=lr)
        criterion  = nn.MSELoss()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        scaler     = torch.cuda.amp.GradScaler() if use_amp else None

        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_warmup
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs - n_warmup, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[n_warmup]
        )

        model.train()
        for _ in range(n_epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        output = model(batch_x)
                        loss   = criterion(output, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(batch_x)
                    loss   = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
            scheduler.step()

        model.eval()
        run_preds   = []
        pred_inputs = torch.tensor(X_pred[-1], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            for _ in range(periods_to_forecast):
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        pred = model(pred_inputs)
                else:
                    pred = model(pred_inputs)
                run_preds.append(pred[0].cpu().numpy())
                pred_inputs = torch.cat((pred_inputs[:, 1:, :], pred.unsqueeze(1)), dim=1)

        all_preds_runs.append(_denormalise(np.array(run_preds), mu, sigma))

    return np.array(all_preds_runs)


def winsorize_to_history(preds_df, returns_df, lower_pct=1, upper_pct=99):
    """Clip forecasts to the [lower_pct, upper_pct] percentile range of historical returns."""
    lower_w = np.percentile(returns_df.values, lower_pct)
    upper_w = np.percentile(returns_df.values, upper_pct)
    return preds_df.clip(lower=lower_w, upper=upper_w)


def _normalise(returns_df):
    """Per-stock Z-score normalisation. Returns (data, mu, sigma).

    sigma is clipped to 1e-8 to prevent division by zero for dormant stocks.
    Both mu and sigma are 1-D ndarrays of shape (n_stocks,).
    """
    mu = returns_df.mean().values
    sigma = returns_df.std().clip(lower=1e-8).values
    data = (returns_df.values - mu) / sigma
    return data, mu, sigma


def _denormalise(preds_arr, mu, sigma):
    """Reverse per-stock Z-score normalisation.

    preds_arr: ndarray of shape (periods_to_forecast, n_stocks) in normalised space.
    Returns an ndarray of the same shape in original return scale.
    """
    return preds_arr * sigma + mu


def _normalise_crosssectional(returns_df):
    """Per-timestep cross-sectional Z-score normalisation.

    At each timestep, standardises across stocks (not across time per stock).
    Returns (data, mu_per_t, sigma_per_t) where mu and sigma have shape (n_periods,).
    At prediction time use the last timestep's stats for denormalisation.
    """
    data = returns_df.values.copy()
    mu_per_t = data.mean(axis=1)
    sigma_per_t = data.std(axis=1, ddof=1).clip(min=1e-8)
    data_norm = (data - mu_per_t[:, np.newaxis]) / sigma_per_t[:, np.newaxis]
    return data_norm, mu_per_t, sigma_per_t


def _denormalise_crosssectional(preds_arr, mu_last, sigma_last):
    """Reverse cross-sectional normalization using the last training timestep's stats.

    preds_arr: ndarray (n_steps, n_stocks) in normalized space.
    mu_last, sigma_last: scalars from the final training timestep.
    """
    return preds_arr * sigma_last + mu_last


def train_and_predict(returns_df, cfg, n_runs=None, verbose=True):
    """Train n_runs Transformers on returns_df and return averaged, winsorised forecasts.

    Pure with respect to the filesystem: no reads/writes, no date handling. The caller
    supplies returns_df (rows = periods, columns = stocks) and assigns dates to the result.
    """
    runs = train_runs(returns_df, cfg, n_runs=n_runs, verbose=verbose)
    if verbose:
        print(f"Predictions averaged across {runs.shape[0]} runs.")
    preds_df = pd.DataFrame(runs.mean(axis=0), columns=returns_df.columns)
    lower_pct = cfg.get('winsorization_lower_pct', 1)
    upper_pct = cfg.get('winsorization_upper_pct', 99)
    return winsorize_to_history(preds_df, returns_df, lower_pct, upper_pct)
