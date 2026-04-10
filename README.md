# Trii Stocks Allocation: Portfolio Optimization Project

## Overview
This project implements a portfolio optimization pipeline for stocks available on the Trii platform. It downloads historical price data, filters stocks using technical signals, forecasts future returns using a Transformer Neural Network, estimates a covariance matrix using Ledoit-Wolf shrinkage, and finds the allocation that maximizes the Sharpe ratio. The full capital budget is deployed into the selected equity positions.

---

## GPU Acceleration (Strongly Recommended)

The Transformer Neural Network is the most computationally intensive step in the pipeline. **Running with a CUDA-compatible NVIDIA GPU is strongly recommended** — it can reduce training time by 10–30× compared to CPU.

The pipeline automatically detects and uses the GPU if available. When a GPU is present, it also enables:
- **TF32 precision** via Tensor Cores (free speed boost on RTX 30/40 series)
- **FP16 mixed-precision (AMP)** during training and inference for maximum throughput

### Install the GPU build of PyTorch

By default, `pip install torch` installs the CPU-only build. To enable GPU acceleration, visit the [PyTorch installation page](https://pytorch.org/get-started/locally/), select your OS, package manager, and CUDA version, and run the generated command. Example for CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> To find your CUDA version, run `nvidia-smi` and check the **CUDA Version** shown in the top-right corner.

### Verify GPU is in use

At runtime the pipeline prints a confirmation line:

```
[GPU CONFIG] GPU detected: <your GPU name>
[GPU CONFIG] Tensor Cores enabled (TF32 + AMP acceleration).
```

If you see `No GPU found. Running on CPU.` instead, the CPU-only PyTorch build is installed — follow the install step above.

---

## Running the Model

There are two ways to run the pipeline depending on how much control you need.

### Option 1 — Main program (recommended for regular use)

Runs the full pipeline end-to-end and saves a CSV with the final portfolio weights and COP allocations.

```bash
python run_allocation.py
```

Output is saved to `results/allocation_output.csv` and contains the following columns for each selected stock:

| Column | Description |
|---|---|
| `Portfolio Weight` | Optimal weight in the portfolio |
| `Expected Annual Return` | Annualised return predicted by the Transformer NN |
| `Current Price` | Last available market price |
| `Forecasted Price (date)` | Price projected by the model over the forecast horizon |
| `Investment (COP k)` | COP thousands allocated to this stock |

---

### Option 2 — Jupyter Notebooks (recommended when you want more control or want to see charts)

With your environment active, open Jupyter from the project root:

```bash
jupyter notebook Notebooks/
```

Run the four notebooks in order:

| # | Notebook | What it does |
|---|---|---|
| 1 | `1. Trii Catalog Stock Pre-selection.ipynb` | Downloads prices, computes SMA/EMA/MACD signals, filters stocks |
| 2 | `2. Future returns and Covariance matrix estimation.ipynb` | Trains Transformer NN to forecast returns; estimates covariance |
| 3 | `3. Trii Catalog Sharpe-Ratio Maximizing Allocation.ipynb` | Maximises Sharpe ratio with weight constraints; plots efficient frontier |
| 4 | `4. Trii Catalog CPPI Strategy on Chosen Allocation with Brownian Motion Simulation.ipynb` | Backtests CPPI strategy; runs Brownian motion simulation |

Each notebook saves intermediate CSV files to `temp_references/` which are picked up by the next notebook.

---

## Adjustable Parameters

### Main program (`run_allocation.py`)

All key parameters are defined at the top of the file under the `CONFIGURATION` block:

```python
PERIODS_PER_YEAR    = 54     # 54 = weekly, 252 = daily, 12 = monthly
INTERVAL            = '1wk'  # yfinance download interval: '1d', '1wk', '1mo'
DAYS_OF_DATA        = 3650   # Historical data window (in days)
MA_TERMS            = 10     # Moving-average window for technical signals (weeks)
PERIODS_TO_FORECAST = 4      # Weeks ahead predicted by the Transformer
TIME_WINDOW         = 54     # Transformer input sequence length (weeks)
N_TRANSFORMER_RUNS  = 10     # Independent training runs; predictions are averaged
RF_RATE             = 0.11   # Risk-free rate (e.g. 10-Y Colombian bond yield)
MAX_WEIGHT          = 0.15   # Maximum portfolio weight per stock
MIN_WEIGHT          = 0.05   # Minimum portfolio weight per stock
INVESTMENT_COP      = 105_000_000  # Total capital to allocate (COP)
```

The most commonly adjusted parameters before each run are:
- `INVESTMENT_COP` — update to your current available capital
- `RF_RATE` — update to the current 10-year Colombian bond yield
- `N_TRANSFORMER_RUNS` — increase for more stable predictions, decrease to run faster

### Notebooks

The same parameters appear at the top of each notebook as plain variables (e.g. `periods_per_year`, `rf_rate`, `MA_terms`). Edit them directly in the relevant cell before running.

---

## Covariance Estimation Methods

Notebook 2 and the main program both offer two methods for estimating the covariance matrix. **Ledoit-Wolf is enabled by default.**

| Method | Status | Description |
|---|---|---|
| **Ledoit-Wolf Shrinkage** | **Enabled** | Analytically optimal shrinkage estimator. Significantly reduces estimation error compared to raw sample covariance, especially when the number of stocks exceeds the number of observations. |
| **DCC-GARCH** | Commented out | Captures time-varying volatility clustering and dynamic cross-asset correlations. To enable, install `arch` (`pip install arch`), comment out the Ledoit-Wolf cell, and uncomment the DCC-GARCH block. |

---

## Project Structure

```
Trii Stocks allocation/
├── run_allocation.py           # Main program — runs full pipeline, outputs CSV
├── results/
│   └── allocation_output.csv  # Output: final weights and COP allocations
├── Notebooks/
│   ├── 1. Trii Catalog Stock Pre-selection.ipynb
│   ├── 2. Future returns and Covariance matrix estimation.ipynb
│   ├── 3. Trii Catalog Sharpe-Ratio Maximizing Allocation.ipynb
│   └── 4. Trii Catalog CPPI Strategy on Chosen Allocation with Brownian Motion Simulation.ipynb
├── src/
│   └── risk_kit.py             # Core module: financial stats, optimization, simulation
├── stock_tickers/
│   ├── colombia_stocks_trii.csv
│   └── global_stocks_trii.csv
├── temp_references/            # Intermediate CSVs shared between notebooks
├── experimental_notebooks/     # Alternative modeling experiments (not required)
└── requirements.txt
```

---

## Setup

1. Clone the repository and navigate to the project folder:
   ```bash
   git clone https://github.com/Jumarti96/Trii-stocks-allocation.git
   cd "Trii Stocks allocation"
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create
   python -m venv venv

   # Activate — macOS/Linux
   source venv/bin/activate

   # Activate — Windows
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. *(Recommended)* Install the GPU build of PyTorch for faster training — see [GPU Acceleration](#gpu-acceleration-strongly-recommended) above.

5. *(Optional)* To use DCC-GARCH covariance estimation:
   ```bash
   pip install arch
   ```

---

## Dependencies

Key packages: `pandas`, `numpy`, `scikit-learn`, `yfinance`, `torch`, `statsmodels`, `matplotlib`, `seaborn`, `ipywidgets`

See `requirements.txt` for the full list.
