# Stock Portfolio Allocation: Portfolio Optimization Project

## Overview
This project implements a portfolio optimization pipeline for stocks available on the Trii platform. It downloads historical price data for a universe of ~2,700+ ISINs, applies an **activity filter** to remove genuinely inactive names (step 1), forecasts future returns using a Transformer Neural Network trained on the **full active universe** (step 2), then pre-selects the top-N candidates by Sharpe proxy and finds the allocation that maximises the Sharpe ratio over that set (step 3). The full capital budget is deployed into the selected equity positions.

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
[GPU] <your GPU name> detected.
[GPU] Tensor Cores enabled (TF32 + AMP acceleration).
```

If you see `No GPU found — running on CPU.` instead, the CPU-only PyTorch build is installed — follow the install step above.

---

## Configuration

All pipeline parameters live in **`params.yaml`** at the project root. Edit this file before running any option below — the orchestrator, individual pipeline scripts, and notebooks all read from it.

```yaml
# Timing & data
periods_per_year: 54          # 52 for weekly, 12 for monthly
interval: "1wk"              # "1wk" or "1mo"
days_of_data: 3650

# Activity filter (Step 1)
min_active_fraction: 0.90     # Keep stocks traded in >= 90% of recent weeks

# Transformer model
periods_to_forecast: 4
n_transformer_runs: 50        # Increase for stability; decrease to run faster

# Portfolio optimisation (Step 3)
allocation_top_n: 150         # Candidates fed to the optimizer; null = no cap
allocation_ranking: sharpe    # "sharpe" (mu/sigma) | "return" (mu only)
allocation_method: parametric_michaud
michaud_spread: 4.0
michaud_mc_draws: 1000
rf_rate: 0.11                 # 10-Y Colombian bond yield
max_weight: 0.15
min_weight: 0.05

# Output
investment_cop: 115000000     # Total capital (COP)
output_path: "results/allocation_output.csv"
```

The most commonly adjusted parameters before each run are:
- `investment_cop` — update to your current available capital
- `rf_rate` — update to the current 10-year Colombian bond yield
- `n_transformer_runs` — increase for more stable predictions; decrease to run faster
- `allocation_top_n` — increase if you have more compute budget (optimizer scales O(N³))

---

## Running the Model

There are three ways to run the pipeline.

---

### Option 1 — Modular Pipeline via Orchestrator (recommended)

The pipeline is broken into four sequential steps, each producing intermediate files in `data/` that can be inspected between runs.

```bash
python orchestrator.py
```

The orchestrator is **resumable**: use `--resume` to skip steps whose output files already exist, so you can re-run step 3 without repeating the GPU-intensive Transformer step.

**Useful flags:**

| Command | Effect |
|---|---|
| `python orchestrator.py` | Run all steps unconditionally |
| `python orchestrator.py --resume` | Run all steps, skip any already cached |
| `python orchestrator.py --steps 3` | Run only step 3 |
| `python orchestrator.py --from 2` | Run from step 2 to the end |
| `python orchestrator.py --steps 3 --resume` | Run step 3 only if not cached |
| `python orchestrator.py --list` | Show step status and exit |

**Running a single step standalone** (without the orchestrator):

```bash
python pipeline/03_allocate.py
```

Each script in `pipeline/` is fully self-contained and can be run independently, as long as its input files in `data/` already exist.

**Pipeline steps and intermediate files:**

| Step | Script | Outputs to `data/` |
|---|---|---|
| 1 | `01_download.py` | `01_prices.csv`, `01_returns.csv` |
| 2 | `02_predict.py` | `02_expected_returns.csv`, `02_covmat.csv`, `02_predictions.csv`, `02_metadata.json` |
| 3 | `03_allocate.py` | `03_weights.csv` |
| 4 | `04_report.py` | `results/allocation_output.csv` |

The Transformer model itself lives in `src/transformer_model.py`.

---

### Option 2 — Single-script (legacy)

Runs the full pipeline end-to-end in a single process. No intermediate files are written.

```bash
python run_allocation.py
```

> Parameters for this option are still hardcoded at the top of `run_allocation.py`. For new runs, Option 1 is recommended.

---

### Option 3 — Jupyter Notebooks (recommended for exploration and charts)

With your environment active, open Jupyter from the project root:

```bash
jupyter notebook Notebooks/
```

Run the four notebooks in order:

| # | Notebook | What it does |
|---|---|---|
| 1 | `1. Trii Catalog Stock Pre-selection.ipynb` | Downloads prices, applies signal-based pre-selection |
| 2 | `2. Future returns and Covariance matrix estimation.ipynb` | Trains Transformer NN to forecast returns; estimates covariance |
| 3 | `3. Trii Catalog Sharpe-Ratio Maximizing Allocation.ipynb` | Maximises Sharpe ratio with weight constraints; plots efficient frontier |
| 4 | `4. Trii Catalog CPPI Strategy on Chosen Allocation with Brownian Motion Simulation.ipynb` | Backtests CPPI strategy; runs Brownian motion simulation |

Each notebook loads core parameters from `params.yaml` automatically. Intermediate CSV files are saved to `temp_references/` and picked up by the next notebook.

> **Note:** `pipeline/` is the source of truth. The notebooks are kept for exploration and charts and may lag the pipeline in methodology.

---

## Output

`results/allocation_output.csv` contains the following columns for each selected stock:

| Column | Description |
|---|---|
| `Portfolio Weight` | Optimal weight in the portfolio |
| `Expected Annual Return` | Annualised return predicted by the Transformer NN |
| `Current Price` | Last available market price |
| `Forecasted Price (date)` | Price projected by the model over the forecast horizon |
| `Investment (COP k)` | COP thousands allocated to this stock |

A `PORTFOLIO INDEX` summary row is appended at the bottom with aggregate statistics.

---

## Covariance Estimation Methods

Notebook 2 and the pipeline both offer two methods for estimating the covariance matrix. **Ledoit-Wolf is enabled by default.**

| Method | Status | Description |
|---|---|---|
| **Ledoit-Wolf Shrinkage** | **Enabled** | Analytically optimal shrinkage estimator. Significantly reduces estimation error compared to raw sample covariance, especially when the number of stocks exceeds the number of observations. |
| **DCC-GARCH** | Commented out | Captures time-varying volatility clustering and dynamic cross-asset correlations. To enable, install `arch` (`pip install arch`), comment out the Ledoit-Wolf block, and uncomment the DCC-GARCH block. |

---

## Project Structure

```
Trii Stocks allocation/
├── params.yaml                 # Single source of truth for all parameters
├── orchestrator.py             # Pipeline runner — run all steps or a subset
├── run_allocation.py           # Legacy single-script pipeline
│
├── pipeline/                   # One script per pipeline step
│   ├── config.py               # Shared config loader (reads params.yaml)
│   ├── 01_download.py          # Download & preprocess stock data + activity filter
│   ├── 02_predict.py           # Transformer prediction + covariance (full universe)
│   ├── 03_allocate.py          # Top-N pre-selection + Sharpe ratio optimisation
│   └── 04_report.py            # Final report assembly
│
├── data/                       # Intermediate files between pipeline steps
│   ├── 01_prices.csv
│   ├── 01_returns.csv
│   ├── 02_expected_returns.csv
│   ├── 02_covmat.csv
│   ├── 02_predictions.csv
│   ├── 02_metadata.json
│   └── 03_weights.csv
│
├── results/
│   └── allocation_output.csv   # Final allocation output
│
├── Notebooks/
│   ├── 1. Trii Catalog Stock Pre-selection.ipynb
│   ├── 2. Future returns and Covariance matrix estimation.ipynb
│   ├── 3. Trii Catalog Sharpe-Ratio Maximizing Allocation.ipynb
│   └── 4. Trii Catalog CPPI Strategy on Chosen Allocation with Brownian Motion Simulation.ipynb
│
├── src/
│   ├── risk_kit.py             # Core module: financial stats, optimisation, simulation
│   └── transformer_model.py    # Transformer model + train_and_predict
│
├── stock_tickers/
│   ├── colombia_stocks_trii.csv
│   └── global_stocks_trii.csv
│
├── temp_references/            # Intermediate CSVs shared between notebooks
└── experimental_notebooks/     # Alternative modelling experiments (not required)
```

> `data/` and `temp_references/` are excluded from version control (`.gitignore`). Only `.gitkeep` files are tracked to preserve the directory structure.

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

Key packages: `pandas`, `numpy`, `scikit-learn`, `yfinance`, `torch`, `statsmodels`, `matplotlib`, `seaborn`, `ipywidgets`, `PyYAML`

See `requirements.txt` for the full list.
