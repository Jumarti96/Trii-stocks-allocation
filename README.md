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

**→ [`docs/PARAMETERS.md`](docs/PARAMETERS.md) documents every parameter**: what it does, valid values, which step reads it, the cross-parameter constraints, and the handful of options that exist in code but not in the YAML. It is the single source of truth; this section is only a starting point.

The settings you are most likely to change:

| Parameter | Default | Change it when |
|---|---|---|
| `investment` | `120000000` | Your available capital changes. (Renamed from `investment_cop`, which now raises an error.) |
| `rf_rate` | `0.11` | The risk-free rate moves, **or you change `report_currency`** — it must be denominated in the same currency. |
| `allocation_method` | `parametric_michaud` | You want a different strategy. Eight are available, five of which ignore the forecast entirely — see the method table in the parameter reference. |
| `universe_topn` | `500` | Your catalogue is large and you want to change how many stocks the model forecasts. Do not exceed ~600. |
| `n_transformer_runs` | `150` | You want steadier forecasts (raise) or a faster run (lower). Linear in training time. |

Two settings deserve a warning before you touch them:

- **`days_of_data`** — a longer window silently deletes younger companies, because any name missing more than 15% of the window is dropped. At 20 years that removed 45 of the top 300, including Tesla and Meta.
- **`min_weight`** — it caps the book at `1/min_weight` positions no matter which strategy you pick, so at the default `0.05` no method can hold more than 20 names.

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

### Option 2 — Jupyter Notebooks (exploration and charts)

With your environment active, open Jupyter from the project root:

```bash
jupyter notebook notebooks/
```

Run the four notebooks in order:

| # | Notebook | What it does |
|---|---|---|
| 1 | `1. Trii Catalog Stock Pre-selection.ipynb` | Downloads prices, applies signal-based pre-selection |
| 2 | `2. Future returns and Covariance matrix estimation.ipynb` | Trains Transformer NN to forecast returns; estimates covariance |
| 3 | `3. Trii Catalog Sharpe-Ratio Maximizing Allocation.ipynb` | Maximises Sharpe ratio with weight constraints; plots efficient frontier |
| 4 | `4. Trii Catalog CPPI Strategy on Chosen Allocation with Brownian Motion Simulation.ipynb` | Backtests CPPI strategy; runs Brownian motion simulation |

Each notebook loads core parameters from `params.yaml` automatically.

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

## Transformer Model

The Transformer Neural Network (step 2) is multivariate: at each timestep it receives the full
return cross-section — all stocks simultaneously — projected to a shared d_model=128 embedding
via self-attention. This lets the model learn cross-stock relationships directly from the return
data without requiring explicit industry tags or factor labels, similar in spirit to a Vector
Autoregressive (VAR) model but with a nonlinear attention-based architecture.

Each rebalance, the model is trained from scratch `n_transformer_runs` times with different
random initialisations. The final forecast is the average across all runs, dampening
initialisation noise.

| Technique | Detail |
|---|---|
| **Per-stock Z-score normalisation** | Each stock's return series is normalised to zero mean, unit variance before training and denormalised after prediction. This prevents high-volatility stocks from dominating the MSE loss — especially important in large universes where return scale disparity between micro-caps and large-caps can reach 10:1. |
| **LR warmup + cosine decay** | Learning rate ramps from 10% → 100% of `lr=1e-4` over the first 5 epochs (warmup), then decays via cosine annealing to near-zero over the remaining 45. This avoids large noisy gradient steps during random initialisation and allows fine-tuning toward the end of training. |
| **Winsorisation** | Predictions are clipped to the 1st–99th percentile of historical returns before being passed to the optimiser, preventing extreme outlier forecasts from distorting the allocation. |

---

## Project Structure

```
Trii-stocks-allocation/
├── params.yaml                 # Single source of truth for all parameters
├── params_trii_10y.yaml.disabled  # Inert snapshot of the 10-year Trii setup
├── orchestrator.py             # Pipeline runner — run all steps or a subset
│
├── pipeline/                   # One script per pipeline step (thin; no logic)
│   ├── config.py               # Shared config loader (reads params.yaml)
│   ├── 01_download.py          # Download, activity filter, currency/FX, USD returns
│   ├── 02_predict.py           # Universe screen + transformer forecast + covariance
│   ├── 03_allocate.py          # Dispatch to the configured allocation method
│   └── 04_report.py            # Final report assembly, currency conversion
│
├── src/                        # Importable logic, unit-tested
│   ├── data_intake.py          # Download, cleaning, currency resolution, universe screen
│   ├── transformer_model.py    # Transformer + train_and_predict + capacity_report
│   ├── allocation.py           # The eight allocation methods behind allocation_method
│   ├── strategies.py           # Model-free weighting rules, shared with the backtester
│   ├── backtesting.py          # Walk-forward engine: schedule, paired stats, mechanics
│   └── risk_kit.py             # Financial stats, optimisation, simulation
│
├── tests/                      # pytest; run: .venv/Scripts/python.exe -m pytest tests/ -q
│
├── docs/
│   └── PARAMETERS.md           # Every parameter: what it does, what reads it
│
├── experiments/                # Untracked. Research harnesses; results are not
│                               # committed, findings go into docstrings + commit messages
├── data/                       # Untracked. Intermediate files between steps
├── results/                    # allocation_output.csv
├── stock_tickers/              # *.csv is the live catalogue; *.inactive is switched off
├── notebooks/                  # Original exploratory notebooks
└── experimental_notebooks/     # Alternative modelling experiments (not required)
```

> `data/`, `data_*_backup/`, `results/*` and `experiments/` are excluded from version
> control. Experiment results are deliberately not committed: a markdown report in the
> working tree looks current no matter how stale it is. When a run drives a decision the
> numbers go in the commit message that changes `params.yaml`, and the durable findings
> go in the docstring of the code they constrain.

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

4. **If you have an NVIDIA GPU — required.** Step 3 installs the CPU-only build of
   PyTorch, which cannot use your GPU at all. Replace it:
   ```bash
   pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
   ```
   Substitute the `cuXXX` matching the CUDA version `nvidia-smi` reports. Then verify:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   A version ending in `+cpu`, or `False`, means the GPU is still unusable — see
   [GPU Acceleration](#gpu-acceleration-strongly-recommended) above.

5. *(Optional)* To use DCC-GARCH covariance estimation:
   ```bash
   pip install arch
   ```

---

## Dependencies

Key packages: `pandas`, `numpy`, `scikit-learn`, `yfinance`, `torch`, `statsmodels`, `matplotlib`, `seaborn`, `ipywidgets`, `PyYAML`

See `requirements.txt` for the full list.
