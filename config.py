# config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ----------------------- Data -----------------------------------------------------
SEASONS = list(range(2016, 2024))  # 8 years

# ----------------------- Paths -----------------------------------------------------
FIGURES_EDA = ROOT / "outputs" / "figures" / "eda"
FIGURES_MODELS = ROOT / "outputs" / "figures" / "models"
FIGURES_ERRORS = ROOT / "outputs" / "figures" / "errors"
RESULTS_METRICS = ROOT / "outputs" / "results" / "metrics.csv"
RESULTS_ERRORS = ROOT / "outputs" / "results" / "error_analysis.csv"
CACHE_PATH = ROOT / "data" / "cache" / "pbp_raw.parquet"

# ----------------------- Figs -----------------------------------------------------
PLOT_PALETTE = "hls"
PLOT_DPI = 120

