# config.py
from pathlib import Path
ROOT = Path(__file__).resolve().parent

# ----------------------- Data -----------------------------------------------------
SEASONS = list(range(2016, 2024))  # 8 years

# ----------------------- Feature Selection ----------------------------------------
MISSING_THRESHOLD = 0.05      # Stage 2: max. allowed missing rate per column
ASSOCIATION_THRESHOLD = 0.05  # Stage 3: min. |r_pb| or Cramér's V to retain a feature

FINAL_FEATURES = ['shotgun', 'down','score_differential', 'ydstogo', 'posteam_timeouts_remaining', 'quarter_seconds_remaining', 'no_huddle']

# Treshold 0.05; 8 features
FEATURES_005 = ['wp',
                    'shotgun', 'down', 'score_differential', 'ydstogo', 'posteam_timeouts_remaining', 'quarter_seconds_remaining', 'no_huddle']

# Treshold 0.03; 13 features
FEATURES_003 = ['goal_to_go','game_seconds_remaining','defteam_timeouts_remaining','yardline_100','total_line',
                    'wp',
                        'shotgun', 'down', 'score_differential', 'ydstogo', 'posteam_timeouts_remaining', 'quarter_seconds_remaining', 'no_huddle']

# Treshold 1.00; 20 features
FEATURES_100 = ['surface','season','week','roof','location','div_game','spread_line',
                    'goal_to_go','game_seconds_remaining','defteam_timeouts_remaining','yardline_100','total_line',
                        'wp',
                            'shotgun', 'down', 'score_differential', 'ydstogo', 'posteam_timeouts_remaining', 'quarter_seconds_remaining', 'no_huddle']

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

