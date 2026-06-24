# config.py
from pathlib import Path
ROOT = Path(__file__).resolve().parent

# ----------------------- Data -----------------------------------------------------
SEASONS = list(range(2016, 2024))  # 8 years

# ----------------------- Feature Selection ----------------------------------------
MISSING_THRESHOLD = 0.05      # Stage 2: max. allowed missing rate per column
ASSOCIATION_THRESHOLD = 0.05  # Stage 3: min. |r_pb| or Cramér's V to retain a feature

# final after VIF and treshold 0.05; 7 features
# treshold 0.05; 8 features
# treshold 0.03; 13 features
# treshold 0.00; 20 features
FEATURE_SETS = {
    "minimalistic_": ["shotgun", "down", "score_differential", "ydstogo", "posteam_timeouts_remaining", "quarter_seconds_remaining", "no_huddle"],
    #"minimalistic_005": [
    #    "wp",
    #    "shotgun", "down", "score_differential", "ydstogo", "posteam_timeouts_remaining", "quarter_seconds_remaining", "no_huddle"
    #],
    #"minimalistic_003": [
    #    "goal_to_go", "game_seconds_remaining", "defteam_timeouts_remaining", "yardline_100", "total_line",
    #    "wp",
    #    "shotgun", "down", "score_differential", "ydstogo", "posteam_timeouts_remaining", "quarter_seconds_remaining", "no_huddle"
    #],
    "significant": [
        "down", "ydstogo", "yardline_100", "goal_to_go", "shotgun", "no_huddle",
        "score_differential", "defteam_score", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        "game_seconds_remaining", "two_minute_drill", "qtr", "ep", "wp", "total_line",
        "season_type", "drive_start_transition", "roof"
    ]
}

FEATURE_CONFIG = {
    "numeric": ["wp", "score_differential", "ydstogo", "yardline_100", "quarter_seconds_remaining", "game_seconds_remaining", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "total_line", "spread_line"],
    "ordinal": {"down": [1, 2, 3, 4]},
    "binary": ["shotgun", "no_huddle", "goal_to_go", "div_game"],
    "nominal": ["surface", "roof", "location"]
}


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

