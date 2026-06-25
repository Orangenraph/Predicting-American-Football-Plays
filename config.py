# config.py
from pathlib import Path
ROOT = Path(__file__).resolve().parent

# ----------------------- Data -----------------------------------------------------
SEASONS = list(range(2016, 2024))  # 8 years

# ----------------------- Feature Selection ----------------------------------------
MISSING_THRESHOLD = 0.05      # Stage 2: max. allowed missing rate per column
ASSOCIATION_THRESHOLD = 0.05  # Stage 3: min. |r_pb| or Cramér's V to retain a feature

FEATURE_SETS = {
    #"mini": ["shotgun", "down", "score_differential", "ydstogo", "posteam_timeouts_remaining", "quarter_seconds_remaining", "no_huddle"],
    
    "comprehensive": [
        "down", "ydstogo", "yardline_100", "goal_to_go", "shotgun", "no_huddle",
        "score_differential", "defteam_score", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "season_type",
        "game_seconds_remaining", "half_seconds_remaining", "qtr", "ep", "wp", "total_line", "drive_start_transition", "roof"
    ],

    "maxi": [
        "down", "ydstogo", "yardline_100", "goal_to_go", "shotgun", "no_huddle",
        "score_differential", "posteam_score", "defteam_score",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining","game_seconds_remaining", "half_seconds_remaining",
        "quarter_seconds_remaining", "qtr","season_type","ep", "wp", "vegas_wp","td_prob", "fg_prob", "opp_fg_prob", 
        "opp_td_prob", "no_score_prob","total_line", "drive_play_count", "drive_first_downs", "drive_inside20", "ydsnet", 
        "drive_start_transition", "roof",  "surface", "location", "posteam_type", "season", "week",
        "posteam", "defteam", "total_home_epa", "total_away_epa","total_home_pass_epa", "total_away_pass_epa",
    ]
}

FEATURE_CONFIG = {
    "numeric": [
        "wp", "vegas_wp",
        "ep",
        "score_differential", "posteam_score", "defteam_score",
        "ydstogo", "yardline_100",
        "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        "total_line", "spread_line",
        "td_prob", "fg_prob", "opp_fg_prob", "opp_td_prob", "no_score_prob",
        "drive_play_count", "drive_first_downs", "ydsnet",
        "total_home_epa", "total_away_epa",
        "total_home_pass_epa", "total_away_pass_epa",
        "season", "week",
    ],
    "ordinal": {
        "down": [1, 2, 3, 4],
        "qtr": [1, 2, 3, 4, 5],
    },
    "binary": [
        "shotgun", "no_huddle", "goal_to_go", "div_game",
        "drive_inside20",
    ],
    "nominal": [
        "roof",                  # 'dome' / 'outdoors' / 'closed' / 'open'
        "surface",               # 'grass' / 'turf' / ...
        "drive_start_transition",# 'PUNT' / 'KICKOFF' / 'INT' / ...
        "posteam",               # 32 NFL teams
        "defteam",               # 32 NFL teams
        "season_type",           # 'REG' / 'POST'
        "posteam_type",          # 'home' / 'away'
        "location",              # 'Home' / 'Neutral'
    ],
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

