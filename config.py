# config.py
from pathlib import Path
ROOT = Path(__file__).resolve().parent

# ----------------------- Data -----------------------------------------------------
SEASONS = list(range(2016, 2024))  # 8 years

# ----------------------- Feature Selection ----------------------------------------
MISSING_THRESHOLD = 0.05      # Stage 2: max. allowed missing rate per column
ASSOCIATION_THRESHOLD = 0.05  # Stage 3: min. |r_pb| or Cramér's V to retain a feature

FEATURE_SETS = {
    "mini": ["shotgun", "down", "score_differential", "ydstogo", "posteam_timeouts_remaining", "quarter_seconds_remaining", "no_huddle"],
    
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

# ----------------------- Models ----------------------------------------

FNN_DEFAULTS = {
    "hidden_dims":   [128, 64, 32],
    "dropout":       0.3,
    "epochs":        150,        # war 100 — comprehensive braucht mehr Luft
    "batch_size":    512,
    "val_split":     0.1,
    "lr":            3e-4,
    "patience":      15,         # war 10 — passt zu mehr epochs
    "random_state":  42,
}

FNN_PARAMS = {
    "mini": dict(
        hidden_dims=[128, 64],   # war [64, 32] — etwas mehr Kapazität gegen Underfitting
        dropout=0.15,            # war 0.2 — weniger regularisierung bei kleinem Netz
        lr=5e-4,                 # schneller konvergieren bei wenig features
    ),
    "comprehensive": dict(
        hidden_dims=[128, 64, 32],  # defaults passen, explizit zur Klarheit
        lr=3e-4,
    ),
    "maxi": dict(
        hidden_dims=[128, 64, 32],  # war [256, 128, 64] — runterscalen gegen Overfitting
        dropout=0.5,                # war 0.4 — stärker regularisieren
        lr=1e-4,                    # langsamer, stabiler bei vielen features
        patience=20,                # mehr Geduld, der Fit ist noisier
    ),
}

RESFNN_DEFAULTS = {
    # Architecture
    "proj_dim":      128,
    "dropout":       0.4,
    # Training
    "epochs":        200,
    "batch_size":    512,
    "val_split":     0.1,
    "lr":            3e-4,
    "patience":      15,
    "weight_decay":  1e-2,
    "random_state":  42,
    # Scheduler
    "sched_factor":  0.5,
    "sched_patience": 8,
    "min_lr":        1e-6,
}

RESFNN_PARAMS = {
    "mini": dict(
        proj_dim=64,
    ),
    "comprehensive": dict(
        proj_dim=128,
        sched_patience=10
    ),
    "maxi": dict(
        proj_dim=128,
        dropout=0.5,
        weight_decay= 2e-2,
    ),
}


TABNET_DEFAULTS = {
    # Architecture
    "momentum":           0.02,
    "virtual_batch_size": 256,
    "att_dropout":        0.1,
    "final_dropout":      0.15,
    # Training
    "epochs":             300,
    "batch_size":         4096,
    "val_split":          0.1,
    "lr":                 1e-3,
    "patience":           20,
    "lambda_sparse":      1e-4,
    "weight_decay":       1e-4,
    "random_state":       42,
    # Optimizer / Scheduler
    "grad_clip":          1.0,
    "T_0":                75,
    "T_mult":             1,
    "eta_min":            1e-6,
}


TABNET_PARAMS = {
    "mini": dict(
        n_steps=4,
        n_d=32,
        n_a=32,
        gamma=1.0,
        lambda_sparse=1e-5,
        att_dropout=0,
        final_dropout=0,
        patience=20,    
        T_0=100,        
    ),

    "comprehensive": dict(
        n_steps=4,
        n_d=32,
        n_a=32,
        gamma=1.3,
        lambda_sparse=1e-4,
        att_dropout=0.1,
        final_dropout=0.1,
        patience=30,    
    ),

    "maxi": dict(
        n_steps=5,
        n_d=48,         
        n_a=48,         
        gamma=1.5,
        lambda_sparse=5e-4,
        att_dropout=0.15,
        final_dropout=0.2,
        weight_decay=2e-3,  
        patience=40,
    ),
}


# ----------------------- Paths -----------------------------------------------------
FIGURES_EDA = ROOT / "outputs" / "figures" / "eda"
FIGURES_MODELS = ROOT / "outputs" / "figures" / "models"
RESULTS_METRICS = ROOT / "outputs" / "results" / "metrics.csv"
CACHE_PATH = ROOT / "data" / "cache" / "pbp_raw.parquet"
RESULTS_METRICS = ROOT / "outputs" / "results" / "metrics.csv"

# ----------------------- Figs -----------------------------------------------------
PLOT_PALETTE = "hls"
PLOT_DPI = 120

