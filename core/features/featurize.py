import pandas as pd


# ---------------------------
# Config-derived parameters
# ---------------------------
def get_label_start_dt(project_cfg) -> pd.Timestamp:
    val = (
        project_cfg.get("modeling", {})
                   .get("label_start_saturday_utc", "2021-01-02")
    )
    dt = pd.to_datetime(str(val), utc=True, errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime("2021-01-02", utc=True)
    return dt


# ---------------------------
# Row filters
# ---------------------------
def _ensure_week_saturday_dt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "week_saturday_dt" in out.columns:
        # Coerce to tz-aware datetime even if it’s currently strings/objects
        out["week_saturday_dt"] = pd.to_datetime(out["week_saturday_dt"], utc=True, errors="coerce")
        # If any remain NaT and we have the utc string, fill from it
        if "week_saturday_utc" in out.columns:
            mask = out["week_saturday_dt"].isna()
            if mask.any():
                out.loc[mask, "week_saturday_dt"] = pd.to_datetime(
                    out.loc[mask, "week_saturday_utc"], utc=True, errors="coerce"
                )
    elif "week_saturday_utc" in out.columns:
        out["week_saturday_dt"] = pd.to_datetime(out["week_saturday_utc"], utc=True, errors="coerce")
    else:
        raise KeyError("Expected 'week_saturday_dt' or 'week_saturday_utc' in weekly table")
    return out

def filter_label_period(df: pd.DataFrame, label_start_dt: pd.Timestamp) -> pd.DataFrame:
    out = _ensure_week_saturday_dt(df)
    return out[out["week_saturday_dt"] >= label_start_dt].copy()


# ---------------------------
# Column selection / hygiene
# ---------------------------
def select_feature_columns(df: pd.DataFrame) -> list[str]:
    core = [
        # base weekly counts
        "scrobbles_week",
        "unique_days_week",
        "scrobbles_last_fri_sat",
        "scrobbles_saturday",
        "last_scrobble_gap_days",
        # within-week competition
        "within_week_rank_by_scrobbles",
        # momentum
        "scrobbles_prev_1w",
        "scrobbles_prev_4w",
        "week_over_week_change",
        "momentum_4w_ratio",
        # history & novelty
        "prior_scrobbles_all_time",
        "prior_weeks_with_scrobbles",
        "weeks_since_first_scrobble",
        "first_seen_week",
        # release (days)
        "days_since_release",
    ]
    # be flexible: whichever released_within_*d the weekly builder produced
    rel_flags = [c for c in df.columns if c.startswith("released_within_") and c.endswith("d")]
    cols = [c for c in core + rel_flags if c in df.columns]
    return cols


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    leaky = {
        # "spotify_popularity",
        "artist_listeners", "artist_playcount",
        "album_listeners", "album_playcount",
    }
    keep = [c for c in df.columns if c not in leaky]
    return df[keep]


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    id_like = {
        "artist_name", "track_name",
        "week_saturday_utc", "week_saturday_dt",
        "track_mbid", "artist_mbid", "album_mbid",
        "track_key", "artist_key", "album_key",
        "spotify_track_id",
    }
    keep = [c for c in df.columns if c not in id_like]
    return df[keep]


def remove_high_corr_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit, fixed list of redundant features to drop due to high spearman correlation
    """
    to_drop = [
        "prior_weeks_with_scrobbles",
        "weeks_since_first_scrobble",
    ] # correlated with "prior_scrobbles_all_time"
    return X.drop(columns=[c for c in to_drop if c in X.columns], errors="ignore")

# ---------------------------
# Imputation
# ---------------------------
def impute_days_since_release(df: pd.DataFrame) -> pd.DataFrame:
    if "days_since_release" not in df.columns:
        return df
    out = df.copy()
    is_na = out["days_since_release"].isna()
    if is_na.any():
        med = out["days_since_release"].median()
        out["days_since_release_was_missing"] = is_na.astype(int)
        out["days_since_release"] = out["days_since_release"].fillna(med)
    else:
        out["days_since_release_was_missing"] = 0
    return out


# ---------------------------
# Assembly
# ---------------------------
def make_weekly_for_model(df_weekly: pd.DataFrame, label_start_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a modeling-view weekly table (ready for training notebooks).
    Steps:
      - Filter to label period
      - Impute days_since_release (+ flag)
      - Drop identifier/key columns
      - Drop known leaky columns
      - Drop explicit high-correlation features (fixed list)
    """
    filtered = filter_label_period(df_weekly, label_start_dt)
    imputed = impute_days_since_release(filtered)
    no_ids = drop_identifier_columns(imputed)
    no_leaks = drop_leaky_columns(no_ids)
    no_redundant = remove_high_corr_features(no_leaks)
    return no_redundant


def make_X_y(weekly_for_model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds X and y DataFrames from the weekly modeling view:
    - Selects core feature columns (+ any released_within_* flag present)
    - (IDs/leaky already removed upstream in make_weekly_for_model)
    """
    feat_cols = select_feature_columns(weekly_for_model)
    X = weekly_for_model[feat_cols]
    y = weekly_for_model[["is_week_favorite"]].copy()
    return X, y
