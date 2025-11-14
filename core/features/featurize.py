
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import mutual_info_classif


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
        # original metadata
        "spotify_popularity",
        "track_duration"
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
    genre_ohe = [c for c in df.columns if c.startswith("genre__")]
    cols = [c for c in core + rel_flags + genre_ohe if c in df.columns]
    return cols


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    leaky = {
        # "spotify_popularity",
        # "artist_listeners", "artist_playcount",
        # "album_listeners", "album_playcount",
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
    Explicit, fixed list of redundant features to drop due to high Spearman correlation.
    Kept simple and deterministic:
      - drops prior_weeks_with_scrobbles
      - drops weeks_since_first_scrobble
    both of which are strongly correlated with prior_scrobbles_all_time.
    """
    to_drop = [
        "prior_weeks_with_scrobbles",
        "weeks_since_first_scrobble",
    ]
    return X.drop(columns=[c for c in to_drop if c in X.columns], errors="ignore")

# ---------------------------
# One Hot Encoding
# ---------------------------
def _collapse_rare_levels(series: pd.Series, min_freq: int) -> pd.Series:
    s = series.fillna("unknown").astype(str)
    vc = s.value_counts(dropna=False)
    keep = set(vc[vc >= max(1, int(min_freq))].index.tolist())
    return s.where(s.isin(keep), other="other")

def fit_dv_ohe(
    df: pd.DataFrame,
    column: str,
    min_freq: int = 20,
    prefix: str | None = None,
    keep_original: bool = True,
) -> tuple[pd.DataFrame, DictVectorizer, list[str]]:
    """
    Fit + transform OHE for a single categorical column using DictVectorizer.
    - Collapses rare levels before fitting (>= min_freq kept, others -> 'other').
    - Builds columns as '{prefix}__{level}' where prefix defaults to the column name.
    - Returns (df_with_ohe, fitted_dv, feature_names).
    """
    out = df.copy()
    if column not in out.columns:
        return out, DictVectorizer(sparse=False), []

    pfx = prefix or column
    cats = _collapse_rare_levels(out[column], min_freq=min_freq)
    records = [{pfx: v} for v in cats]

    dv = DictVectorizer(sparse=False)
    mat = dv.fit_transform(records)
    levels = [f"{pfx}__{name.split('=')[-1]}" for name in dv.get_feature_names_out()]
    for j, name in enumerate(levels):
        out[name] = mat[:, j].astype(int)

    if not keep_original:
        out = out.drop(columns=[column])
    return out, dv, levels

def transform_dv_ohe(
    df: pd.DataFrame,
    dv: DictVectorizer,
    column: str,
    prefix: str | None = None,
    keep_original: bool = True,
) -> pd.DataFrame:
    """
    Transform-only with a previously fitted DictVectorizer for the same column/prefix.
    - Unseen levels map to all-zeros across DV columns.
    """
    out = df.copy()
    if column not in out.columns:
        return out

    pfx = prefix or column
    cats = out[column].fillna("unknown").astype(str)
    records = [{pfx: v} for v in cats]
    mat = dv.transform(records)
    levels = [f"{pfx}__{name.split('=')[-1]}" for name in dv.get_feature_names_out()]
    for j, name in enumerate(levels):
        out[name] = mat[:, j].astype(int)

    if not keep_original:
        out = out.drop(columns=[column])
    return out

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
      - Drop known leaky columns
      - Drop explicit high-correlation features (fixed list)

    Notes
    -----
    - Keeps identifier-like columns (artist_name, track_name, week_saturday_utc,
      week_saturday_dt, keys, IDs) so notebooks can:
        * perform time-based train/val/test splits
        * run audits and debugging on specific tracks/weeks
    - One-Hot Encoding is intentionally *not* applied here.
      OHE should be applied in the modeling code *after* the temporal split,
      fitting the DictVectorizer only on the training data.
    """
    filtered = filter_label_period(df_weekly, label_start_dt)
    imputed = impute_days_since_release(filtered)
    no_leaks = drop_leaky_columns(imputed)
    no_redundant = remove_high_corr_features(no_leaks)
    return no_redundant


def make_X_y(weekly_for_model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds X and y DataFrames from the weekly modeling view.

    - X: only feature columns returned by `select_feature_columns`
          (Core V1 features + any released_within_*d + genre__* if present).
    - y: single-column DataFrame with `is_week_favorite`.

    The input `weekly_for_model` is expected to still contain identifiers and
    week columns, but they are not included in X.
    """
    feat_cols = select_feature_columns(weekly_for_model)
    X = weekly_for_model[feat_cols]
    y = weekly_for_model[["is_week_favorite"]].copy()
    return X, y
