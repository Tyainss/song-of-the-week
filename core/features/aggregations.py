import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================
# Weekly base (from row-level scrobbles)
# ===========================
def build_weekly_base(df):
    """
    Aggregate row-level scrobbles to (artist_name, track_name, week_saturday_utc).
    Produces the base weekly table used by all feature helpers.
    """
    df = df.copy()

    ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
    wk = pd.to_datetime(df["week_saturday_utc"], utc=True, errors="coerce")
    df["__ts"] = ts
    df["__wk"] = wk
    df["__dow"] = df["__ts"].dt.weekday  # Mon=0 ... Sat=5, Sun=6
    df["__day"] = df["__ts"].dt.floor("D")
    df["__wk_end"] = df["__wk"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    group_keys = ["artist_name", "track_name", "week_saturday_utc"]

    agg_week = (
        df.groupby(group_keys, sort=False)
          .agg(
              scrobbles_week=("__ts", "size"),
              unique_days_week=("__day", pd.Series.nunique),
              scrobbles_last_fri_sat=("__dow", lambda s: int(((s == 4) | (s == 5)).sum())),
              scrobbles_saturday=("__dow", lambda s: int((s == 5).sum())),
          )
          .reset_index()
    )

    # last scrobble within week → gap to Saturday 23:59:59
    last_ts = df.groupby(group_keys, sort=False)["__ts"].max().rename("__last_ts").reset_index()
    wk_end = df[group_keys + ["__wk_end"]].drop_duplicates()
    agg_week = agg_week.merge(last_ts, on=group_keys, how="left").merge(wk_end, on=group_keys, how="left")
    agg_week["last_scrobble_gap_days"] = (agg_week["__wk_end"] - agg_week["__last_ts"]).dt.total_seconds() / 86400.0

    # label (if present)
    if "is_week_favorite" in df.columns:
        lbl = (
            df[group_keys + ["is_week_favorite"]]
            .dropna(subset=["is_week_favorite"])
            .groupby(group_keys, sort=False)["is_week_favorite"]
            .max()
            .astype(int)
            .reset_index()
        )
        agg_week = agg_week.merge(lbl, on=group_keys, how="left")
    else:
        agg_week["is_week_favorite"] = 0
    agg_week["is_week_favorite"] = agg_week["is_week_favorite"].fillna(0).astype(int)

    requested_carry = [
        # IDs / keys (for EDA & traceability; will be dropped for modeling)
        "track_mbid", "artist_mbid", "album_mbid",
        "track_key", "artist_key", "album_key",
        "spotify_track_id",
        # static facts / genres / popularity
        "track_duration",
        "spotify_release_date", "release_date_granularity",
        "spotify_genres", "genre_bucket", "genre_missing",
        "spotify_popularity",
        # listeners / playcounts (potentially leaky; kept for EDA, dropped for model)
        "artist_listeners", "artist_playcount",
        "album_listeners", "album_playcount",
        # data-quality flags
        "date_was_missing", "added_at_utc_was_missing",
        "week_saturday_utc_was_missing", "spotify_release_date_was_missing",
        "artist_listeners_was_missing", "artist_playcount_was_missing",
        "album_listeners_was_missing", "album_playcount_was_missing",
        "spotify_popularity_was_missing",
    ]
    carry_cols = [c for c in requested_carry if c in df.columns]
    if carry_cols:
        df_sorted = df.sort_values(["artist_name", "track_name", "__wk", "__ts"])
        meta = (
            df_sorted[group_keys + carry_cols + ["__ts"]]
            .sort_values(["artist_name", "track_name", "week_saturday_utc", "__ts"], ascending=True)
            .drop_duplicates(subset=group_keys, keep="last")
            .drop(columns=["__ts"])
        )
        agg_week = agg_week.merge(meta, on=group_keys, how="left")

    agg_week["week_saturday_dt"] = pd.to_datetime(agg_week["week_saturday_utc"], utc=True, errors="coerce")

    ordered = [
        "artist_name",
        "track_name",
        "week_saturday_utc",
        "week_saturday_dt",
        "scrobbles_week",
        "unique_days_week",
        "scrobbles_last_fri_sat",
        "scrobbles_saturday",
        "last_scrobble_gap_days",
        "is_week_favorite",
        *carry_cols,
    ]
    return (
        agg_week[ordered]
        .sort_values(["week_saturday_dt", "artist_name", "track_name"], kind="stable")
        .reset_index(drop=True)
    )


# ===========================
# Single-column helpers
# (each helper adds exactly one column)
# ===========================

# ---- Within-week competition ----
def add_within_week_rank_by_scrobbles(df):
    df = df.copy()
    df["within_week_rank_by_scrobbles"] = (
        df.groupby("week_saturday_utc")["scrobbles_week"]
          .rank(method="dense", ascending=False)
          .astype("Int64")
    )
    return df


# ---- Momentum (previous windows) ----
def add_scrobbles_prev_w(df, window: int, col_name: str = None):
    """
    Adds scrobbles_prev_{w}w computed as the sum of the previous `window` weeks
    (strict look-back). For window=1, this is just the previous week's value.
    """
    df = df.copy().sort_values(["artist_name", "track_name", "week_saturday_dt"], kind="stable")
    g = df.groupby(["artist_name", "track_name"], sort=False, group_keys=False)
    if window <= 0:
        raise ValueError("window must be >= 1")
    # shift(1) to exclude current week, then rolling sum over `window`
    series = g["scrobbles_week"].shift(1)
    if window == 1:
        prev = series.fillna(0).astype("Int64")
    else:
        prev = series.rolling(window=window, min_periods=1).sum().fillna(0).astype("Int64")
    name = col_name if col_name else f"scrobbles_prev_{window}w"
    df[name] = prev
    return df


def add_week_over_week_change(df):
    df = df.copy()
    if "scrobbles_prev_1w" not in df.columns:
        raise ValueError("scrobbles_prev_1w must exist before calling add_week_over_week_change()")
    df["week_over_week_change"] = (df["scrobbles_week"] - df["scrobbles_prev_1w"]).astype("Int64")
    return df


def add_momentum_ratio(df, window: int, col_name: str = None):
    """
    Adds momentum_{window}w_ratio = scrobbles_week / (1 + scrobbles_prev_{window}w).
    Requires scrobbles_prev_{window}w to exist.
    """
    df = df.copy()
    prev_col = f"scrobbles_prev_{window}w"
    if prev_col not in df.columns:
        raise ValueError(f"{prev_col} must exist before calling add_momentum_ratio(window={window})")
    name = col_name if col_name else f"momentum_{window}w_ratio"
    df[name] = df["scrobbles_week"] / (1.0 + df[prev_col].astype(float))
    return df


# ---- History & novelty ----
def add_prior_scrobbles_all_time(df):
    df = df.copy().sort_values(["artist_name", "track_name", "week_saturday_dt"], kind="stable")
    g = df.groupby(["artist_name", "track_name"], sort=False)
    # prior = cumulative sum up to previous row within each track
    csum = g["scrobbles_week"].transform("cumsum")
    df["prior_scrobbles_all_time"] = (csum - df["scrobbles_week"]).astype("Int64")
    return df


def add_prior_weeks_with_scrobbles(df):
    df = df.copy().sort_values(["artist_name", "track_name", "week_saturday_dt"], kind="stable")
    g = df.groupby(["artist_name", "track_name"], sort=False)
    # cumulative count of weeks with ≥1 scrobble, excluding current week
    pos = (df["scrobbles_week"] > 0).astype(int)
    cum_pos = g[["scrobbles_week"]].transform(lambda s: (s > 0).astype(int).cumsum())["scrobbles_week"]
    df["prior_weeks_with_scrobbles"] = (cum_pos - pos).astype("Int64")
    return df


def add_first_seen_week(df):
    df = df.copy()
    if "prior_scrobbles_all_time" not in df.columns:
        raise ValueError("prior_scrobbles_all_time must exist before calling add_first_seen_week()")
    df["first_seen_week"] = (df["prior_scrobbles_all_time"] == 0).astype(int)
    return df


def add_weeks_since_first_scrobble(df):
    df = df.copy().sort_values(["artist_name", "track_name", "week_saturday_dt"], kind="stable")
    g = df.groupby(["artist_name", "track_name"], sort=False)
    first_week_dt = g["week_saturday_dt"].transform("min")
    delta_days = (df["week_saturday_dt"] - first_week_dt).dt.days
    df["weeks_since_first_scrobble"] = np.floor_divide(delta_days, 7).astype("Int64")
    return df


# ---- Release (Spotify) ----
def parse_release_date(value):
    """
    Normalize release date strings:
      - 'YYYY'      -> YYYY-01-01
      - 'YYYY-MM'   -> YYYY-MM-01
      - 'YYYY-MM-DD' as-is
    Returns UTC pandas.Timestamp or NaT.
    """
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if not s:
        return pd.NaT
    if len(s) == 4 and s.isdigit():
        s = f"{s}-01-01"
    elif len(s) == 7 and s[:4].isdigit() and s[4] == "-":
        s = f"{s}-01"
    return pd.to_datetime(s, utc=True, errors="coerce")


def build_track_release_lookup(df):
    """
    Resolve earliest non-null release datetime per (artist_name, track_name).
    Returns a DataFrame with columns: artist_name, track_name, __release_dt
    """
    if "spotify_release_date" not in df.columns:
        return pd.DataFrame(columns=["artist_name", "track_name", "__release_dt"])

    rel = (
        df[["artist_name", "track_name", "spotify_release_date"]]
        .dropna(subset=["spotify_release_date"])
        .copy()
    )
    if len(rel) == 0:
        return pd.DataFrame(columns=["artist_name", "track_name", "__release_dt"])

    rel["__release_dt"] = rel["spotify_release_date"].map(parse_release_date)
    rel = rel.dropna(subset=["__release_dt"]).sort_values("__release_dt")
    if len(rel) == 0:
        return pd.DataFrame(columns=["artist_name", "track_name", "__release_dt"])

    key = ["artist_name", "track_name"]
    return rel.drop_duplicates(subset=key, keep="first")[key + ["__release_dt"]]


def add_days_since_release(df, release_lookup=None):
    """
    Adds days_since_release (float). Null-tolerant if lookup is empty.
    """
    df = df.copy()
    if release_lookup is None:
        release_lookup = build_track_release_lookup(df)

    if release_lookup.empty:
        df["days_since_release"] = np.nan
        return df

    key = ["artist_name", "track_name"]
    out = df.merge(release_lookup, on=key, how="left")
    out["days_since_release"] = (out["week_saturday_dt"] - out["__release_dt"]).dt.total_seconds() / 86400.0
    out = out.drop(columns=["__release_dt"])
    return out


def add_released_within_d(df, days: int = 28):
    """
    Adds released_within_{days}d (0/1). Requires days_since_release.
    """
    df = df.copy()
    if "days_since_release" not in df.columns:
        raise ValueError("days_since_release must exist before calling add_released_within_d()")
    cond = (df["days_since_release"] >= 0) & (df["days_since_release"] <= days)
    df[f"released_within_{days}d"] = cond.fillna(False).astype(int)
    return df


# ===========================
# Small family orchestrators (readability only)
# ===========================
def add_within_week_features(df):
    df = add_within_week_rank_by_scrobbles(df)
    return df


def add_momentum_family(df):
    df = add_scrobbles_prev_w(df, window=1)
    df = add_scrobbles_prev_w(df, window=4)
    df = add_week_over_week_change(df)
    df = add_momentum_ratio(df, window=4)
    return df


def add_history_family(df):
    df = add_prior_scrobbles_all_time(df)
    df = add_prior_weeks_with_scrobbles(df)
    df = add_first_seen_week(df)
    df = add_weeks_since_first_scrobble(df)
    return df


def add_release_family(df):
    lookup = build_track_release_lookup(df)
    df = add_days_since_release(df, release_lookup=lookup)
    df = add_released_within_d(df, days=28)
    return df


# ===========================
# Core V1 composition (no I/O)
# ===========================
def compute_core_v1_features(weekly_df):
    """
    Compose Core V1 features on top of a weekly base table.
    """
    df = weekly_df.copy()
    df = add_within_week_features(df)
    df = add_momentum_family(df)
    df = add_history_family(df)
    df = add_release_family(df)
    return df
