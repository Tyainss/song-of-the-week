
import re
import pandas as pd

def drop_columns(df, columns):
    cols = [c for c in columns if c in df.columns]
    return df.drop(columns=cols) if cols else df

# -------------------------
# Text
# -------------------------

def trim_text_columns(df, columns):
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(str).str.strip()
    return out

# -------------------------
# Dates
# -------------------------

def _standardize_release_date_value(s):
    if not isinstance(s, str) or not s.strip():
        return None, None

    x = s.strip()

    # YYYY-MM-DD HH:MM:SS -> keep date part
    m_dt = re.match(r"^(\d{4}-\d{2}-\d{2})\s\d{2}:\d{2}:\d{2}$", x)
    if m_dt:
        return m_dt.group(1), "day"

    # YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", x):
        return x, "day"

    # YYYY-MM
    if re.match(r"^\d{4}-\d{2}$", x):
        return f"{x}-01", "month"

    # YYYY
    if re.match(r"^\d{4}$", x):
        return f"{x}-01-01", "year"

    return None, None

def standardize_spotify_release_date(df, col="spotify_release_date"):
    if col not in df.columns:
        return df
    out = df.copy()
    std_vals = []
    granularities = []
    for v in out[col].tolist():
        val, gran = _standardize_release_date_value(v if pd.notna(v) else None)
        std_vals.append(val)
        granularities.append(gran)

    out[col] = pd.Series(std_vals, index=out.index)
    out["release_date_granularity"] = pd.Series(granularities, index=out.index)
    return out

def add_date_missing_flags(df, date_cols):
    cols = [c for c in date_cols if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        out[f"{c}_was_missing"] = out[c].isna().astype(int)

    return out

# -------------------------
# Genres
# -------------------------

# Simple, ordered rules using substring checks (lowercase).
BUCKET_RULES = [
    ("hip_hop_rap",       ["hip hop", "rap", "trap", "drill", "boom bap", "g-funk", "grime"]),
    ("rnb_soul",          ["r&b", "rnb", "soul", "motown", "new jack swing", "neo soul", "philly soul"]),
    ("electronic_dance",  ["edm", "house", "techno", "trance", "disco", "vaporwave", "ambient", "downtempo",
                           "future house", "future bass", "progressive house", "slap house", "garage", "chillstep",
                           "chillwave", "psytrance", "moombahton", "big room", "eurodance", "italo disco",
                           "jersey club", "nightcore", "stutter house", "disco house", "dubstep", "electroclash"]),
    ("jazz",              ["jazz", "bebop", "hard bop", "fusion", "vocal jazz", "nu jazz", "cool jazz", "free jazz",
                           "jazz funk", "jazz house", "jazz blues", "jazz beats"]),
    ("classical_art",     ["classical", "chamber", "opera", "neoclassical", "minimalism", "baroque",
                           "soundtrack", "musicals", "orchestral"]),
    ("folk_country_americana", ["folk", "bluegrass", "americana", "singer-songwriter", "country", "alt country"]),
    ("metal_hard",        ["metal", "hardcore", "thrash", "grindcore", "sludge", "stoner", "post-hardcore",
                           "metalcore", "gothic metal", "power metal", "black metal", "nu metal", "melodic death"]),
    ("rock",              ["rock", "shoegaze", "post-rock", "post-punk", "proto-punk", "new wave", "glam rock",
                           "progressive rock", "math rock", "indie rock", "garage rock", "hard rock", "space rock",
                           "grunge", "power pop", "yacht rock", "classic rock", "art rock", "riot grrrl", "punk",
                           "surf rock", "rap rock", "krautrock"]),
    ("pop",               [" pop", "synthpop", "hyperpop", "art pop", "bedroom pop", "indie pop", "dance pop",
                           "power pop", "pop punk", "britpop", "city pop", "soft pop"]),
    ("latin",             ["latin", "mpb", "bossa nova", "samba", "cumbia", "reggaeton", "salsa", "bachata",
                           "bolero", "sertanejo", "pagode", "tecnobrega", "neoperreo", "calypso", "forro", "forró"]),
    # ("reggae_dancehall",  ["reggae", "dancehall", "rocksteady", "dub"]),
    ("world_regional",    ["fado", "flamenco", "afrobeat", "afrobeats", "highlife", "gnawa",
                           "k-pop", "kpop", "k-rap", "k rap", "k-rock", "k rock",
                           "j-pop", "jpop", "j-rap", "j rap", "j-rock", "j rock", "j-r&b", "mandopop",
                           "celtic", "mariachi", "kizomba", "kuduro", "bhangra", "pagode baiano",
                           "sertanejo", "funk carioca", "brazilian funk", "brazilian hip hop",
                           "turkish hip hop", "german hip hop", "french rap"]),
    ("experimental_avant",["experimental", "avant", "noise", "idm", "drone", "witch house", "psych", "post-"]),
]

def _map_genre_to_bucket(genre):
    if not isinstance(genre, str) or not genre:
        return "other"
    g = genre.lower().strip()
    for bucket, needles in BUCKET_RULES:
        for needle in needles:
            if needle in g:
                return bucket
    if g in {"spoken word", "comedy"}:
        return "other"
    return "other"

def add_genre_bucket(df, source_col="spotify_genres"):
    if source_col not in df.columns:
        return df
    out = df.copy()

    # Normalize and split per row
    raw = out[source_col].fillna("").astype(str)
    split_lists = raw.apply(lambda x: [t.strip().lower() for t in x.split(",") if t.strip()])

    # Map each genre to a bucket
    mapped_lists = split_lists.apply(lambda lst: [_map_genre_to_bucket(g) for g in lst])

    # Choose the most frequent bucket per row (mode). If tie or empty, pick the first or 'unknown'.
    def pick_bucket(buckets):
        if not buckets:
            return "unknown"
        s = pd.Series(buckets)
        return s.mode().iloc[0]

    out["genre_bucket"] = mapped_lists.apply(pick_bucket)
    out["genre_missing"] = split_lists.apply(lambda lst: len(lst) == 0).astype(int)
    return out

# -------------------------
# Numeric columns
# -------------------------

def to_numeric(df, columns):
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def fill_with_median_and_flag(df, columns):
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        flag_col = f"{c}_was_missing"
        was_na = out[c].isna()
        med = out[c].median()
        out[flag_col] = was_na.astype(int)
        out[c] = out[c].fillna(med)
    return out

def round_counts_to_int(df, columns):
    """
    For count-like numeric columns, round and store as nullable Int64.
    """
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].round().astype("Int64")
    return out