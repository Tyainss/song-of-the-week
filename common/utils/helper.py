
import pycountry
from datetime import datetime
import pandas as pd

def get_image_text(image_list, size: str) -> str | None:
    for image in image_list:
        if image.get("size") == size:
            return image.get("#text") or None
    return None

def country_name_from_iso2(iso_code: str | None) -> str:
    if not iso_code:
        return "Unknown"
    try:
        country = pycountry.countries.get(alpha_2=iso_code.upper())
        return country.name if country else "Unknown"
    except Exception:
        return "Unknown"


def format_date(date_str: str | None) -> str | None:
    """
    Accepts 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD'. Returns a normalized ISO string.
    Fills missing month/day with '01'. Returns None on empty/invalid.
    """
    if not date_str:
        return None
    s = str(date_str).strip()
    try:
        if len(s) == 4:          # YYYY
            dt = datetime.strptime(s + "-01-01", "%Y-%m-%d")
        elif len(s) == 7:        # YYYY-MM
            dt = datetime.strptime(s + "-01", "%Y-%m-%d")
        else:                    # assume YYYY-MM-DD
            dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None

def unix_from_lastfm_datetime(date_str: str | None) -> str | None:
    if not date_str:
        return None
    dt = datetime.strptime(date_str, "%d %b %Y, %H:%M")
    return str(int(dt.timestamp()))

def to_bool_or_na(x):
    if pd.isna(x):
        return pd.NA
    return bool(x)

def replace_nan_by_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    for column, dtype in schema.items():
        if column not in df.columns:
            continue
        if dtype == "str":
            df[column] = df[column].fillna("")
        elif dtype in ("int64", "int32", "int"):
            df[column] = df[column].fillna(0)
    return df
