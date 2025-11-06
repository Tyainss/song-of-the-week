
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


def unix_from_lastfm_datetime(date_str: str | None) -> str | None:
    # Last.fm format: '01 Jan 2024, 13:45'
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
