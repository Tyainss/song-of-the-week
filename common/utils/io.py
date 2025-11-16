
import json
import pandas as pd
from pathlib import Path


def write_csv(path: Path, df: pd.DataFrame, append: bool = False) -> None:
    """
    Always write LF line-endings and UTF-8 so git diffs are stable across OS.
    Avoid duplicate headers when appending.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        df.to_csv(
            path,
            mode="a",
            index=False,
            header=False,
            encoding="utf-8",
            lineterminator="\n",
        )
    else:
        df.to_csv(
            path,
            index=False,
            header=True,
            encoding="utf-8",
            lineterminator="\n",
        )


def read_csv(
    path: Path,
    usecols: list[str] | None = None,
    dtype: dict | None = None,
    parse_dates: list[str] | None = None,
    safe: bool = False,
) -> pd.DataFrame:
    """
    Wrapper around pandas.read_csv with consistent defaults.
    - usecols: same semantics as pandas.
    - safe=True: if requested columns are missing, read file without usecols and
      return only the intersection (no exception).
    """
    try:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, parse_dates=parse_dates)
    except Exception:
        if not safe:
            raise
        df = pd.read_csv(path, dtype=dtype, parse_dates=parse_dates)
        if usecols:
            keep = [c for c in usecols if c in df.columns]
            return df[keep]
        return df

def write_json(path: Path, data: dict, indent: int = 2) -> None:
    """
    Write a JSON file with UTF-8 encoding and pretty-printing.
    Creates parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=indent,
        )