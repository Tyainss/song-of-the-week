
import pandas as pd
from pathlib import Path


def write_csv(path: Path, df: pd.DataFrame, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)


def read_csv(path: Path, dtype: dict | None = None, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=dtype, parse_dates=parse_dates)
