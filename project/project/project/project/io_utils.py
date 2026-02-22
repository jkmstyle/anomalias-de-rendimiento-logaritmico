"""Input/output helper utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


KNOWN_PRICE_COLS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid",
    "ask",
    "Bid",
    "Ask",
]


def read_csv_prices(path: str) -> pd.DataFrame:
    """Read price data from CSV.

    Supports timestamp in a dedicated ``timestamp`` column or in index.
    Tries to preserve known OHLC or bid/ask related columns.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Input CSV is empty.")

    cols_lower = {c.lower(): c for c in df.columns}
    if "timestamp" in cols_lower:
        ts_col = cols_lower["timestamp"]
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.set_index(ts_col)
    else:
        try:
            idx = pd.to_datetime(df.index, errors="coerce")
            if idx.isna().all():
                df = pd.read_csv(csv_path, index_col=0)
                idx = pd.to_datetime(df.index, errors="coerce")
            if idx.isna().all():
                raise ValueError
            df.index = idx
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "Could not detect timestamp. Provide a 'timestamp' column or datetime index."
            ) from exc

    keep_cols = [c for c in df.columns if c in KNOWN_PRICE_COLS]
    if keep_cols:
        df = df[keep_cols]

    return df.sort_index()


def save_df(df: pd.DataFrame, path: str) -> None:
    """Save dataframe to CSV creating parent dirs as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
