"""Yahoo Finance data source utilities for FX prices."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

ALLOWED_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}

ALLOWED_PERIODS = {
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
}


def download_fx_yahoo(
    symbol: str,
    interval: str = "5m",
    period: Optional[str] = "30d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Download FX OHLCV data from Yahoo Finance with input validation.

    Args:
        symbol: Yahoo ticker, e.g. "EURUSD=X".
        interval: Bar interval accepted by yfinance.
        period: Period spec; if None, start/end must be provided.
        start: Start date/time string when period is None.
        end: End date/time string when period is None.
        auto_adjust: Forwarded to yfinance.download.

    Returns:
        DataFrame indexed by datetime with standardized columns.

    Raises:
        ValueError: On invalid arguments or empty result.
    """
    if interval not in ALLOWED_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. Allowed: {sorted(ALLOWED_INTERVALS)}"
        )

    kwargs: dict = {
        "tickers": symbol,
        "interval": interval,
        "auto_adjust": auto_adjust,
        "progress": False,
    }

    if period is not None:
        if period not in ALLOWED_PERIODS:
            raise ValueError(
                f"Invalid period '{period}'. Allowed: {sorted(ALLOWED_PERIODS)}"
            )
        kwargs["period"] = period
    else:
        if not start or not end:
            raise ValueError("When period is None, both start and end must be provided.")
        kwargs["start"] = start
        kwargs["end"] = end

    df = yf.download(**kwargs)

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] for col in df.columns]

    if df.empty:
        raise ValueError(
            "No data returned. Try smaller period for intraday. Note: intraday data cannot "
            "extend past ~60 days and 1m often limited to ~7 days."
        )

    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    try:
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
    except AttributeError:
        pass
    df = df[~df.index.isna()].sort_index()

    standard_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if not standard_cols:
        raise ValueError("Downloaded data does not include expected OHLC columns.")

    return df[standard_cols]
