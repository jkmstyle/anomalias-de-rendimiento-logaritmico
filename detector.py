"""Core anomaly detection pipeline for FX log-returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EventAggregation:
    """Internal state holder for event aggregation."""

    event_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    max_abs_z_mad: float
    max_z_ewma: float
    max_abs_return: float
    count_anomaly_bars: int
    has_spread_flag: bool


def _find_column(df: pd.DataFrame, name: str) -> str | None:
    mapping = {c.lower(): c for c in df.columns}
    return mapping.get(name.lower())


def clean_prices(
    df: pd.DataFrame,
    timestamp_col: str | None = None,
    keep: str = "last",
    dropna: bool = True,
    spike_revert_ratio: float = 0.7,
    spike_lookahead: int = 2,
) -> pd.DataFrame:
    """Clean and validate raw prices, marking potential feed spikes via ``data_issue``."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    cleaned = df.copy()
    if timestamp_col is not None:
        if timestamp_col not in cleaned.columns:
            raise ValueError(f"timestamp_col '{timestamp_col}' not found in dataframe.")
        cleaned[timestamp_col] = pd.to_datetime(cleaned[timestamp_col], errors="coerce")
        cleaned = cleaned.set_index(timestamp_col)

    cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
    cleaned = cleaned[~cleaned.index.isna()]
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep=keep)]

    if dropna:
        cleaned = cleaned.dropna(how="any")

    if cleaned.empty:
        raise ValueError("No rows left after cleaning (possibly all NaN or invalid timestamps).")

    cleaned["data_issue"] = False

    close_col = _find_column(cleaned, "close")
    if close_col is None:
        bid_col = _find_column(cleaned, "bid")
        ask_col = _find_column(cleaned, "ask")
        if bid_col and ask_col:
            prelim_price = (cleaned[bid_col].astype(float) + cleaned[ask_col].astype(float)) / 2.0
        else:
            return cleaned
    else:
        prelim_price = cleaned[close_col].astype(float)

    prelim_ret = np.log(prelim_price).diff()
    rolling_std = prelim_ret.rolling(200, min_periods=20).std()
    fallback = prelim_ret.abs().rolling(200, min_periods=20).quantile(0.95)
    threshold = 8.0 * rolling_std
    threshold = threshold.where((threshold > 0) & threshold.notna(), fallback)
    threshold = threshold.fillna(np.inf)

    spike_mask = prelim_ret.abs() > threshold
    for i in np.where(spike_mask.fillna(False).to_numpy())[0]:
        if i >= len(prelim_ret) - 1:
            continue
        end_i = min(i + spike_lookahead, len(prelim_ret) - 1)
        future_window = prelim_ret.iloc[i + 1 : end_i + 1]
        if future_window.empty:
            continue
        rev = (future_window.sum() * np.sign(prelim_ret.iloc[i])) <= (
            -spike_revert_ratio * abs(prelim_ret.iloc[i])
        )
        if bool(rev):
            cleaned.iloc[i, cleaned.columns.get_loc("data_issue")] = True

    return cleaned


def build_price_series(df: pd.DataFrame, price_mode: str) -> pd.DataFrame:
    """Build standardized ``price`` and optional ``spread`` series."""
    out = df.copy()
    mode = price_mode.lower()

    if mode == "close":
        close_col = _find_column(out, "close")
        if close_col is None:
            raise ValueError("price_mode='close' requires a Close/close column.")
        out["price"] = out[close_col].astype(float)
        if "spread" not in out.columns:
            out["spread"] = np.nan
    elif mode == "mid":
        bid_col = _find_column(out, "bid")
        ask_col = _find_column(out, "ask")
        if not bid_col or not ask_col:
            raise ValueError(
                "price_mode='mid' requires bid/ask columns (case-insensitive)."
            )
        bid = out[bid_col].astype(float)
        ask = out[ask_col].astype(float)
        out["price"] = (bid + ask) / 2.0
        out["spread"] = ask - bid
    else:
        raise ValueError("price_mode must be one of {'close', 'mid'}.")

    return out


def compute_log_returns(price: pd.Series) -> pd.Series:
    """Compute log returns: log(P_t) - log(P_{t-1})."""
    if price is None or price.empty:
        raise ValueError("Price series is empty.")
    if (price <= 0).any():
        raise ValueError("Price series contains non-positive values; log-return undefined.")
    return np.log(price).diff()


def compute_rolling_mad_zscore(r: pd.Series, window: int, eps: float) -> pd.Series:
    """Compute robust rolling MAD z-score for returns."""
    if window < 5:
        raise ValueError("mad window must be >= 5.")
    med = r.rolling(window, min_periods=max(10, window // 5)).median()
    mad = (r - med).abs().rolling(window, min_periods=max(10, window // 5)).median()
    z = (r - med) / (1.4826 * mad + eps)
    return z


def compute_ewma_vol(r: pd.Series, lam: float, eps: float) -> pd.Series:
    """Compute EWMA volatility estimate."""
    if not (0.0 < lam < 1.0):
        raise ValueError("ewma lambda must be in (0, 1).")

    shifted_sq = r.shift(1).pow(2).fillna(0.0)
    sigma2 = shifted_sq.ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(sigma2) + eps


def detect_anomaly_bars(
    df: pd.DataFrame,
    mad_window: int,
    mad_threshold: float,
    ewma_lambda: float,
    ewma_threshold: float,
    combine_mode: str = "AND",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Detect anomaly bars from price using MAD and EWMA signals."""
    if "price" not in df.columns:
        raise ValueError("Input dataframe must contain 'price' column.")

    out = df.copy()
    out["log_return"] = compute_log_returns(out["price"].astype(float))
    out["z_mad"] = compute_rolling_mad_zscore(out["log_return"], mad_window, eps)
    out["sigma_ewma"] = compute_ewma_vol(out["log_return"], ewma_lambda, eps)
    out["z_ewma"] = out["log_return"].abs() / out["sigma_ewma"]

    cmode = combine_mode.upper()
    mad_flag = out["z_mad"].abs() >= mad_threshold
    ewma_flag = out["z_ewma"] >= ewma_threshold
    if cmode == "AND":
        out["is_anomaly_bar"] = mad_flag & ewma_flag
    elif cmode == "OR":
        out["is_anomaly_bar"] = mad_flag | ewma_flag
    else:
        raise ValueError("combine_mode must be 'AND' or 'OR'.")

    out["is_anomaly_bar"] = out["is_anomaly_bar"].fillna(False)
    out["anomaly_type"] = ""
    return out


def apply_spread_filter(
    df: pd.DataFrame,
    spread_window: int,
    spread_multiplier: float,
) -> pd.DataFrame:
    """Apply spread-based liquidity filter to anomaly bars."""
    out = df.copy()
    if "is_anomaly_bar" not in out.columns:
        raise ValueError("Missing 'is_anomaly_bar' column.")

    if "spread" in out.columns and out["spread"].notna().any():
        spread_med = out["spread"].rolling(spread_window, min_periods=max(10, spread_window // 5)).median()
        high_spread = out["spread"] > (spread_multiplier * spread_med)
        out["is_market_anomaly"] = out["is_anomaly_bar"] & (~high_spread.fillna(False))
        liq_mask = out["is_anomaly_bar"] & high_spread.fillna(False)
        out.loc[liq_mask, "anomaly_type"] = "liquidity/spread"
        out.loc[~out["is_anomaly_bar"], "anomaly_type"] = ""
    else:
        out["is_market_anomaly"] = out["is_anomaly_bar"]
        out.loc[~out["is_anomaly_bar"], "anomaly_type"] = ""
    return out


def group_anomalies_into_events(df: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame:
    """Group anomaly bars into events while respecting cooldown rules."""
    if "is_market_anomaly" not in df.columns:
        raise ValueError("Missing 'is_market_anomaly' column.")

    ordered = df.sort_index()
    events: list[EventAggregation] = []
    active: EventAggregation | None = None
    no_anomaly_count = 0

    for ts, row in ordered.iterrows():
        is_anom = bool(row.get("is_market_anomaly", False))
        if active is None:
            if is_anom:
                active = EventAggregation(
                    event_id=len(events) + 1,
                    start_time=ts,
                    end_time=ts,
                    max_abs_z_mad=float(abs(row.get("z_mad", np.nan))),
                    max_z_ewma=float(row.get("z_ewma", np.nan)),
                    max_abs_return=float(abs(row.get("log_return", np.nan))),
                    count_anomaly_bars=1,
                    has_spread_flag=(row.get("anomaly_type", "") == "liquidity/spread"),
                )
                no_anomaly_count = 0
            continue

        if is_anom:
            active.end_time = ts
            active.count_anomaly_bars += 1
            active.max_abs_z_mad = float(np.nanmax([active.max_abs_z_mad, abs(row.get("z_mad", np.nan))]))
            active.max_z_ewma = float(np.nanmax([active.max_z_ewma, row.get("z_ewma", np.nan)]))
            active.max_abs_return = float(
                np.nanmax([active.max_abs_return, abs(row.get("log_return", np.nan))])
            )
            active.has_spread_flag = active.has_spread_flag or (
                row.get("anomaly_type", "") == "liquidity/spread"
            )
            no_anomaly_count = 0
        else:
            no_anomaly_count += 1
            if no_anomaly_count >= cooldown_bars:
                events.append(active)
                active = None
                no_anomaly_count = 0

    if active is not None:
        events.append(active)

    if not events:
        return pd.DataFrame(
            columns=[
                "event_id",
                "start_time",
                "end_time",
                "max_abs_z_mad",
                "max_z_ewma",
                "max_abs_return",
                "count_anomaly_bars",
                "anomaly_type",
            ]
        )

    return pd.DataFrame(
        {
            "event_id": [e.event_id for e in events],
            "start_time": [e.start_time for e in events],
            "end_time": [e.end_time for e in events],
            "max_abs_z_mad": [e.max_abs_z_mad for e in events],
            "max_z_ewma": [e.max_z_ewma for e in events],
            "max_abs_return": [e.max_abs_return for e in events],
            "count_anomaly_bars": [e.count_anomaly_bars for e in events],
            "anomaly_type": ["liquidity/spread" if e.has_spread_flag else "" for e in events],
        }
    )


def classify_events(
    df: pd.DataFrame,
    events: pd.DataFrame,
    reversion_window: int,
    reversion_ratio: float,
    follow_through_bars: int,
    regime_bars: int,
    regime_sigma_quantile: float = 0.9,
) -> pd.DataFrame:
    """Heuristically classify event types for Forex anomalies."""
    if events.empty:
        return events

    bars = df.sort_index()
    sigma_q = bars["sigma_ewma"].quantile(regime_sigma_quantile)

    classified = events.copy()
    idx = bars.index

    for i, ev in classified.iterrows():
        if ev["anomaly_type"] == "liquidity/spread":
            continue

        start = pd.to_datetime(ev["start_time"])
        end = pd.to_datetime(ev["end_time"])
        if start not in idx or end not in idx:
            classified.at[i, "anomaly_type"] = "unknown"
            continue

        s_pos = idx.get_loc(start)
        e_pos = idx.get_loc(end)
        p_start = float(bars.iloc[s_pos]["price"])
        p_end = float(bars.iloc[e_pos]["price"])
        initial_move = p_end - p_start

        # spike/reversion
        rev_end = min(len(bars) - 1, e_pos + reversion_window)
        rev_slice = bars.iloc[e_pos + 1 : rev_end + 1]
        if initial_move != 0 and not rev_slice.empty:
            if np.sign(initial_move) > 0:
                best_revert = p_end - rev_slice["price"].min()
            else:
                best_revert = rev_slice["price"].max() - p_end
            if best_revert >= reversion_ratio * abs(initial_move):
                classified.at[i, "anomaly_type"] = "spike/reversion"
                continue

        # break/event follow-through
        ft_end = min(len(bars) - 1, e_pos + follow_through_bars)
        ft_slice = bars.iloc[e_pos + 1 : ft_end + 1]
        direction = np.sign(initial_move)
        if direction == 0:
            event_ret = bars.iloc[s_pos : e_pos + 1]["log_return"].sum()
            direction = np.sign(event_ret)
        ft_ret = ft_slice["log_return"].sum() if not ft_slice.empty else 0.0
        sigma_start = float(bars.iloc[s_pos]["sigma_ewma"])
        if direction != 0 and (ft_ret * direction) > (1.5 * sigma_start):
            classified.at[i, "anomaly_type"] = "break/event"
            continue

        # regime shift
        reg_end = min(len(bars) - 1, e_pos + regime_bars)
        reg_slice = bars.iloc[e_pos : reg_end + 1]["sigma_ewma"]
        if len(reg_slice) >= regime_bars and (reg_slice > sigma_q).mean() >= 0.8:
            classified.at[i, "anomaly_type"] = "regime_shift"
            continue

        classified.at[i, "anomaly_type"] = "unknown"

    return classified


def run_detector(df_raw: pd.DataFrame, price_mode: str, params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run complete anomaly detector pipeline returning bars and events dataframes."""
    cleaned = clean_prices(
        df_raw,
        timestamp_col=params.get("timestamp_col"),
        keep=params.get("keep", "last"),
        dropna=params.get("dropna", True),
    )
    priced = build_price_series(cleaned, price_mode=price_mode)
    bars = detect_anomaly_bars(
        priced,
        mad_window=params.get("mad_window", 500),
        mad_threshold=params.get("mad_threshold", 5.0),
        ewma_lambda=params.get("ewma_lambda", 0.94),
        ewma_threshold=params.get("ewma_threshold", 4.0),
        combine_mode=params.get("combine_mode", "AND"),
        eps=params.get("eps", 1e-12),
    )
    bars = apply_spread_filter(
        bars,
        spread_window=params.get("spread_window", 200),
        spread_multiplier=params.get("spread_multiplier", 3.0),
    )

    events = group_anomalies_into_events(
        bars,
        cooldown_bars=params.get("cooldown_bars", 5),
    )
    events = classify_events(
        bars,
        events,
        reversion_window=params.get("reversion_window", 5),
        reversion_ratio=params.get("reversion_ratio", 0.6),
        follow_through_bars=params.get("follow_through_bars", 3),
        regime_bars=params.get("regime_bars", 50),
        regime_sigma_quantile=params.get("regime_sigma_quantile", 0.9),
    )

    bars = bars.copy()
    bars.index.name = "timestamp"
    return bars, events
