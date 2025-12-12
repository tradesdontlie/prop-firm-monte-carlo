"""
Historical Data Processor for MFE/MAE Analysis

This module processes parquet files containing futures data to extract:
- MFE/MAE distributions for candle opens (15min, 30min, 1H)
- Volatility regime classifications
- Stop/target hit probability matrices

Uses Polars for fast parquet I/O. This is a BUILD-TIME tool only.
The Streamlit app does NOT import this module.

Usage:
    from historical_data_processor import build_asset_regime_data
    data = build_asset_regime_data("NQ", Path("/path/to/parquet"))
"""

import polars as pl
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from datetime import time


# RTH window (Regular Trading Hours) - 9:30 AM to 3:00 PM ET
# Using 15:00 (3 PM) instead of 16:00 (4 PM) to ensure full candles complete
RTH_START = time(9, 30)
RTH_END = time(15, 0)

# Volatility regime percentile thresholds
REGIME_PERCENTILES = {
    "low_upper": 0.25,      # Low: <= 25th percentile
    "normal_upper": 0.75,   # Normal: 25th-75th percentile
    "high_upper": 0.90,     # High: 75th-90th percentile
    # Extreme: > 90th percentile
}

# Stop/target levels for hit probability matrix (in points)
STOP_LEVELS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
TARGET_LEVELS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]

# Candle timeframes to process
TIMEFRAMES = ["15min", "30min", "1H"]


def load_minute_data(asset: str, data_dir: Path) -> pl.DataFrame:
    """
    Load 1-minute continuous contract data for an asset.

    Args:
        asset: Symbol (e.g., "NQ", "ES")
        data_dir: Directory containing parquet files

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    file_path = data_dir / f"{asset}_1min_continuous.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Minute data not found: {file_path}")

    df = pl.read_parquet(file_path)

    # Ensure timestamp is datetime
    if df["timestamp"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

    return df


def load_daily_data(asset: str, data_dir: Path) -> pl.DataFrame:
    """
    Load daily continuous contract data for an asset.
    Used for volatility regime classification.
    """
    file_path = data_dir / f"{asset}_daily_continuous.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Daily data not found: {file_path}")

    df = pl.read_parquet(file_path)

    if df["timestamp"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

    return df


def filter_rth(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter DataFrame to Regular Trading Hours (9:30 AM - 3:00 PM ET).

    Args:
        df: DataFrame with timestamp column (UTC)

    Returns:
        DataFrame filtered to RTH only
    """
    # Convert UTC to Eastern time
    df = df.with_columns([
        pl.col("timestamp").dt.convert_time_zone("America/New_York").alias("timestamp_et")
    ])

    # Extract time component - cast to i32 to avoid overflow
    df = df.with_columns([
        pl.col("timestamp_et").dt.hour().cast(pl.Int32).alias("hour"),
        pl.col("timestamp_et").dt.minute().cast(pl.Int32).alias("minute")
    ])

    # Filter to RTH (9:30 - 15:00 ET)
    # time_minutes >= 9*60+30 AND time_minutes < 15*60
    df = df.with_columns([
        (pl.col("hour") * 60 + pl.col("minute")).alias("time_minutes")
    ])

    rth_df = df.filter(
        (pl.col("time_minutes") >= 9 * 60 + 30) &
        (pl.col("time_minutes") < 15 * 60)
    )

    return rth_df


def resample_to_candles(minute_df: pl.DataFrame, candle_size: str) -> pl.DataFrame:
    """
    Resample minute data to larger candles (15min, 30min, 1H).

    Args:
        minute_df: 1-minute data with RTH filter already applied
        candle_size: "15min", "30min", or "1H"

    Returns:
        DataFrame with OHLC candles at specified timeframe
    """
    # Map candle size to Polars duration string
    duration_map = {
        "15min": "15m",
        "30min": "30m",
        "1H": "1h"
    }

    if candle_size not in duration_map:
        raise ValueError(f"Invalid candle_size: {candle_size}. Use: {list(duration_map.keys())}")

    duration = duration_map[candle_size]

    # Group by candle period and aggregate OHLC
    candles = minute_df.group_by_dynamic(
        "timestamp",
        every=duration,
        closed="left",
        label="left"
    ).agg([
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.len().alias("bar_count")  # Number of minute bars in candle
    ])

    # Filter out incomplete candles (should have full bar count)
    expected_bars = {"15min": 15, "30min": 30, "1H": 60}
    min_bars = int(expected_bars[candle_size] * 0.8)  # Allow 80% completeness

    candles = candles.filter(pl.col("bar_count") >= min_bars)

    return candles.drop("bar_count")


def compute_mfe_mae(candles: pl.DataFrame) -> pl.DataFrame:
    """
    Compute MFE/MAE for each candle for both LONG and SHORT directions.

    For LONG entry at open:
        MFE = high - open (maximum favorable excursion)
        MAE = open - low  (maximum adverse excursion)

    For SHORT entry at open:
        MFE = open - low  (maximum favorable excursion)
        MAE = high - open (maximum adverse excursion)

    Args:
        candles: DataFrame with open, high, low, close columns

    Returns:
        DataFrame with additional columns: mfe_long, mae_long, mfe_short, mae_short
    """
    candles = candles.with_columns([
        # Long direction
        (pl.col("high") - pl.col("open")).alias("mfe_long"),
        (pl.col("open") - pl.col("low")).alias("mae_long"),

        # Short direction
        (pl.col("open") - pl.col("low")).alias("mfe_short"),
        (pl.col("high") - pl.col("open")).alias("mae_short"),

        # Close P&L if held to end of candle
        (pl.col("close") - pl.col("open")).alias("close_pnl_long"),
        (pl.col("open") - pl.col("close")).alias("close_pnl_short"),

        # Did candle close up or down?
        (pl.col("close") > pl.col("open")).alias("bullish"),
    ])

    return candles


def determine_high_low_order(
    minute_df: pl.DataFrame,
    candle_start: pl.Datetime,
    candle_size: str
) -> bool:
    """
    Determine if high came before low within a candle period.

    This is used to determine stop/target hit order when both levels
    could theoretically be hit within the same candle.

    Args:
        minute_df: Full minute data
        candle_start: Start timestamp of the candle
        candle_size: "15min", "30min", or "1H"

    Returns:
        True if high was reached before low, False otherwise
    """
    # Calculate candle end
    minutes = {"15min": 15, "30min": 30, "1H": 60}[candle_size]
    candle_end = candle_start + pl.duration(minutes=minutes)

    # Filter to candle period
    candle_bars = minute_df.filter(
        (pl.col("timestamp") >= candle_start) &
        (pl.col("timestamp") < candle_end)
    )

    if len(candle_bars) == 0:
        return True  # Default

    # Find which bar had the high and which had the low
    high_idx = candle_bars["high"].arg_max()
    low_idx = candle_bars["low"].arg_min()

    return high_idx < low_idx


def batch_determine_high_low_order(
    minute_df: pl.DataFrame,
    candles: pl.DataFrame,
    candle_size: str
) -> pl.DataFrame:
    """
    Batch determine high/low order for all candles.

    More efficient than calling determine_high_low_order per candle.
    """
    minutes = {"15min": 15, "30min": 30, "1H": 60}[candle_size]

    # Add candle period info to minute data
    minute_df = minute_df.with_columns([
        # Truncate timestamp to candle boundary
        pl.col("timestamp").dt.truncate(f"{minutes}m").alias("candle_start")
    ])

    # For each candle period, find the bar index of max high and min low
    order_df = minute_df.group_by("candle_start").agg([
        pl.col("high").arg_max().alias("high_idx"),
        pl.col("low").arg_min().alias("low_idx")
    ])

    # high_first = True if high came before low
    order_df = order_df.with_columns([
        (pl.col("high_idx") < pl.col("low_idx")).alias("high_first")
    ])

    # Join back to candles
    candles = candles.join(
        order_df.select(["candle_start", "high_first"]),
        left_on="timestamp",
        right_on="candle_start",
        how="left"
    )

    # Fill any missing values with True (default assumption)
    candles = candles.with_columns([
        pl.col("high_first").fill_null(True)
    ])

    return candles


def classify_volatility_regimes(
    daily_df: pl.DataFrame,
    lookback: int = 20
) -> tuple[pl.DataFrame, dict]:
    """
    Classify each day into a volatility regime based on rolling ATR%.

    Regimes:
        - Low: <= 25th percentile of ATR%
        - Normal: 25th-75th percentile
        - High: 75th-90th percentile
        - Extreme: > 90th percentile

    Args:
        daily_df: Daily OHLC data
        lookback: Rolling window for ATR calculation

    Returns:
        Tuple of (DataFrame with regime column, dict of percentile thresholds)
    """
    # Calculate daily range and range percentage
    daily_df = daily_df.with_columns([
        (pl.col("high") - pl.col("low")).alias("daily_range"),
    ])

    daily_df = daily_df.with_columns([
        (pl.col("daily_range") / pl.col("open") * 100).alias("range_pct")
    ])

    # Calculate rolling ATR%
    daily_df = daily_df.with_columns([
        pl.col("range_pct").rolling_mean(window_size=lookback).alias("atr_pct")
    ])

    # Drop rows with null ATR (first `lookback` rows)
    daily_df = daily_df.drop_nulls(subset=["atr_pct"])

    # Calculate percentile thresholds
    atr_values = daily_df["atr_pct"].to_numpy()
    thresholds = {
        "p25": float(np.percentile(atr_values, 25)),
        "p75": float(np.percentile(atr_values, 75)),
        "p90": float(np.percentile(atr_values, 90))
    }

    # Classify regimes
    daily_df = daily_df.with_columns([
        pl.when(pl.col("atr_pct") <= thresholds["p25"])
        .then(pl.lit("Low"))
        .when(pl.col("atr_pct") <= thresholds["p75"])
        .then(pl.lit("Normal"))
        .when(pl.col("atr_pct") <= thresholds["p90"])
        .then(pl.lit("High"))
        .otherwise(pl.lit("Extreme"))
        .alias("regime")
    ])

    return daily_df, thresholds


def assign_regimes_to_candles(
    candles: pl.DataFrame,
    daily_regimes: pl.DataFrame
) -> pl.DataFrame:
    """
    Assign volatility regime to each candle based on the day's regime.
    """
    # Extract date from candle timestamp
    candles = candles.with_columns([
        pl.col("timestamp").dt.date().alias("date")
    ])

    # Extract date from daily data
    daily_regimes = daily_regimes.with_columns([
        pl.col("timestamp").dt.date().alias("date")
    ])

    # Join on date
    candles = candles.join(
        daily_regimes.select(["date", "regime", "atr_pct"]),
        on="date",
        how="left"
    )

    # Fill missing regimes with "Normal"
    candles = candles.with_columns([
        pl.col("regime").fill_null("Normal")
    ])

    return candles


def compute_hit_probability_matrix(
    candles: pl.DataFrame,
    stop_levels: list[float],
    target_levels: list[float],
    direction: str = "long"
) -> dict:
    """
    Compute probability matrix for stop/target hits.

    For each stop_level × target_level combination, calculate:
        - P(target hit before stop)
        - P(stop hit before target)
        - P(neither hit - time exit)
        - Average P&L for time exits

    Args:
        candles: DataFrame with mfe/mae columns and high_first indicator
        stop_levels: List of stop loss levels in points
        target_levels: List of target levels in points
        direction: "long" or "short"

    Returns:
        Nested dict: {stop_level: {target_level: {probabilities}}}
    """
    mfe_col = f"mfe_{direction}"
    mae_col = f"mae_{direction}"
    close_pnl_col = f"close_pnl_{direction}"

    matrix = {}

    for stop in stop_levels:
        matrix[stop] = {}

        for target in target_levels:
            # Determine outcomes for each candle
            # Stop hit: MAE >= stop
            # Target hit: MFE >= target

            # Case 1: Only stop could be hit (MAE >= stop, MFE < target)
            stop_only = candles.filter(
                (pl.col(mae_col) >= stop) & (pl.col(mfe_col) < target)
            )

            # Case 2: Only target could be hit (MFE >= target, MAE < stop)
            target_only = candles.filter(
                (pl.col(mfe_col) >= target) & (pl.col(mae_col) < stop)
            )

            # Case 3: Both could be hit - need to check order
            both = candles.filter(
                (pl.col(mfe_col) >= target) & (pl.col(mae_col) >= stop)
            )

            # For direction=long: if high_first, target hit first; else stop hit first
            # For direction=short: if high_first, stop hit first; else target hit first
            if direction == "long":
                both_target_first = both.filter(pl.col("high_first"))
                both_stop_first = both.filter(~pl.col("high_first"))
            else:  # short
                both_target_first = both.filter(~pl.col("high_first"))
                both_stop_first = both.filter(pl.col("high_first"))

            # Case 4: Neither hit (MFE < target AND MAE < stop)
            neither = candles.filter(
                (pl.col(mfe_col) < target) & (pl.col(mae_col) < stop)
            )

            # Calculate counts
            total = len(candles)
            n_target = len(target_only) + len(both_target_first)
            n_stop = len(stop_only) + len(both_stop_first)
            n_neither = len(neither)

            # Calculate average P&L for time exits
            avg_time_exit_pnl = 0.0
            if n_neither > 0:
                avg_time_exit_pnl = float(neither[close_pnl_col].mean())

            matrix[stop][target] = {
                "target_first": n_target / total if total > 0 else 0,
                "stop_first": n_stop / total if total > 0 else 0,
                "neither": n_neither / total if total > 0 else 0,
                "avg_time_exit_pnl": avg_time_exit_pnl,
                "total_candles": total
            }

    return matrix


def compute_mfe_mae_distributions(
    candles: pl.DataFrame,
    direction: str = "long"
) -> dict:
    """
    Compute MFE/MAE percentile distributions by regime.

    Args:
        candles: DataFrame with mfe/mae columns and regime
        direction: "long" or "short"

    Returns:
        Dict with MFE and MAE distributions per regime
    """
    mfe_col = f"mfe_{direction}"
    mae_col = f"mae_{direction}"

    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]

    distributions = {
        "mfe": {},
        "mae": {},
        "regime_frequencies": {}
    }

    total_candles = len(candles)

    for regime in ["Low", "Normal", "High", "Extreme"]:
        regime_data = candles.filter(pl.col("regime") == regime)

        if len(regime_data) == 0:
            distributions["mfe"][regime] = [0.0] * len(percentiles)
            distributions["mae"][regime] = [0.0] * len(percentiles)
            distributions["regime_frequencies"][regime] = 0.0
            continue

        # Calculate frequency
        distributions["regime_frequencies"][regime] = len(regime_data) / total_candles

        # Calculate MFE percentiles
        mfe_values = regime_data[mfe_col].to_numpy()
        distributions["mfe"][regime] = [
            float(np.percentile(mfe_values, p)) for p in percentiles
        ]

        # Calculate MAE percentiles
        mae_values = regime_data[mae_col].to_numpy()
        distributions["mae"][regime] = [
            float(np.percentile(mae_values, p)) for p in percentiles
        ]

    distributions["percentiles"] = percentiles

    return distributions


def build_asset_data(
    asset: str,
    data_dir: Path,
    timeframe: str,
    start_date: Optional[str] = None
) -> dict:
    """
    Build complete MFE/MAE data for a single asset and timeframe.

    Args:
        asset: Symbol (e.g., "NQ")
        data_dir: Directory containing parquet files
        timeframe: "15min", "30min", or "1H"
        start_date: Optional start date filter (e.g., "2020-01-01")

    Returns:
        Dict containing all distributions and hit matrices
    """
    print(f"Processing {asset} {timeframe}...")

    # Load data
    minute_df = load_minute_data(asset, data_dir)
    daily_df = load_daily_data(asset, data_dir)

    # Filter by start date if specified
    if start_date:
        # Use Polars datetime literal for proper dtype matching
        start_expr = pl.lit(start_date).str.to_datetime("%Y-%m-%d").dt.cast_time_unit("ns").dt.replace_time_zone("UTC")
        minute_df = minute_df.filter(pl.col("timestamp") >= start_expr)
        daily_df = daily_df.filter(pl.col("timestamp") >= start_expr)
        print(f"  Filtered to data from {start_date} onwards")

    # Filter to RTH
    rth_minute = filter_rth(minute_df)

    # Resample to candles
    candles = resample_to_candles(rth_minute, timeframe)

    # Compute MFE/MAE
    candles = compute_mfe_mae(candles)

    # Determine high/low order
    candles = batch_determine_high_low_order(rth_minute, candles, timeframe)

    # Classify volatility regimes
    daily_with_regimes, regime_thresholds = classify_volatility_regimes(daily_df)

    # Assign regimes to candles
    candles = assign_regimes_to_candles(candles, daily_with_regimes)

    print(f"  {len(candles)} candles processed")

    # Build result
    result = {
        "asset": asset,
        "timeframe": timeframe,
        "total_candles": len(candles),
        "regime_thresholds": regime_thresholds,
    }

    # Compute distributions for both directions
    for direction in ["long", "short"]:
        distributions = compute_mfe_mae_distributions(candles, direction)
        result[f"distributions_{direction}"] = distributions

        # Compute hit probability matrix
        hit_matrix = compute_hit_probability_matrix(
            candles, STOP_LEVELS, TARGET_LEVELS, direction
        )
        result[f"hit_matrix_{direction}"] = hit_matrix

    # Compute by regime
    result["by_regime"] = {}
    for regime in ["Low", "Normal", "High", "Extreme"]:
        regime_candles = candles.filter(pl.col("regime") == regime)
        if len(regime_candles) == 0:
            continue

        result["by_regime"][regime] = {
            "candle_count": len(regime_candles)
        }

        for direction in ["long", "short"]:
            hit_matrix = compute_hit_probability_matrix(
                regime_candles, STOP_LEVELS, TARGET_LEVELS, direction
            )
            result["by_regime"][regime][f"hit_matrix_{direction}"] = hit_matrix

    return result


def get_available_assets(data_dir: Path) -> list[str]:
    """
    Get list of available assets from parquet files in directory.
    """
    assets = set()
    for file in data_dir.glob("*_1min_continuous.parquet"):
        asset = file.name.replace("_1min_continuous.parquet", "")
        assets.add(asset)
    return sorted(list(assets))


if __name__ == "__main__":
    # Test with NQ data
    import sys

    data_dir = Path("/Users/tradesdontlie/Downloads/futures_data 2")

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    print("Available assets:", get_available_assets(data_dir))

    # Test processing NQ 1H
    result = build_asset_data("NQ", data_dir, "1H")

    print("\nResults:")
    print(f"Total candles: {result['total_candles']}")
    print(f"Regime thresholds: {result['regime_thresholds']}")
    print(f"Regime frequencies (long): {result['distributions_long']['regime_frequencies']}")
    print(f"MFE percentiles (long, Normal): {result['distributions_long']['mfe']['Normal']}")
    print(f"MAE percentiles (long, Normal): {result['distributions_long']['mae']['Normal']}")
