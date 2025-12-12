"""
Regime Cache Loader for Monte Carlo Simulation

This module loads the pre-computed regime cache (JSON) and provides
fast sampling functions for the Monte Carlo simulation.

NO POLARS DEPENDENCY - This is a runtime module.
Uses only: json (stdlib), numpy, pathlib

The cache is loaded once at startup and provides microsecond-level
sampling for MFE/MAE distributions and hit probabilities.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_CACHE_PATH = Path.home() / ".queen_streamlit" / "regime_cache.json"

# Cache is loaded once and stored globally
_REGIME_CACHE: Optional[dict] = None


@dataclass
class RegimeDistributions:
    """MFE/MAE distributions for a specific regime."""
    mfe_percentiles: list[float]
    mae_percentiles: list[float]
    percentile_values: list[int]  # [5, 10, 25, 50, 75, 90, 95, 99]


@dataclass
class HitProbabilities:
    """Pre-computed hit probabilities for a stop/target combination."""
    target_first: float
    stop_first: float
    neither: float
    avg_time_exit_pnl: float


@dataclass
class AssetTimeframeData:
    """Data for a single asset at a specific timeframe."""
    asset: str
    timeframe: str
    total_candles: int

    # Regime frequencies (probability of each regime)
    regime_frequencies: dict[str, float]

    # Distributions per regime per direction
    distributions_long: dict[str, RegimeDistributions]
    distributions_short: dict[str, RegimeDistributions]

    # Hit matrices per regime per direction
    # Structure: hit_matrices[direction][regime][stop][target] = HitProbabilities
    hit_matrices_long: dict[str, dict]
    hit_matrices_short: dict[str, dict]


def load_regime_cache(cache_path: Optional[Path] = None) -> dict:
    """
    Load the regime cache from JSON file.

    Args:
        cache_path: Path to cache file. If None, uses default location.

    Returns:
        The full cache dictionary

    Raises:
        FileNotFoundError: If cache file doesn't exist
    """
    global _REGIME_CACHE

    if cache_path is None:
        cache_path = DEFAULT_CACHE_PATH

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Regime cache not found at {cache_path}. "
            "Run 'python build_regime_cache.py' to create it."
        )

    with open(cache_path, "r") as f:
        _REGIME_CACHE = json.load(f)

    return _REGIME_CACHE


def get_cache() -> dict:
    """Get the loaded cache, loading it if necessary."""
    global _REGIME_CACHE

    if _REGIME_CACHE is None:
        load_regime_cache()

    return _REGIME_CACHE


def get_available_assets() -> list[str]:
    """Get list of assets in the cache."""
    cache = get_cache()
    return list(cache.get("assets", {}).keys())


def get_available_timeframes() -> list[str]:
    """Get list of timeframes in the cache."""
    cache = get_cache()
    metadata = cache.get("metadata", {})
    return metadata.get("timeframes", ["15min", "30min", "1H"])


def get_asset_data(asset: str, timeframe: str) -> Optional[dict]:
    """
    Get raw data for a specific asset and timeframe.

    Returns None if asset/timeframe not found.
    """
    cache = get_cache()
    assets = cache.get("assets", {})

    if asset not in assets:
        return None

    return assets[asset].get(timeframe)


def sample_regime(asset: str, timeframe: str) -> str:
    """
    Sample a volatility regime based on historical frequencies.

    Args:
        asset: Symbol (e.g., "NQ")
        timeframe: "15min", "30min", or "1H"

    Returns:
        Regime name: "Low", "Normal", "High", or "Extreme"
    """
    data = get_asset_data(asset, timeframe)
    if data is None:
        return "Normal"  # Default

    # Get frequencies from long distributions (same for short)
    distributions = data.get("distributions_long", {})
    frequencies = distributions.get("regime_frequencies", {})

    if not frequencies:
        return "Normal"

    regimes = list(frequencies.keys())
    probs = [frequencies.get(r, 0.25) for r in regimes]

    # Normalize probabilities
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [0.25, 0.25, 0.25, 0.25]
        regimes = ["Low", "Normal", "High", "Extreme"]

    return np.random.choice(regimes, p=probs)


def sample_mfe_mae(
    asset: str,
    timeframe: str,
    regime: str,
    direction: str = "long"
) -> tuple[float, float]:
    """
    Sample MFE and MAE from the regime's distribution.

    Uses linear interpolation between percentiles for more realistic
    continuous values rather than just discrete percentile values.

    Args:
        asset: Symbol (e.g., "NQ")
        timeframe: "15min", "30min", or "1H"
        regime: "Low", "Normal", "High", or "Extreme"
        direction: "long" or "short"

    Returns:
        Tuple of (mfe, mae) in points
    """
    data = get_asset_data(asset, timeframe)
    if data is None:
        # Return reasonable defaults
        return (20.0, 10.0) if direction == "long" else (20.0, 10.0)

    dist_key = f"distributions_{direction}"
    distributions = data.get(dist_key, {})

    mfe_dist = distributions.get("mfe", {}).get(regime, [10, 15, 20, 30, 40, 50, 60, 80])
    mae_dist = distributions.get("mae", {}).get(regime, [5, 8, 12, 18, 25, 35, 45, 60])
    percentiles = distributions.get("percentiles", [5, 10, 25, 50, 75, 90, 95, 99])

    # Sample from distributions using interpolation
    mfe = _sample_from_percentiles(mfe_dist, percentiles)
    mae = _sample_from_percentiles(mae_dist, percentiles)

    return (mfe, mae)


def _sample_from_percentiles(values: list[float], percentiles: list[int]) -> float:
    """
    Sample a value from a percentile distribution using interpolation.

    Generates a random percentile (0-100) and interpolates between
    the nearest known percentile values.
    """
    if not values or not percentiles:
        return 10.0  # Default

    # Random percentile
    p = np.random.uniform(0, 100)

    # Find bracketing percentiles
    for i in range(len(percentiles) - 1):
        if percentiles[i] <= p <= percentiles[i + 1]:
            # Linear interpolation
            t = (p - percentiles[i]) / (percentiles[i + 1] - percentiles[i])
            return values[i] + t * (values[i + 1] - values[i])

    # Edge cases
    if p < percentiles[0]:
        # Extrapolate below (linear from 0 to first percentile)
        return values[0] * (p / percentiles[0])
    else:
        # Above max percentile, return max value
        return values[-1]


def get_hit_probabilities(
    asset: str,
    timeframe: str,
    regime: str,
    stop_points: float,
    target_points: float,
    direction: str = "long"
) -> HitProbabilities:
    """
    Get pre-computed hit probabilities for a stop/target combination.

    Args:
        asset: Symbol (e.g., "NQ")
        timeframe: "15min", "30min", or "1H"
        regime: "Low", "Normal", "High", or "Extreme"
        stop_points: Stop loss in points
        target_points: Target in points
        direction: "long" or "short"

    Returns:
        HitProbabilities dataclass with target_first, stop_first, neither probabilities
    """
    data = get_asset_data(asset, timeframe)
    if data is None:
        # Return neutral probabilities
        return HitProbabilities(
            target_first=0.33,
            stop_first=0.33,
            neither=0.34,
            avg_time_exit_pnl=0.0
        )

    # Get hit matrix for this regime
    by_regime = data.get("by_regime", {})
    regime_data = by_regime.get(regime, {})
    matrix_key = f"hit_matrix_{direction}"
    hit_matrix = regime_data.get(matrix_key, {})

    if not hit_matrix:
        # Fall back to overall hit matrix
        hit_matrix = data.get(matrix_key, {})

    # Find closest stop level in matrix
    stop_key = _find_closest_key(hit_matrix, stop_points)
    if stop_key is None:
        return HitProbabilities(0.33, 0.33, 0.34, 0.0)

    stop_data = hit_matrix.get(str(stop_key), {})

    # Find closest target level
    target_key = _find_closest_key(stop_data, target_points)
    if target_key is None:
        return HitProbabilities(0.33, 0.33, 0.34, 0.0)

    probs = stop_data.get(str(target_key), {})

    return HitProbabilities(
        target_first=probs.get("target_first", 0.33),
        stop_first=probs.get("stop_first", 0.33),
        neither=probs.get("neither", 0.34),
        avg_time_exit_pnl=probs.get("avg_time_exit_pnl", 0.0)
    )


def _find_closest_key(d: dict, value: float) -> Optional[float]:
    """Find the closest numeric key in a dictionary to the given value."""
    if not d:
        return None

    try:
        keys = [float(k) for k in d.keys() if k not in ["error"]]
        if not keys:
            return None
        return min(keys, key=lambda k: abs(k - value))
    except (ValueError, TypeError):
        return None


def sample_random_timeframe() -> str:
    """Randomly sample a timeframe for trade simulation."""
    return np.random.choice(["15min", "30min", "1H"])


def determine_trade_outcome(
    asset: str,
    stop_points: float,
    target_points: float,
    direction: str = "long",
    regime_override: Optional[str] = None
) -> tuple[str, float, str, str]:
    """
    Determine trade outcome using cached distributions.

    This is the main function called by the Monte Carlo simulation
    for each trade when using realistic price paths.

    Args:
        asset: Symbol (e.g., "NQ")
        stop_points: Stop loss in points
        target_points: Target in points
        direction: "long" or "short"
        regime_override: Force specific regime (optional)

    Returns:
        Tuple of (outcome, pnl_points, regime, timeframe)
        outcome: "target", "stop", or "time_exit"
        pnl_points: P&L in points (positive for win, negative for loss)
        regime: The volatility regime that was sampled
        timeframe: The timeframe that was sampled
    """
    # Sample timeframe
    timeframe = sample_random_timeframe()

    # Sample or use override regime
    if regime_override:
        regime = regime_override
    else:
        regime = sample_regime(asset, timeframe)

    # Sample MFE/MAE for this regime
    mfe, mae = sample_mfe_mae(asset, timeframe, regime, direction)

    # Determine outcome based on MFE/MAE vs stop/target
    if mae >= stop_points and mfe >= target_points:
        # Both could be hit - use probability matrix
        probs = get_hit_probabilities(
            asset, timeframe, regime, stop_points, target_points, direction
        )

        # Sample outcome
        outcomes = ["target", "stop", "time_exit"]
        probabilities = [probs.target_first, probs.stop_first, probs.neither]

        # Normalize
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [0.33, 0.33, 0.34]

        outcome = np.random.choice(outcomes, p=probabilities)

        if outcome == "target":
            pnl = target_points
        elif outcome == "stop":
            pnl = -stop_points
        else:
            pnl = probs.avg_time_exit_pnl

    elif mae >= stop_points:
        # Stop hit, target not reached
        outcome = "stop"
        pnl = -stop_points

    elif mfe >= target_points:
        # Target reached, stop not hit
        outcome = "target"
        pnl = target_points

    else:
        # Neither hit - time exit
        outcome = "time_exit"
        # Use average time exit P&L from probabilities
        probs = get_hit_probabilities(
            asset, timeframe, regime, stop_points, target_points, direction
        )
        pnl = probs.avg_time_exit_pnl

    return (outcome, pnl, regime, timeframe)


def get_regime_statistics(asset: str) -> dict:
    """
    Get regime statistics for display in UI.

    Returns summary info about the asset's regime distributions
    for use in Streamlit displays.
    """
    cache = get_cache()
    assets = cache.get("assets", {})

    if asset not in assets:
        return {"error": f"Asset {asset} not found in cache"}

    stats = {
        "asset": asset,
        "timeframes": {}
    }

    for timeframe in ["15min", "30min", "1H"]:
        data = assets[asset].get(timeframe)
        if data is None:
            continue

        tf_stats = {
            "total_candles": data.get("total_candles", 0),
            "regime_frequencies": data.get("distributions_long", {}).get("regime_frequencies", {}),
        }

        # Get MFE/MAE medians per regime
        distributions = data.get("distributions_long", {})
        mfe = distributions.get("mfe", {})
        mae = distributions.get("mae", {})
        percentiles = distributions.get("percentiles", [5, 10, 25, 50, 75, 90, 95, 99])

        # Find median index (50th percentile)
        median_idx = percentiles.index(50) if 50 in percentiles else 3

        tf_stats["mfe_medians"] = {
            regime: values[median_idx] if len(values) > median_idx else 0
            for regime, values in mfe.items()
        }
        tf_stats["mae_medians"] = {
            regime: values[median_idx] if len(values) > median_idx else 0
            for regime, values in mae.items()
        }

        stats["timeframes"][timeframe] = tf_stats

    return stats


# Convenience function to check if cache exists
def cache_exists(cache_path: Optional[Path] = None) -> bool:
    """Check if the regime cache file exists."""
    if cache_path is None:
        cache_path = DEFAULT_CACHE_PATH
    return cache_path.exists()


if __name__ == "__main__":
    # Test the cache loader
    if not cache_exists():
        print(f"Cache not found at {DEFAULT_CACHE_PATH}")
        print("Run 'python build_regime_cache.py' to create it.")
    else:
        load_regime_cache()
        print("Available assets:", get_available_assets())

        # Test sampling
        for _ in range(5):
            outcome, pnl, regime, tf = determine_trade_outcome(
                asset="NQ",
                stop_points=15,
                target_points=30,
                direction="long"
            )
            print(f"  {tf} {regime}: {outcome} -> {pnl:+.1f} pts")
