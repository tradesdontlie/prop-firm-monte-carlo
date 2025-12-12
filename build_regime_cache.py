#!/usr/bin/env python3
"""
Build Regime Cache CLI

This script processes historical parquet data and builds a JSON cache
containing pre-computed MFE/MAE distributions and hit probability matrices.

The resulting JSON file is used by the Monte Carlo simulator at runtime.
Once built, the parquet files are no longer needed.

Usage:
    python build_regime_cache.py --data-dir "/path/to/parquet/files"

    # Process specific assets only
    python build_regime_cache.py --data-dir "/path/to/data" --assets NQ,ES,MNQ

    # Specify output location
    python build_regime_cache.py --data-dir "/path/to/data" --output ./my_cache.json

    # Process all assets
    python build_regime_cache.py --data-dir "/path/to/data" --assets all
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Import the data processor
from historical_data_processor import (
    build_asset_data,
    get_available_assets,
    TIMEFRAMES,
    STOP_LEVELS,
    TARGET_LEVELS
)


DEFAULT_CACHE_DIR = Path.home() / ".queen_streamlit"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "regime_cache.json"


def build_full_cache(
    data_dir: Path,
    assets: list[str],
    timeframes: list[str],
    output_path: Path,
    start_date: str = None
) -> dict:
    """
    Build the complete regime cache for all specified assets and timeframes.

    Args:
        data_dir: Directory containing parquet files
        assets: List of asset symbols to process
        timeframes: List of timeframes ("15min", "30min", "1H")
        output_path: Where to save the JSON cache
        start_date: Optional start date filter (e.g., "2020-01-01")

    Returns:
        The complete cache dictionary
    """
    cache = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "data_dir": str(data_dir),
            "assets": assets,
            "timeframes": timeframes,
            "stop_levels": STOP_LEVELS,
            "target_levels": TARGET_LEVELS,
            "start_date": start_date,
            "version": "1.1.0"
        },
        "assets": {}
    }

    total = len(assets) * len(timeframes)
    current = 0

    for asset in assets:
        cache["assets"][asset] = {}

        for timeframe in timeframes:
            current += 1
            print(f"\n[{current}/{total}] Processing {asset} {timeframe}...")

            try:
                data = build_asset_data(asset, data_dir, timeframe, start_date)
                cache["assets"][asset][timeframe] = data
                print(f"  -> {data['total_candles']} candles processed")
            except FileNotFoundError as e:
                print(f"  -> SKIPPED: {e}")
                cache["assets"][asset][timeframe] = None
            except Exception as e:
                print(f"  -> ERROR: {e}")
                cache["assets"][asset][timeframe] = {"error": str(e)}

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    print(f"\nSaving cache to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)

    # Report file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Cache saved: {size_mb:.2f} MB")

    return cache


def main():
    parser = argparse.ArgumentParser(
        description="Build regime cache from historical parquet data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all available assets
    python build_regime_cache.py --data-dir "/path/to/data" --assets all

    # Process specific assets
    python build_regime_cache.py --data-dir "/path/to/data" --assets NQ,ES,MNQ,MES

    # Custom output location
    python build_regime_cache.py --data-dir "/path/to/data" --output ./cache.json
        """
    )

    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        required=True,
        help="Directory containing parquet files"
    )

    parser.add_argument(
        "--assets", "-a",
        type=str,
        default="all",
        help="Comma-separated list of assets or 'all' (default: all)"
    )

    parser.add_argument(
        "--timeframes", "-t",
        type=str,
        default="15min,30min,1H",
        help="Comma-separated list of timeframes (default: 15min,30min,1H)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_CACHE_FILE,
        help=f"Output JSON file path (default: {DEFAULT_CACHE_FILE})"
    )

    parser.add_argument(
        "--list-assets",
        action="store_true",
        help="List available assets and exit"
    )

    parser.add_argument(
        "--start-date", "-s",
        type=str,
        default="2020-01-01",
        help="Start date for data filter (default: 2020-01-01). Use 'all' for no filter."
    )

    args = parser.parse_args()

    # Validate data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Get available assets
    available_assets = get_available_assets(args.data_dir)

    if args.list_assets:
        print("Available assets:")
        for asset in available_assets:
            print(f"  - {asset}")
        sys.exit(0)

    if not available_assets:
        print(f"Error: No parquet files found in {args.data_dir}")
        sys.exit(1)

    # Parse assets
    if args.assets.lower() == "all":
        assets = available_assets
    else:
        assets = [a.strip().upper() for a in args.assets.split(",")]
        # Validate assets exist
        for asset in assets:
            if asset not in available_assets:
                print(f"Warning: Asset '{asset}' not found in data directory")

    # Parse timeframes
    timeframes = [t.strip() for t in args.timeframes.split(",")]
    valid_timeframes = ["15min", "30min", "1H"]
    for tf in timeframes:
        if tf not in valid_timeframes:
            print(f"Error: Invalid timeframe '{tf}'. Use: {valid_timeframes}")
            sys.exit(1)

    # Handle start date
    start_date = None if args.start_date.lower() == "all" else args.start_date

    # Print summary
    print("=" * 60)
    print("REGIME CACHE BUILDER")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output file: {args.output}")
    print(f"Start date filter: {start_date or 'None (using all data)'}")
    print(f"Assets ({len(assets)}): {', '.join(assets)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Total combinations: {len(assets) * len(timeframes)}")
    print("=" * 60)

    # Confirm
    response = input("\nProceed? [Y/n]: ").strip().lower()
    if response and response != "y":
        print("Cancelled.")
        sys.exit(0)

    # Build cache
    try:
        cache = build_full_cache(
            data_dir=args.data_dir,
            assets=assets,
            timeframes=timeframes,
            output_path=args.output,
            start_date=start_date
        )

        # Print summary stats
        print("\n" + "=" * 60)
        print("CACHE BUILD COMPLETE")
        print("=" * 60)
        print(f"Output: {args.output}")

        # Count successful assets
        successful = sum(
            1 for asset in cache["assets"].values()
            for tf in asset.values()
            if tf is not None and "error" not in tf
        )
        print(f"Successful: {successful}/{len(assets) * len(timeframes)}")

    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
