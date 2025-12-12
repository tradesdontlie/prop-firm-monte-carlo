# Prop Firm Monte Carlo Simulator

A Monte Carlo simulation tool for evaluating prop firm trading strategies and risk management. Simulates thousands of trading scenarios to estimate survival rates, drawdown risks, and profit target probabilities.

## Features

- **Prop Firm Rules**: Configurable drawdown limits (daily, trailing, static)
- **Trailing Drawdown Types**: End-of-day or real-time trailing mechanics
- **Realistic Price Paths**: MFE/MAE-based simulation using historical futures data
- **Basis Points Support**: Define stop/target in points or basis points of asset price
- **Multiple Assets**: NQ, ES, MNQ, MES, GC, CL, YM, MYM
- **Contract Scaling**: Tier-based position sizing as equity grows
- **Detailed Analytics**: Survival rates, risk of ruin, time to target/breach distributions

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run monte_carlo.py
```

## Files

| File | Purpose |
|------|---------|
| `monte_carlo.py` | Main Streamlit app with simulation engine |
| `regime_cache.py` | Loads pre-computed MFE/MAE distributions (runtime) |
| `build_regime_cache.py` | CLI to build cache from parquet data (offline) |
| `historical_data_processor.py` | Polars-based data processing (build-time only) |

## Realistic Price Path Simulation

Instead of simple win/loss outcomes, the simulator can use historical MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion) distributions to model realistic trade outcomes.

### Building the Regime Cache

If you have historical futures data in parquet format:

```bash
# Install polars for data processing
pip install polars

# Build the cache (uses 2020+ data by default)
python build_regime_cache.py --data-dir "/path/to/parquet/files"

# Use all historical data
python build_regime_cache.py --data-dir "/path/to/data" --start-date all
```

The cache is saved to `~/.queen_streamlit/regime_cache.json` and contains:
- MFE/MAE percentile distributions by volatility regime
- Hit probability matrices for stop/target combinations
- Regime frequency distributions

### Volatility Regimes

Trades are classified into four volatility regimes based on rolling ATR%:
- **Low**: <= 25th percentile
- **Normal**: 25th-75th percentile
- **High**: 75th-90th percentile
- **Extreme**: > 90th percentile

## Basis Points

Stop loss and targets can be defined in basis points (bps) relative to asset price:

| Asset | Price | 50 bps = |
|-------|-------|----------|
| NQ | 21,000 | 105 points |
| ES | 6,000 | 30 points |
| YM | 44,000 | 220 points |

This normalizes risk across different assets.

## Edge Calculation

```
Edge % = (WinRate × AvgWin - LossRate × AvgLoss) / AvgLoss × 100
```

A positive edge doesn't guarantee survival - variance can still cause drawdown breaches before the edge materializes. That's what the Monte Carlo reveals.

## License

MIT
