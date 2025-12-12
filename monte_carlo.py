"""
Monte Carlo Day Trading Risk Management System
A Streamlit application for simulating trading scenarios against prop firm rules.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from datetime import datetime
import io

# =============================================================================
# CONFIGURATION & DATA MODELS
# =============================================================================

class TrailingDrawdownType(Enum):
    STATIC = "Static"
    END_OF_DAY = "End-of-Day Trailing"
    REAL_TIME = "Real-Time Trailing"


@dataclass
class AssetConfig:
    """Configuration for a tradeable asset"""
    symbol: str
    name: str
    tick_value: float
    ticks_per_point: int
    typical_price: float  # Reference price for basis point calculations

    @property
    def point_value(self) -> float:
        return self.tick_value * self.ticks_per_point

    def bps_to_points(self, bps: float) -> float:
        """Convert basis points to price points"""
        return self.typical_price * (bps / 10000)

    def points_to_bps(self, points: float) -> float:
        """Convert price points to basis points"""
        return (points / self.typical_price) * 10000


# Asset specifications (typical_price as of late 2024)
ASSETS = {
    'NQ': AssetConfig('NQ', 'E-mini NASDAQ-100', 5.00, 4, 21000),
    'ES': AssetConfig('ES', 'E-mini S&P 500', 12.50, 4, 6000),
    'MNQ': AssetConfig('MNQ', 'Micro E-mini NASDAQ-100', 0.50, 4, 21000),
    'MES': AssetConfig('MES', 'Micro E-mini S&P 500', 1.25, 4, 6000),
    'GC': AssetConfig('GC', 'Gold Futures', 10.00, 10, 2650),
    'CL': AssetConfig('CL', 'Crude Oil Futures', 10.00, 100, 75),
    'YM': AssetConfig('YM', 'E-mini Dow', 5.00, 1, 44000),
    'MYM': AssetConfig('MYM', 'Micro E-mini Dow', 0.50, 1, 44000),
}


@dataclass
class ContractScalingTier:
    """Defines max contracts for an equity tier"""
    min_equity: float
    max_equity: float
    max_contracts: int


@dataclass
class ExitLevel:
    """Configuration for a take profit level"""
    points: float
    portion: float  # Percentage of position (0.0 to 1.0)
    probability: float  # Probability of hitting this level given entry


@dataclass
class AccountConfig:
    """Account and prop firm configuration"""
    starting_balance: float = 50000.0
    max_drawdown: float = 2500.0
    daily_drawdown_limit: Optional[float] = None
    trailing_type: TrailingDrawdownType = TrailingDrawdownType.END_OF_DAY
    trailing_stop_point: float = 0.0  # Profit level where trailing stops
    profit_target: float = 3000.0  # Payout eligibility target
    contract_scaling: List[ContractScalingTier] = field(default_factory=list)


@dataclass
class TradeConfig:
    """Trade strategy configuration"""
    risk_per_trade: float = 100.0  # Dollar amount
    risk_is_percentage: bool = False  # If True, risk_per_trade is %
    trades_per_day_min: int = 1
    trades_per_day_max: int = 3
    win_rate: float = 0.50
    avg_win_points: float = 10.0
    avg_loss_points: float = 8.0
    stop_loss_points: Optional[float] = None
    exit_levels: List[ExitLevel] = field(default_factory=list)
    use_partial_exits: bool = False


@dataclass
class RealisticTradeConfig:
    """Configuration for realistic MFE/MAE-based price path simulation"""
    use_realistic_paths: bool = False
    # candle_timeframe is NOT configurable - randomly sampled from [15min, 30min, 1H] per trade
    trade_direction: str = "random"  # "long", "short", or "random" (50/50)
    regime_override: Optional[str] = None  # Force specific regime or None for random
    break_even_trigger: float = 0.0  # Points in profit to move stop to BE (0 = disabled)
    target_points: Optional[float] = None  # Target in points (if None, uses avg_win_points)


@dataclass
class SimulationConfig:
    """Simulation parameters"""
    num_simulations: int = 1000
    num_days: int = 60  # Trading days
    random_seed: Optional[int] = None
    distribution: str = "normal"  # normal, lognormal


# =============================================================================
# PROP FIRM PRESETS
# =============================================================================

PROP_FIRM_PRESETS = {
    "Custom": {
        "starting_balance": 50000,
        "max_drawdown": 2500,
        "daily_drawdown_limit": None,
        "trailing_type": TrailingDrawdownType.END_OF_DAY,
        "trailing_stop_point": 0,
        "profit_target": 3000,
    },
    "Apex 50K": {
        "starting_balance": 50000,
        "max_drawdown": 2500,
        "daily_drawdown_limit": None,
        "trailing_type": TrailingDrawdownType.END_OF_DAY,
        "trailing_stop_point": 2500,
        "profit_target": 3000,
    },
    "Apex 100K": {
        "starting_balance": 100000,
        "max_drawdown": 3000,
        "daily_drawdown_limit": None,
        "trailing_type": TrailingDrawdownType.END_OF_DAY,
        "trailing_stop_point": 3000,
        "profit_target": 6000,
    },
    "Apex 150K": {
        "starting_balance": 150000,
        "max_drawdown": 5000,
        "daily_drawdown_limit": None,
        "trailing_type": TrailingDrawdownType.END_OF_DAY,
        "trailing_stop_point": 5000,
        "profit_target": 9000,
    },
    "Apex 250K": {
        "starting_balance": 250000,
        "max_drawdown": 6500,
        "daily_drawdown_limit": None,
        "trailing_type": TrailingDrawdownType.END_OF_DAY,
        "trailing_stop_point": 6500,
        "profit_target": 15000,
    },
    "Apex 300K": {
        "starting_balance": 300000,
        "max_drawdown": 7500,
        "daily_drawdown_limit": None,
        "trailing_type": TrailingDrawdownType.END_OF_DAY,
        "trailing_stop_point": 7500,
        "profit_target": 20000,
    },
    "Topstep 50K": {
        "starting_balance": 50000,
        "max_drawdown": 2000,
        "daily_drawdown_limit": 1000,
        "trailing_type": TrailingDrawdownType.END_OF_DAY,
        "trailing_stop_point": 2000,
        "profit_target": 3000,
    },
    "Topstep 100K": {
        "starting_balance": 100000,
        "max_drawdown": 3000,
        "daily_drawdown_limit": 2000,
        "trailing_type": TrailingDrawdownType.END_OF_DAY,
        "trailing_stop_point": 3000,
        "profit_target": 6000,
    },
    "Topstep 150K": {
        "starting_balance": 150000,
        "max_drawdown": 4500,
        "daily_drawdown_limit": 3000,
        "trailing_type": TrailingDrawdownType.END_OF_DAY,
        "trailing_stop_point": 4500,
        "profit_target": 9000,
    },
    "Earn2Trade 50K": {
        "starting_balance": 50000,
        "max_drawdown": 2000,
        "daily_drawdown_limit": None,
        "trailing_type": TrailingDrawdownType.END_OF_DAY,
        "trailing_stop_point": 2000,
        "profit_target": 3000,
    },
    "Earn2Trade 100K": {
        "starting_balance": 100000,
        "max_drawdown": 3500,
        "daily_drawdown_limit": None,
        "trailing_type": TrailingDrawdownType.END_OF_DAY,
        "trailing_stop_point": 3500,
        "profit_target": 6000,
    },
}


# =============================================================================
# MONTE CARLO SIMULATION ENGINE
# =============================================================================

class MonteCarloEngine:
    """
    Monte Carlo simulation engine for day trading scenarios.
    Accurately models trailing drawdown, high water marks, and prop firm rules.
    """

    def __init__(
        self,
        account_config: AccountConfig,
        trade_config: TradeConfig,
        asset: AssetConfig,
        sim_config: SimulationConfig,
        realistic_config: Optional[RealisticTradeConfig] = None
    ):
        self.account = account_config
        self.trade = trade_config
        self.asset = asset
        self.sim = sim_config
        self.realistic = realistic_config or RealisticTradeConfig()

        # Load regime cache if using realistic paths
        self._regime_cache_loaded = False
        if self.realistic.use_realistic_paths:
            self._load_regime_cache()

        if sim_config.random_seed is not None:
            np.random.seed(sim_config.random_seed)

    def _load_regime_cache(self):
        """Load the pre-computed regime cache for realistic trade simulation"""
        try:
            from regime_cache import load_regime_cache, cache_exists
            if cache_exists():
                load_regime_cache()
                self._regime_cache_loaded = True
            else:
                st.warning("Regime cache not found. Using legacy simulation mode.")
                self.realistic.use_realistic_paths = False
        except ImportError:
            st.warning("regime_cache module not found. Using legacy simulation mode.")
            self.realistic.use_realistic_paths = False
        except Exception as e:
            st.warning(f"Error loading regime cache: {e}. Using legacy simulation mode.")
            self.realistic.use_realistic_paths = False

    def calculate_position_size(self, current_equity: float) -> int:
        """Calculate number of contracts based on risk settings"""
        if self.trade.risk_is_percentage:
            risk_amount = current_equity * (self.trade.risk_per_trade / 100)
        else:
            risk_amount = self.trade.risk_per_trade

        if self.trade.stop_loss_points and self.trade.stop_loss_points > 0:
            contracts = int(risk_amount / (self.trade.stop_loss_points * self.asset.point_value))
        else:
            # Use average loss as proxy for stop loss
            contracts = int(risk_amount / (self.trade.avg_loss_points * self.asset.point_value))

        # Apply contract scaling if defined
        if self.account.contract_scaling:
            max_allowed = 1
            for tier in self.account.contract_scaling:
                if tier.min_equity <= current_equity < tier.max_equity:
                    max_allowed = tier.max_contracts
                    break
            contracts = min(contracts, max_allowed)

        return max(1, contracts)

    def generate_trade_outcome(self, num_contracts: int) -> float:
        """Generate P&L for a single trade"""
        is_winner = np.random.random() < self.trade.win_rate

        if self.sim.distribution == "lognormal":
            if is_winner:
                points = np.random.lognormal(
                    np.log(self.trade.avg_win_points),
                    0.3
                )
            else:
                points = -np.random.lognormal(
                    np.log(self.trade.avg_loss_points),
                    0.3
                )
        else:  # normal distribution
            if is_winner:
                points = max(0.25, np.random.normal(
                    self.trade.avg_win_points,
                    self.trade.avg_win_points * 0.3
                ))
            else:
                points = -max(0.25, np.random.normal(
                    self.trade.avg_loss_points,
                    self.trade.avg_loss_points * 0.3
                ))

        return points * num_contracts * self.asset.point_value

    def generate_realistic_trade_outcome(self, num_contracts: int) -> Tuple[float, str, dict]:
        """
        Generate P&L for a single trade using realistic MFE/MAE distributions.

        Returns:
            Tuple of (pnl_dollars, exit_reason, metadata)
            exit_reason: "target", "stop", "time_exit", or "break_even"
            metadata: dict with regime, timeframe, direction info
        """
        from regime_cache import determine_trade_outcome

        # Determine trade direction
        if self.realistic.trade_direction == "random":
            direction = np.random.choice(["long", "short"])
        else:
            direction = self.realistic.trade_direction

        # Get stop and target points
        stop_points = self.trade.stop_loss_points or self.trade.avg_loss_points
        target_points = self.realistic.target_points or self.trade.avg_win_points

        # Map asset symbol for cache lookup (handle MNQ -> NQ, MES -> ES, etc.)
        cache_asset = self._get_cache_asset_symbol()

        # Get outcome from cached distributions
        outcome, pnl_points, regime, timeframe = determine_trade_outcome(
            asset=cache_asset,
            stop_points=stop_points,
            target_points=target_points,
            direction=direction,
            regime_override=self.realistic.regime_override
        )

        # Handle break-even logic
        if self.realistic.break_even_trigger > 0 and outcome == "stop":
            # Check if we would have hit BE trigger first
            # For simplicity: if pnl_points was positive before reversal,
            # simulate the BE chance based on MFE
            from regime_cache import sample_mfe_mae
            mfe, mae = sample_mfe_mae(cache_asset, timeframe, regime, direction)

            if mfe >= self.realistic.break_even_trigger:
                # Trade hit BE trigger before stop
                # 50% chance we caught the BE move (simplified)
                if np.random.random() < 0.5:
                    outcome = "break_even"
                    pnl_points = 0.0

        # Convert points to dollars
        pnl_dollars = pnl_points * num_contracts * self.asset.point_value

        metadata = {
            "direction": direction,
            "regime": regime,
            "timeframe": timeframe,
            "stop_points": stop_points,
            "target_points": target_points,
        }

        return pnl_dollars, outcome, metadata

    def _get_cache_asset_symbol(self) -> str:
        """Map micro contracts to their full-size equivalents for cache lookup"""
        symbol_map = {
            "MNQ": "NQ",
            "MES": "ES",
            "MYM": "YM",
            # Add more mappings as needed
        }
        return symbol_map.get(self.asset.symbol, self.asset.symbol)

    def run_single_simulation(self) -> Dict:
        """
        Run a single simulation and return detailed results.

        Returns dict with:
        - equity_curve: daily equity values
        - high_water_marks: daily HWM values
        - drawdown_floor: daily drawdown floor values
        - trade_log: list of trade details
        - breached: bool if account was breached
        - breach_day: day of breach (if any)
        - final_equity: ending equity
        - max_drawdown_experienced: worst drawdown hit
        - hit_profit_target: bool if profit target reached
        """
        equity = self.account.starting_balance
        high_water_mark = self.account.starting_balance

        # Initial drawdown floor
        drawdown_floor = self.account.starting_balance - self.account.max_drawdown
        trailing_locked = False

        equity_curve = [equity]
        hwm_curve = [high_water_mark]
        floor_curve = [drawdown_floor]
        trade_log = []

        breached = False
        breach_day = None
        max_dd_experienced = 0
        hit_target = False
        target_day = None
        daily_loss_breach = False

        for day in range(self.sim.num_days):
            # Stop simulation if breached OR hit profit target (both are terminal states)
            if breached or hit_target:
                equity_curve.append(equity)
                hwm_curve.append(high_water_mark)
                floor_curve.append(drawdown_floor)
                continue

            # Track daily start equity for daily drawdown limit
            day_start_equity = equity

            # Generate number of trades for this day
            num_trades = np.random.randint(
                self.trade.trades_per_day_min,
                self.trade.trades_per_day_max + 1
            )

            daily_pnl = 0

            for trade_num in range(num_trades):
                # Calculate position size
                contracts = self.calculate_position_size(equity)

                # Generate trade outcome (realistic or legacy)
                exit_reason = "legacy"
                trade_metadata = {}

                if self.realistic.use_realistic_paths and self._regime_cache_loaded:
                    pnl, exit_reason, trade_metadata = self.generate_realistic_trade_outcome(contracts)
                else:
                    pnl = self.generate_trade_outcome(contracts)

                # Update equity
                equity += pnl
                daily_pnl += pnl

                # Real-time trailing: update floor tick-by-tick
                if (self.account.trailing_type == TrailingDrawdownType.REAL_TIME
                    and not trailing_locked):
                    if equity > high_water_mark:
                        high_water_mark = equity
                        # Update floor based on new HWM
                        new_floor = high_water_mark - self.account.max_drawdown
                        drawdown_floor = max(drawdown_floor, new_floor)

                        # Check if trailing should lock
                        profit = high_water_mark - self.account.starting_balance
                        if profit >= self.account.trailing_stop_point > 0:
                            trailing_locked = True
                            drawdown_floor = self.account.starting_balance

                # Log trade
                trade_entry = {
                    'day': day + 1,
                    'trade_num': trade_num + 1,
                    'contracts': contracts,
                    'pnl': pnl,
                    'equity': equity,
                    'hwm': high_water_mark,
                    'floor': drawdown_floor,
                    'exit_reason': exit_reason,
                }
                # Add realistic mode metadata if available
                if trade_metadata:
                    trade_entry.update(trade_metadata)
                trade_log.append(trade_entry)

                # Check for breach after each trade (real-time)
                if equity <= drawdown_floor:
                    breached = True
                    breach_day = day + 1
                    break

                # Check daily drawdown limit
                if self.account.daily_drawdown_limit:
                    if day_start_equity - equity >= self.account.daily_drawdown_limit:
                        daily_loss_breach = True
                        break  # Stop trading for the day

            # End-of-day trailing update
            if (self.account.trailing_type == TrailingDrawdownType.END_OF_DAY
                and not trailing_locked and not breached):
                if equity > high_water_mark:
                    high_water_mark = equity
                    new_floor = high_water_mark - self.account.max_drawdown
                    drawdown_floor = max(drawdown_floor, new_floor)

                    # Check if trailing should lock
                    profit = high_water_mark - self.account.starting_balance
                    if profit >= self.account.trailing_stop_point > 0:
                        trailing_locked = True
                        drawdown_floor = self.account.starting_balance

            # Track maximum drawdown experienced
            current_dd = high_water_mark - equity
            max_dd_experienced = max(max_dd_experienced, current_dd)

            # Check for breach at end of day
            if equity <= drawdown_floor and not breached:
                breached = True
                breach_day = day + 1

            # Check profit target (terminal state - stops simulation)
            if equity - self.account.starting_balance >= self.account.profit_target and not hit_target:
                hit_target = True
                target_day = day + 1

            equity_curve.append(equity)
            hwm_curve.append(high_water_mark)
            floor_curve.append(drawdown_floor)

            daily_loss_breach = False  # Reset for next day

        return {
            'equity_curve': equity_curve,
            'high_water_marks': hwm_curve,
            'drawdown_floor': floor_curve,
            'trade_log': trade_log,
            'breached': breached,
            'breach_day': breach_day,
            'final_equity': equity,
            'max_drawdown_experienced': max_dd_experienced,
            'hit_profit_target': hit_target,
            'target_day': target_day,
            'total_trades': len(trade_log),
            'winning_trades': sum(1 for t in trade_log if t['pnl'] > 0),
        }

    def run_simulations(self, progress_callback=None) -> Dict:
        """
        Run all Monte Carlo simulations.

        Returns:
        - all_results: list of individual simulation results
        - summary_stats: aggregated statistics
        """
        all_results = []

        for i in range(self.sim.num_simulations):
            result = self.run_single_simulation()
            all_results.append(result)

            if progress_callback and i % 100 == 0:
                progress_callback(i / self.sim.num_simulations)

        if progress_callback:
            progress_callback(1.0)

        # Calculate summary statistics
        summary = self.calculate_summary_stats(all_results)

        return {
            'all_results': all_results,
            'summary_stats': summary
        }

    def calculate_summary_stats(self, results: List[Dict]) -> Dict:
        """Calculate comprehensive summary statistics"""
        n = len(results)

        # Account survival analysis
        breached_count = sum(1 for r in results if r['breached'])
        survival_rate = 1 - (breached_count / n)

        breach_days = [r['breach_day'] for r in results if r['breached']]

        # Final equity analysis
        final_equities = [r['final_equity'] for r in results]

        # Profit target analysis
        hit_target_count = sum(1 for r in results if r['hit_profit_target'])

        # Win rate analysis
        all_win_rates = []
        for r in results:
            if r['total_trades'] > 0:
                all_win_rates.append(r['winning_trades'] / r['total_trades'])

        # Max drawdown analysis
        max_drawdowns = [r['max_drawdown_experienced'] for r in results]

        # Calculate percentiles
        def percentiles(data, percs=[10, 25, 50, 75, 90]):
            if not data:
                return {p: 0 for p in percs}
            return {p: np.percentile(data, p) for p in percs}

        # Daily returns for VaR/CVaR calculation
        daily_returns = []
        for r in results:
            curve = r['equity_curve']
            for i in range(1, len(curve)):
                daily_returns.append(curve[i] - curve[i-1])

        # Value at Risk (95th and 99th percentile losses)
        var_95 = np.percentile(daily_returns, 5) if daily_returns else 0
        var_99 = np.percentile(daily_returns, 1) if daily_returns else 0

        # Conditional VaR (expected loss beyond VaR)
        losses_beyond_var95 = [r for r in daily_returns if r <= var_95]
        cvar_95 = np.mean(losses_beyond_var95) if losses_beyond_var95 else 0

        # Consecutive loss analysis
        max_losing_streaks = []
        for r in results:
            max_streak = 0
            current_streak = 0
            for t in r['trade_log']:
                if t['pnl'] < 0:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            max_losing_streaks.append(max_streak)

        # Profit factor calculation
        total_gross_profit = 0
        total_gross_loss = 0
        for r in results:
            for t in r['trade_log']:
                if t['pnl'] > 0:
                    total_gross_profit += t['pnl']
                else:
                    total_gross_loss += abs(t['pnl'])

        profit_factor = (total_gross_profit / total_gross_loss
                        if total_gross_loss > 0 else float('inf'))

        # Total trades and expected value
        total_trades = sum(r['total_trades'] for r in results)
        total_pnl = sum(r['final_equity'] - self.account.starting_balance for r in results)
        expected_value_per_trade = total_pnl / total_trades if total_trades > 0 else 0

        return {
            # Survival metrics
            'survival_rate': survival_rate,
            'failure_rate': 1 - survival_rate,
            'total_simulations': n,
            'accounts_breached': breached_count,
            'accounts_survived': n - breached_count,

            # Breach timing
            'mean_time_to_failure': np.mean(breach_days) if breach_days else None,
            'median_time_to_failure': np.median(breach_days) if breach_days else None,
            'breach_day_percentiles': percentiles(breach_days) if breach_days else None,

            # Profitability
            'hit_profit_target_rate': hit_target_count / n,
            'accounts_hit_target': hit_target_count,
            'profit_factor': profit_factor,
            'expected_value_per_trade': expected_value_per_trade,

            # Final equity
            'mean_final_equity': np.mean(final_equities),
            'median_final_equity': np.median(final_equities),
            'final_equity_percentiles': percentiles(final_equities),
            'final_equity_std': np.std(final_equities),

            # Win rate
            'mean_win_rate': np.mean(all_win_rates) if all_win_rates else 0,
            'win_rate_std': np.std(all_win_rates) if all_win_rates else 0,

            # Drawdown metrics
            'mean_max_drawdown': np.mean(max_drawdowns),
            'max_drawdown_percentiles': percentiles(max_drawdowns),

            # Risk metrics
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,

            # Streak analysis
            'mean_max_losing_streak': np.mean(max_losing_streaks),
            'max_losing_streak_ever': max(max_losing_streaks) if max_losing_streaks else 0,

            # Trade statistics
            'total_trades_all_sims': total_trades,
            'avg_trades_per_sim': total_trades / n,
        }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_equity_fan_chart(results: List[Dict], account_config: AccountConfig) -> go.Figure:
    """Create Monte Carlo equity curve fan chart with percentile bands"""

    # Get all equity curves and pad to same length
    max_len = max(len(r['equity_curve']) for r in results)
    curves = []
    for r in results:
        curve = r['equity_curve']
        if len(curve) < max_len:
            curve = curve + [curve[-1]] * (max_len - len(curve))
        curves.append(curve)

    curves = np.array(curves)
    days = list(range(max_len))

    # Calculate percentiles
    p10 = np.percentile(curves, 10, axis=0)
    p25 = np.percentile(curves, 25, axis=0)
    p50 = np.percentile(curves, 50, axis=0)
    p75 = np.percentile(curves, 75, axis=0)
    p90 = np.percentile(curves, 90, axis=0)

    fig = go.Figure()

    # Add percentile bands
    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(p90) + list(p10)[::-1],
        fill='toself',
        fillcolor='rgba(0, 100, 200, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='10th-90th Percentile',
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(p75) + list(p25)[::-1],
        fill='toself',
        fillcolor='rgba(0, 100, 200, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='25th-75th Percentile',
        showlegend=True
    ))

    # Add median line
    fig.add_trace(go.Scatter(
        x=days,
        y=p50,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Median (50th)'
    ))

    # Add starting balance line
    fig.add_hline(
        y=account_config.starting_balance,
        line_dash="dash",
        line_color="gray",
        annotation_text="Starting Balance"
    )

    # Add drawdown floor line (initial)
    fig.add_hline(
        y=account_config.starting_balance - account_config.max_drawdown,
        line_dash="dash",
        line_color="red",
        annotation_text="Initial Drawdown Floor"
    )

    # Add profit target line
    fig.add_hline(
        y=account_config.starting_balance + account_config.profit_target,
        line_dash="dash",
        line_color="green",
        annotation_text="Profit Target"
    )

    fig.update_layout(
        title="Monte Carlo Equity Curves",
        xaxis_title="Trading Day",
        yaxis_title="Account Equity ($)",
        hovermode='x unified',
        height=500
    )

    return fig


def create_survival_curve(results: List[Dict], num_days: int) -> go.Figure:
    """Create Kaplan-Meier style survival curve"""

    # Calculate survival probability at each day
    survival_probs = []
    n = len(results)

    for day in range(num_days + 1):
        survived = sum(1 for r in results
                      if not r['breached'] or r['breach_day'] > day)
        survival_probs.append(survived / n)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(num_days + 1)),
        y=survival_probs,
        mode='lines',
        line=dict(color='green', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 0, 0.1)',
        name='Survival Probability'
    ))

    # Add reference lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                  annotation_text="50% Survival")
    fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                  annotation_text="25% Survival")

    fig.update_layout(
        title="Account Survival Probability Over Time",
        xaxis_title="Trading Day",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05], tickformat='.0%'),
        height=400
    )

    return fig


def create_final_balance_distribution(results: List[Dict],
                                     account_config: AccountConfig) -> go.Figure:
    """Create histogram of final account balances"""

    final_equities = [r['final_equity'] for r in results]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=final_equities,
        nbinsx=50,
        marker_color='steelblue',
        opacity=0.7,
        name='Final Balance'
    ))

    # Add vertical lines for key thresholds
    fig.add_vline(
        x=account_config.starting_balance,
        line_dash="dash",
        line_color="gray",
        annotation_text="Starting"
    )

    fig.add_vline(
        x=account_config.starting_balance + account_config.profit_target,
        line_dash="dash",
        line_color="green",
        annotation_text="Target"
    )

    fig.update_layout(
        title="Distribution of Final Account Balances",
        xaxis_title="Final Equity ($)",
        yaxis_title="Number of Simulations",
        height=400
    )

    return fig


def create_drawdown_distribution(results: List[Dict]) -> go.Figure:
    """Create histogram of maximum drawdowns experienced"""

    max_drawdowns = [r['max_drawdown_experienced'] for r in results]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=max_drawdowns,
        nbinsx=40,
        marker_color='crimson',
        opacity=0.7,
        name='Max Drawdown'
    ))

    fig.update_layout(
        title="Distribution of Maximum Drawdowns",
        xaxis_title="Maximum Drawdown ($)",
        yaxis_title="Number of Simulations",
        height=400
    )

    return fig


def create_breach_timing_chart(results: List[Dict]) -> go.Figure:
    """Create chart showing when accounts typically breach"""

    breach_days = [r['breach_day'] for r in results if r['breached']]

    if not breach_days:
        fig = go.Figure()
        fig.add_annotation(
            text="No accounts breached in simulation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Breach Timing Distribution", height=400)
        return fig

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=breach_days,
        nbinsx=30,
        marker_color='red',
        opacity=0.7,
        name='Breach Day'
    ))

    fig.update_layout(
        title="When Do Accounts Breach?",
        xaxis_title="Trading Day",
        yaxis_title="Number of Breaches",
        height=400
    )

    return fig


def create_win_rate_distribution(results: List[Dict]) -> go.Figure:
    """Create histogram of realized win rates"""

    win_rates = []
    for r in results:
        if r['total_trades'] > 0:
            win_rates.append(r['winning_trades'] / r['total_trades'] * 100)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=win_rates,
        nbinsx=30,
        marker_color='teal',
        opacity=0.7,
        name='Win Rate'
    ))

    fig.update_layout(
        title="Distribution of Realized Win Rates",
        xaxis_title="Win Rate (%)",
        yaxis_title="Number of Simulations",
        height=400
    )

    return fig


def create_sample_paths_chart(results: List[Dict], account_config: AccountConfig,
                             num_paths: int = 50) -> go.Figure:
    """Show individual sample paths"""

    fig = go.Figure()

    # Sample paths to display
    sample_indices = np.random.choice(len(results), min(num_paths, len(results)),
                                      replace=False)

    for idx in sample_indices:
        curve = results[idx]['equity_curve']
        breached = results[idx]['breached']

        color = 'rgba(255, 0, 0, 0.3)' if breached else 'rgba(0, 150, 0, 0.3)'

        fig.add_trace(go.Scatter(
            x=list(range(len(curve))),
            y=curve,
            mode='lines',
            line=dict(color=color, width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add reference lines
    fig.add_hline(
        y=account_config.starting_balance,
        line_dash="dash",
        line_color="gray",
        annotation_text="Starting Balance"
    )

    fig.add_hline(
        y=account_config.starting_balance - account_config.max_drawdown,
        line_dash="dash",
        line_color="red",
        annotation_text="Breach Level"
    )

    fig.update_layout(
        title=f"Sample Equity Paths (n={num_paths})",
        xaxis_title="Trading Day",
        yaxis_title="Account Equity ($)",
        height=500
    )

    return fig


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_results_to_csv(results: List[Dict], summary: Dict) -> str:
    """Export simulation results to CSV format"""

    # Create summary dataframe
    summary_data = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                summary_data.append({
                    'Metric': f"{key}_{sub_key}",
                    'Value': sub_value
                })
        else:
            summary_data.append({
                'Metric': key,
                'Value': value
            })

    df = pd.DataFrame(summary_data)
    return df.to_csv(index=False)


def export_config_to_json(account_config: AccountConfig,
                          trade_config: TradeConfig,
                          sim_config: SimulationConfig,
                          asset_symbol: str) -> str:
    """Export configuration to JSON for reproducibility"""

    config = {
        'account': {
            'starting_balance': account_config.starting_balance,
            'max_drawdown': account_config.max_drawdown,
            'daily_drawdown_limit': account_config.daily_drawdown_limit,
            'trailing_type': account_config.trailing_type.value,
            'trailing_stop_point': account_config.trailing_stop_point,
            'profit_target': account_config.profit_target,
        },
        'trade': {
            'risk_per_trade': trade_config.risk_per_trade,
            'risk_is_percentage': trade_config.risk_is_percentage,
            'trades_per_day_min': trade_config.trades_per_day_min,
            'trades_per_day_max': trade_config.trades_per_day_max,
            'win_rate': trade_config.win_rate,
            'avg_win_points': trade_config.avg_win_points,
            'avg_loss_points': trade_config.avg_loss_points,
            'stop_loss_points': trade_config.stop_loss_points,
        },
        'simulation': {
            'num_simulations': sim_config.num_simulations,
            'num_days': sim_config.num_days,
            'random_seed': sim_config.random_seed,
            'distribution': sim_config.distribution,
        },
        'asset': asset_symbol,
        'exported_at': datetime.now().isoformat()
    }

    return json.dumps(config, indent=2)


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Monte Carlo Trading Simulator",
        page_icon="🎲",
        layout="wide"
    )

    # Initialize session state
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'simulation_summary' not in st.session_state:
        st.session_state.simulation_summary = None

    # Sidebar - Global Configuration
    with st.sidebar:
        st.title("🎲 Monte Carlo Simulator")
        st.markdown("---")

        # Prop Firm Preset Selection
        st.subheader("Prop Firm Preset")
        preset_name = st.selectbox(
            "Select Preset",
            options=list(PROP_FIRM_PRESETS.keys()),
            help="Pre-configured settings for popular prop firms"
        )
        preset = PROP_FIRM_PRESETS[preset_name]

        st.markdown("---")

        # Account Configuration
        st.subheader("Account Settings")

        starting_balance = st.number_input(
            "Starting Balance ($)",
            min_value=1000.0,
            max_value=1000000.0,
            value=float(preset['starting_balance']),
            step=1000.0
        )

        max_drawdown = st.number_input(
            "Max Drawdown ($)",
            min_value=100.0,
            max_value=100000.0,
            value=float(preset['max_drawdown']),
            step=100.0
        )

        daily_dd = st.checkbox(
            "Enable Daily Drawdown Limit",
            value=preset['daily_drawdown_limit'] is not None
        )
        daily_drawdown_limit = None
        if daily_dd:
            daily_drawdown_limit = st.number_input(
                "Daily Drawdown Limit ($)",
                min_value=100.0,
                value=float(preset['daily_drawdown_limit'] or 1000),
                step=100.0
            )

        trailing_type_options = [t.value for t in TrailingDrawdownType]
        trailing_type_str = st.selectbox(
            "Trailing Drawdown Type",
            options=trailing_type_options,
            index=trailing_type_options.index(preset['trailing_type'].value)
        )
        trailing_type = TrailingDrawdownType(trailing_type_str)

        trailing_stop_point = st.number_input(
            "Trailing Stops At Profit ($)",
            min_value=0.0,
            value=float(preset['trailing_stop_point']),
            step=100.0,
            help="Profit level where trailing drawdown locks. 0 = never locks."
        )

        profit_target = st.number_input(
            "Profit Target ($)",
            min_value=0.0,
            value=float(preset['profit_target']),
            step=500.0
        )

        st.markdown("---")

        # Asset Selection
        st.subheader("Asset Selection")
        asset_symbol = st.selectbox(
            "Futures Contract",
            options=list(ASSETS.keys()),
            format_func=lambda x: f"{x} - {ASSETS[x].name}"
        )
        asset = ASSETS[asset_symbol]

        st.info(f"**Point Value:** ${asset.point_value:.2f}")

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Trade Config",
        "🎲 Simulation",
        "📈 Results Dashboard",
        "⚖️ Strategy Comparison",
        "💾 Export"
    ])

    # Tab 1: Trade Configuration
    with tab1:
        st.header("Trade Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Risk Settings")

            risk_type = st.radio(
                "Risk Type",
                options=["Fixed Dollar", "Percentage of Equity"],
                horizontal=True
            )

            if risk_type == "Fixed Dollar":
                risk_per_trade = st.number_input(
                    "Risk Per Trade ($)",
                    min_value=10.0,
                    max_value=10000.0,
                    value=100.0,
                    step=10.0
                )
                risk_is_percentage = False
            else:
                risk_per_trade = st.number_input(
                    "Risk Per Trade (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )
                risk_is_percentage = True

            st.subheader("Trades Per Day")
            col_min, col_max = st.columns(2)
            with col_min:
                trades_min = st.number_input(
                    "Minimum",
                    min_value=1,
                    max_value=20,
                    value=1
                )
            with col_max:
                trades_max = st.number_input(
                    "Maximum",
                    min_value=trades_min,
                    max_value=20,
                    value=3
                )

        with col2:
            st.subheader("Win/Loss Parameters")

            win_rate = st.slider(
                "Win Rate (%)",
                min_value=20,
                max_value=80,
                value=50,
                step=1,
                help="Expected percentage of winning trades"
            ) / 100

            # Toggle between Points and Basis Points
            price_unit = st.radio(
                "Price Unit",
                options=["Points", "Basis Points"],
                horizontal=True,
                help=f"Basis points are relative to asset price. {asset.symbol} @ {asset.typical_price:,.0f}: 10 bps = {asset.bps_to_points(10):.1f} pts"
            )

            if price_unit == "Points":
                avg_win = st.number_input(
                    "Average Win (Points)",
                    min_value=0.5,
                    max_value=500.0,
                    value=10.0,
                    step=0.5
                )

                avg_loss = st.number_input(
                    "Average Loss (Points)",
                    min_value=0.5,
                    max_value=500.0,
                    value=8.0,
                    step=0.5
                )
            else:
                # Basis points input - convert to points
                avg_win_bps = st.number_input(
                    "Average Win (bps)",
                    min_value=1,
                    max_value=500,
                    value=50,
                    step=5,
                    help=f"1 bp = {asset.bps_to_points(1):.2f} pts on {asset.symbol}"
                )
                avg_win = asset.bps_to_points(avg_win_bps)
                st.caption(f"= {avg_win:.1f} points")

                avg_loss_bps = st.number_input(
                    "Average Loss (bps)",
                    min_value=1,
                    max_value=500,
                    value=40,
                    step=5,
                    help=f"1 bp = {asset.bps_to_points(1):.2f} pts on {asset.symbol}"
                )
                avg_loss = asset.bps_to_points(avg_loss_bps)
                st.caption(f"= {avg_loss:.1f} points")

            use_stop = st.checkbox("Define Fixed Stop Loss", value=False)
            stop_loss = None
            if use_stop:
                if price_unit == "Points":
                    stop_loss = st.number_input(
                        "Stop Loss (Points)",
                        min_value=0.5,
                        max_value=500.0,
                        value=avg_loss,
                        step=0.5
                    )
                else:
                    stop_loss_bps = st.number_input(
                        "Stop Loss (bps)",
                        min_value=1,
                        max_value=500,
                        value=int(avg_loss_bps),
                        step=5
                    )
                    stop_loss = asset.bps_to_points(stop_loss_bps)
                    st.caption(f"= {stop_loss:.1f} points")

        # Display expected value
        st.markdown("---")
        st.subheader("Strategy Metrics Preview")

        expected_pnl_per_trade = (win_rate * avg_win - (1 - win_rate) * avg_loss) * asset.point_value

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Expected Win Rate", f"{win_rate*100:.0f}%")
        with col2:
            rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            st.metric("Risk/Reward Ratio", f"{rr_ratio:.2f}")
        with col3:
            st.metric("Exp. P&L/Trade (1 ct)", f"${expected_pnl_per_trade:.2f}")
        with col4:
            edge = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss * 100
            color = "normal" if edge > 0 else "inverse"
            st.metric("Edge", f"{edge:.1f}%", delta_color=color)

        # Realistic Price Path Simulation
        st.markdown("---")
        st.subheader("🔬 Realistic Price Path Simulation")

        use_realistic = st.checkbox(
            "Enable Realistic MFE/MAE Paths",
            value=False,
            help="Use historical MFE/MAE distributions from real market data instead of simple win/loss model"
        )

        realistic_config = RealisticTradeConfig(use_realistic_paths=False)

        if use_realistic:
            # Check if cache exists
            try:
                from regime_cache import cache_exists, get_regime_statistics, get_available_assets
                if cache_exists():
                    st.success("✅ Regime cache loaded")

                    col_r1, col_r2 = st.columns(2)

                    with col_r1:
                        trade_direction = st.selectbox(
                            "Trade Direction",
                            options=["random", "long", "short"],
                            help="Random: 50/50 long/short. Or force one direction."
                        )

                    with col_r2:
                        if price_unit == "Points":
                            break_even_trigger = st.number_input(
                                "Break-Even Trigger (Points)",
                                min_value=0.0,
                                max_value=500.0,
                                value=0.0,
                                step=1.0,
                                help="Move stop to entry after this many points profit. 0 = disabled."
                            )
                        else:
                            be_trigger_bps = st.number_input(
                                "Break-Even Trigger (bps)",
                                min_value=0,
                                max_value=500,
                                value=0,
                                step=5,
                                help="Move stop to entry after this many bps profit. 0 = disabled."
                            )
                            break_even_trigger = asset.bps_to_points(be_trigger_bps)
                            if be_trigger_bps > 0:
                                st.caption(f"= {break_even_trigger:.1f} points")

                    # Show regime stats for selected asset
                    cache_asset = {"MNQ": "NQ", "MES": "ES", "MYM": "YM"}.get(asset_symbol, asset_symbol)
                    available = get_available_assets()

                    if cache_asset in available:
                        stats = get_regime_statistics(cache_asset)
                        if "timeframes" in stats and "1H" in stats["timeframes"]:
                            tf_stats = stats["timeframes"]["1H"]
                            st.markdown("**Regime Distribution (1H candles):**")

                            regime_freqs = tf_stats.get("regime_frequencies", {})
                            if regime_freqs:
                                freq_cols = st.columns(4)
                                for i, (regime, freq) in enumerate(regime_freqs.items()):
                                    with freq_cols[i % 4]:
                                        st.metric(regime, f"{freq*100:.1f}%")

                            st.markdown("**Median MFE/MAE by Regime:**")
                            mfe_medians = tf_stats.get("mfe_medians", {})
                            mae_medians = tf_stats.get("mae_medians", {})
                            if mfe_medians:
                                mfe_mae_cols = st.columns(4)
                                point_val = asset.point_value

                                for i, regime in enumerate(["Low", "Normal", "High", "Extreme"]):
                                    with mfe_mae_cols[i]:
                                        mfe = mfe_medians.get(regime, 0)
                                        mae = mae_medians.get(regime, 0)
                                        # Convert to bps of asset price
                                        mfe_bps = asset.points_to_bps(mfe)
                                        mae_bps = asset.points_to_bps(mae)
                                        mfe_dollars = mfe * point_val
                                        mae_dollars = mae * point_val

                                        st.write(f"**{regime}**")
                                        st.write(f"MFE: {mfe:.1f} pts ({mfe_bps:.0f} bps)")
                                        st.write(f"MAE: {mae:.1f} pts ({mae_bps:.0f} bps)")
                    else:
                        st.warning(f"Asset {cache_asset} not in cache. Available: {', '.join(available)}")

                    realistic_config = RealisticTradeConfig(
                        use_realistic_paths=True,
                        trade_direction=trade_direction,
                        break_even_trigger=break_even_trigger,
                        target_points=avg_win  # Use avg_win as target
                    )
                else:
                    st.warning("⚠️ Regime cache not found. Run `python build_regime_cache.py` first.")
            except ImportError as e:
                st.error(f"❌ Error loading regime_cache module: {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

        st.session_state.realistic_config = realistic_config

        # Store trade config in session state
        st.session_state.trade_config = TradeConfig(
            risk_per_trade=risk_per_trade,
            risk_is_percentage=risk_is_percentage,
            trades_per_day_min=trades_min,
            trades_per_day_max=trades_max,
            win_rate=win_rate,
            avg_win_points=avg_win,
            avg_loss_points=avg_loss,
            stop_loss_points=stop_loss
        )

    # Tab 2: Simulation
    with tab2:
        st.header("Run Simulation")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Simulation Parameters")

            num_sims = st.select_slider(
                "Number of Simulations",
                options=[100, 500, 1000, 2000, 5000, 10000],
                value=1000
            )

            num_days = st.number_input(
                "Trading Days",
                min_value=10,
                max_value=504,
                value=60,
                step=10,
                help="252 days = 1 trading year"
            )

            distribution = st.selectbox(
                "P&L Distribution",
                options=["normal", "lognormal"],
                help="Normal: symmetric outcomes. Lognormal: fat tails."
            )

            use_seed = st.checkbox("Use Random Seed (Reproducible)")
            seed = None
            if use_seed:
                seed = st.number_input(
                    "Random Seed",
                    min_value=0,
                    max_value=99999,
                    value=42
                )

        with col2:
            st.subheader("Configuration Summary")

            st.markdown(f"""
            **Account:**
            - Starting Balance: ${starting_balance:,.0f}
            - Max Drawdown: ${max_drawdown:,.0f}
            - Trailing Type: {trailing_type.value}
            - Profit Target: ${profit_target:,.0f}

            **Asset:** {asset_symbol} (${asset.point_value:.2f}/point)

            **Trade Settings:**
            - Risk/Trade: {'$' + str(risk_per_trade) if not risk_is_percentage else str(risk_per_trade) + '%'}
            - Win Rate: {win_rate*100:.0f}%
            - Avg Win: {avg_win} pts (${avg_win * asset.point_value:.2f})
            - Avg Loss: {avg_loss} pts (${avg_loss * asset.point_value:.2f})
            """)

        st.markdown("---")

        # Run button
        if st.button("🚀 Run Monte Carlo Simulation", type="primary", use_container_width=True):

            # Create configurations
            account_config = AccountConfig(
                starting_balance=starting_balance,
                max_drawdown=max_drawdown,
                daily_drawdown_limit=daily_drawdown_limit,
                trailing_type=trailing_type,
                trailing_stop_point=trailing_stop_point,
                profit_target=profit_target
            )

            sim_config = SimulationConfig(
                num_simulations=num_sims,
                num_days=num_days,
                random_seed=seed,
                distribution=distribution
            )

            trade_config = st.session_state.get('trade_config', TradeConfig())

            # Run simulation with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(pct):
                progress_bar.progress(pct)
                status_text.text(f"Running simulations... {pct*100:.0f}%")

            # Get realistic config from session state
            realistic_config = st.session_state.get('realistic_config', RealisticTradeConfig())

            engine = MonteCarloEngine(
                account_config=account_config,
                trade_config=trade_config,
                asset=asset,
                sim_config=sim_config,
                realistic_config=realistic_config
            )

            results = engine.run_simulations(progress_callback=update_progress)

            # Store results
            st.session_state.simulation_results = results['all_results']
            st.session_state.simulation_summary = results['summary_stats']
            st.session_state.account_config = account_config
            st.session_state.sim_config = sim_config
            st.session_state.asset_symbol = asset_symbol

            progress_bar.progress(1.0)
            status_text.text("Simulation complete!")

            st.success(f"✅ Completed {num_sims} simulations over {num_days} trading days!")

            # Quick results preview
            summary = results['summary_stats']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Survival Rate",
                    f"{summary['survival_rate']*100:.1f}%",
                    help="Percentage of simulations that didn't breach"
                )
            with col2:
                st.metric(
                    "Hit Target Rate",
                    f"{summary['hit_profit_target_rate']*100:.1f}%",
                    help="Percentage that reached profit target"
                )
            with col3:
                st.metric(
                    "Median Final Equity",
                    f"${summary['median_final_equity']:,.0f}"
                )
            with col4:
                pnl = summary['median_final_equity'] - starting_balance
                st.metric(
                    "Median P&L",
                    f"${pnl:,.0f}",
                    delta=f"{pnl/starting_balance*100:.1f}%"
                )

    # Tab 3: Results Dashboard
    with tab3:
        st.header("Results Dashboard")

        if st.session_state.simulation_results is None:
            st.info("👆 Run a simulation first to see results here.")
        else:
            results = st.session_state.simulation_results
            summary = st.session_state.simulation_summary
            account_config = st.session_state.account_config

            # Key metrics row
            st.subheader("Key Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    "Survival Rate",
                    f"{summary['survival_rate']*100:.1f}%",
                    delta=f"{summary['accounts_survived']}/{summary['total_simulations']}"
                )
            with col2:
                st.metric(
                    "Failure Rate",
                    f"{summary['failure_rate']*100:.1f}%",
                    delta=f"{summary['accounts_breached']} breached",
                    delta_color="inverse"
                )
            with col3:
                st.metric(
                    "Hit Profit Target",
                    f"{summary['hit_profit_target_rate']*100:.1f}%",
                    delta=f"{summary['accounts_hit_target']} accounts"
                )
            with col4:
                st.metric(
                    "Profit Factor",
                    f"{summary['profit_factor']:.2f}" if summary['profit_factor'] < 100 else "∞"
                )
            with col5:
                st.metric(
                    "Exp. Value/Trade",
                    f"${summary['expected_value_per_trade']:.2f}"
                )

            st.markdown("---")

            # Charts row 1
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    create_equity_fan_chart(results, account_config),
                    use_container_width=True
                )

            with col2:
                st.plotly_chart(
                    create_survival_curve(results, st.session_state.sim_config.num_days),
                    use_container_width=True
                )

            # Charts row 2
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    create_final_balance_distribution(results, account_config),
                    use_container_width=True
                )

            with col2:
                st.plotly_chart(
                    create_drawdown_distribution(results),
                    use_container_width=True
                )

            # Charts row 3
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    create_breach_timing_chart(results),
                    use_container_width=True
                )

            with col2:
                st.plotly_chart(
                    create_win_rate_distribution(results),
                    use_container_width=True
                )

            # Sample paths
            st.subheader("Sample Equity Paths")
            num_paths = st.slider("Number of paths to display", 10, 200, 50)
            st.plotly_chart(
                create_sample_paths_chart(results, account_config, num_paths),
                use_container_width=True
            )

            # Detailed statistics
            st.markdown("---")
            st.subheader("Detailed Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Final Equity Percentiles**")
                if summary['final_equity_percentiles']:
                    for pct, val in summary['final_equity_percentiles'].items():
                        st.write(f"- {pct}th percentile: ${val:,.0f}")

            with col2:
                st.markdown("**Risk Metrics**")
                st.write(f"- VaR (95%): ${abs(summary['var_95']):,.0f}")
                st.write(f"- VaR (99%): ${abs(summary['var_99']):,.0f}")
                st.write(f"- CVaR (95%): ${abs(summary['cvar_95']):,.0f}")
                st.write(f"- Mean Max DD: ${summary['mean_max_drawdown']:,.0f}")

            with col3:
                st.markdown("**Trade Statistics**")
                st.write(f"- Total Trades: {summary['total_trades_all_sims']:,}")
                st.write(f"- Avg Trades/Sim: {summary['avg_trades_per_sim']:.0f}")
                st.write(f"- Mean Win Rate: {summary['mean_win_rate']*100:.1f}%")
                st.write(f"- Max Losing Streak: {summary['max_losing_streak_ever']}")

            # Breach timing details
            if summary['breach_day_percentiles']:
                st.markdown("---")
                st.subheader("Breach Timing Analysis")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Mean time to failure:** {summary['mean_time_to_failure']:.1f} days")
                    st.write(f"**Median time to failure:** {summary['median_time_to_failure']:.1f} days")

                with col2:
                    st.markdown("**Breach Day Percentiles:**")
                    for pct, val in summary['breach_day_percentiles'].items():
                        st.write(f"- {pct}th percentile: Day {val:.0f}")

    # Tab 4: Strategy Comparison
    with tab4:
        st.header("Strategy Comparison")
        st.info("Run multiple simulations with different parameters, then compare them here.")

        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = []

        col1, col2 = st.columns([3, 1])

        with col1:
            if st.session_state.simulation_results is not None:
                strategy_name = st.text_input(
                    "Strategy Name",
                    value=f"Strategy {len(st.session_state.comparison_results) + 1}"
                )

        with col2:
            if st.session_state.simulation_results is not None:
                if st.button("➕ Add Current Results", use_container_width=True):
                    st.session_state.comparison_results.append({
                        'name': strategy_name,
                        'summary': st.session_state.simulation_summary.copy(),
                        'config': {
                            'account': st.session_state.account_config,
                            'trade': st.session_state.trade_config,
                            'asset': st.session_state.asset_symbol
                        }
                    })
                    st.success(f"Added '{strategy_name}' to comparison!")

        if st.session_state.comparison_results:
            st.markdown("---")

            # Comparison table
            comparison_data = []
            for strat in st.session_state.comparison_results:
                s = strat['summary']
                comparison_data.append({
                    'Strategy': strat['name'],
                    'Survival Rate': f"{s['survival_rate']*100:.1f}%",
                    'Target Hit Rate': f"{s['hit_profit_target_rate']*100:.1f}%",
                    'Profit Factor': f"{s['profit_factor']:.2f}" if s['profit_factor'] < 100 else "∞",
                    'Median Final Equity': f"${s['median_final_equity']:,.0f}",
                    'Mean Max DD': f"${s['mean_max_drawdown']:,.0f}",
                    'VaR 95%': f"${abs(s['var_95']):,.0f}",
                    'Exp. Value/Trade': f"${s['expected_value_per_trade']:.2f}"
                })

            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)

            # Comparison charts
            if len(st.session_state.comparison_results) > 1:
                st.subheader("Visual Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    # Survival rate comparison
                    names = [s['name'] for s in st.session_state.comparison_results]
                    survival_rates = [s['summary']['survival_rate']*100
                                     for s in st.session_state.comparison_results]

                    fig = go.Figure(data=[
                        go.Bar(x=names, y=survival_rates, marker_color='green')
                    ])
                    fig.update_layout(
                        title="Survival Rate Comparison",
                        yaxis_title="Survival Rate (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Profit target hit rate comparison
                    hit_rates = [s['summary']['hit_profit_target_rate']*100
                                for s in st.session_state.comparison_results]

                    fig = go.Figure(data=[
                        go.Bar(x=names, y=hit_rates, marker_color='blue')
                    ])
                    fig.update_layout(
                        title="Profit Target Hit Rate Comparison",
                        yaxis_title="Hit Rate (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

            if st.button("🗑️ Clear All Comparisons"):
                st.session_state.comparison_results = []
                st.rerun()

    # Tab 5: Export
    with tab5:
        st.header("Export Results")

        if st.session_state.simulation_results is None:
            st.info("👆 Run a simulation first to export results.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Export Summary Statistics")
                csv_data = export_results_to_csv(
                    st.session_state.simulation_results,
                    st.session_state.simulation_summary
                )
                st.download_button(
                    label="Download Summary CSV",
                    data=csv_data,
                    file_name=f"monte_carlo_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                st.subheader("⚙️ Export Configuration")
                json_data = export_config_to_json(
                    st.session_state.account_config,
                    st.session_state.trade_config,
                    st.session_state.sim_config,
                    st.session_state.asset_symbol
                )
                st.download_button(
                    label="Download Config JSON",
                    data=json_data,
                    file_name=f"monte_carlo_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            st.markdown("---")

            # Preview exported data
            with st.expander("Preview Summary Data"):
                st.code(csv_data, language='csv')

            with st.expander("Preview Configuration"):
                st.code(json_data, language='json')


if __name__ == "__main__":
    main()
