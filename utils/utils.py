import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc

import numpy as np
import pandas as pd

from backtesterClass.orderBookClass import OBData


def global_perf(strategies : dict, strat_name : str):
    # Auto color mapping
    color_palette = pc.qualitative.Plotly
    strategy_names = list(strategies.keys())
    color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(strategy_names)}

    # Create 2 subplots: Portfolio Value (top), Cash (bottom)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Portfolio Value",
            "Cash"
        )
    )

    for name, strat in strategies.items():
        color = color_map[name]
        # Portfolio Value = AUM + Unrealized PnL
        portfolio_value = np.array(strat.historical_AUM) + np.array(strat.historical_unrealPnL)
        fig.add_trace(
            go.Scatter(
                x=OBData.OBData_[:, 0],
                y=portfolio_value,
                mode='lines',
                name=name,
                line=dict(color=color),
                legendgroup=name,
                showlegend=True
            ),
            row=1, col=1
        )
        # Cash
        fig.add_trace(
            go.Scatter(
                x=OBData.OBData_[:, 0],
                y=strat.historical_AUM,
                mode='lines',
                name=name,
                line=dict(color=color),
                legendgroup=name,
                showlegend=False
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=700,
        width=1000,
        template='plotly_white',
        legend_title="Strategy"
    )

    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
    fig.update_yaxes(title_text="Cash", row=2, col=1)

    fig.show()

def compute_returns(pnl_series):
    return np.diff(pnl_series)/pnl_series[:-1]

def compute_volatility(returns):
    return np.std(returns)

def compute_sharpe(returns):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return np.sqrt(252) * mean_return / std_return if std_return != 0 else np.nan

def compute_max_drawdown(pnl_series):
    pnl_array = np.array(pnl_series)
    cumulative_max = np.maximum.accumulate(pnl_array)
    # Avoid division by zero or negative
    valid = cumulative_max > 0
    drawdowns = np.full_like(pnl_array, np.nan, dtype=np.float64)
    drawdowns[valid] = (pnl_array[valid] - cumulative_max[valid]) / cumulative_max[valid]
    return np.nanmin(drawdowns)

def compute_annualized_return(total_return, n_periods):
    if n_periods == 0 or (1 + total_return) <= 0:
        return np.nan
    return (1 + total_return) ** (252 / n_periods) - 1

def compute_average_holding_period(trades, assets):
    # trades: list of [orderId, asset, sendTime, price, quantity, endTime, status]
    from collections import defaultdict
    holding_periods = defaultdict(list)
    open_positions = defaultdict(list)  # asset -> list of (open_time, quantity)
    for trade in trades:
        asset = trade[1]
        send_time = trade[2]
        quantity = trade[4]
        end_time = trade[5]
        if quantity == 0:
            continue
        # Open position
        if quantity > 0:
            open_positions[asset].append((send_time, quantity))
        else:
            # Close position (match with open positions)
            qty_to_close = -quantity
            while qty_to_close > 0 and open_positions[asset]:
                open_time, open_qty = open_positions[asset][0]
                if open_qty > qty_to_close:
                    holding_periods[asset].append(end_time - open_time)
                    open_positions[asset][0] = (open_time, open_qty - qty_to_close)
                    qty_to_close = 0
                else:
                    holding_periods[asset].append(end_time - open_time)
                    qty_to_close -= open_qty
                    open_positions[asset].pop(0)
    all_periods = [period for asset in assets for period in holding_periods[asset]]
    return np.mean(all_periods) if all_periods else np.nan

def compute_turnover(trades, portfolio_value_series):
    # trades: list of [orderId, asset, sendTime, price, quantity, endTime, status]
    total_bought = np.sum([trade[4] for trade in trades if trade[4] > 0])
    total_sold = np.sum([-trade[4] for trade in trades if trade[4] < 0])
    min_bought_sold = min(total_bought, total_sold)
    avg_portfolio_value = np.mean(portfolio_value_series)
    return (min_bought_sold / avg_portfolio_value) * 100 if avg_portfolio_value != 0 else np.nan

def peformance_metrics(strategies, verbose=False):
    # Compute stats
    data = []

    for name, strat in strategies.items():
        portfolio_value = np.array(strat.historical_unrealPnL) + np.array(strat.historical_AUM)
        n_periods = len(portfolio_value)
        if n_periods < 2:
            continue  # skip incomplete data
        # Compute returns as percent change in portfolio value
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        sharpe = compute_sharpe(returns)
        max_drawdown = compute_max_drawdown(portfolio_value) * 100  # percent
        total_pnl = strat.historical_unrealPnL[-1]  # Final unrealized PnL value
        total_return = ((portfolio_value[-1] - portfolio_value[0]) / portfolio_value[0] * 100) if portfolio_value[0] != 0 else np.nan
        annualized_return = compute_annualized_return((portfolio_value[-1] - portfolio_value[0]) / portfolio_value[0] if portfolio_value[0] != 0 else np.nan, n_periods) * 100
        volatility = np.std(returns) * np.sqrt(252) * 100  # annualized volatility in percent
        n_assets = len(getattr(strat, 'assets', []))
        # Avg PnL per asset in USD (using final unrealized PnL per asset)
        if hasattr(strat, 'historical_unrealPnL_per_asset'):
            per_asset_pnl = np.array(strat.historical_unrealPnL_per_asset)
            if per_asset_pnl.shape[0] > 0:
                avg_pnl_per_asset = np.mean(per_asset_pnl[-1])
                avg_variance_per_asset = np.mean(np.var(per_asset_pnl, axis=0))
            else:
                avg_pnl_per_asset = np.nan
                avg_variance_per_asset = np.nan
        else:
            avg_pnl_per_asset = np.nan
            avg_variance_per_asset = np.nan
        turnover = compute_turnover(getattr(strat, 'historical_trade', []), portfolio_value)
        data.append({
            "Strategy": name,
            "Sharpe Ratio (annualized)": sharpe,
            "Max Drawdown (%)": max_drawdown,
            "Total PnL ($m)": total_pnl/1e6,
            "Total Return (%)": total_return,
            "Annualized Return (%)": annualized_return,
            "Volatility (%)": volatility,
            "Avg PnL/Asset ($)": avg_pnl_per_asset,
            "Avg Variance/Asset": avg_variance_per_asset,
            "Turnover": turnover,
        })
    # Convert to DataFrame
    stats_df = pd.DataFrame(data)
    if not verbose:
        cols = [
            "Strategy",
            "Avg PnL/Asset ($)",
            "Avg Variance/Asset",
            "Sharpe Ratio (annualized)",
            "Max Drawdown (%)",
            "Annualized Return (%)"
        ]
        stats_df = stats_df[cols]
    return stats_df

def check_strategy_variance_and_params(strat, verbose=True, warn_threshold=1e9):
    """
    Checks variance and key statistics for all relevant parameters of a strategy.
    Prints or returns a summary with warnings if variance is suspiciously high.
    """
    import warnings
    results = {}
    def stat_report(arr, name):
        arr = np.array(arr)
        stats = {
            'min': np.nanmin(arr),
            'max': np.nanmax(arr),
            'mean': np.nanmean(arr),
            'std': np.nanstd(arr),
            'variance': np.nanvar(arr)
        }
        if stats['variance'] > warn_threshold:
            stats['warning'] = f"Variance of {name} is very high: {stats['variance']:.2e}"
            if verbose:
                warnings.warn(stats['warning'])
        if verbose:
            print(f"{name}: {stats}")
        return stats

    # Check historical PnL
    if hasattr(strat, 'historical_pnl'):
        results['historical_pnl'] = stat_report(strat.historical_pnl, 'historical_pnl')
    if hasattr(strat, 'historical_unrealPnL'):
        results['historical_unrealPnL'] = stat_report(strat.historical_unrealPnL, 'historical_unrealPnL')
    if hasattr(strat, 'historical_AUM'):
        results['historical_AUM'] = stat_report(strat.historical_AUM, 'historical_AUM')
    if hasattr(strat, 'historical_unrealPnL_per_asset'):
        arr = np.array(strat.historical_unrealPnL_per_asset)
        if arr.ndim == 2:
            for i, asset in enumerate(getattr(strat, 'assets', [])):
                results[f'unrealPnL_{asset}'] = stat_report(arr[:, i], f'unrealPnL_{asset}')
    if hasattr(strat, 'historical_AUM') and hasattr(strat, 'historical_unrealPnL'):
        portfolio_value = np.array(strat.historical_AUM) + np.array(strat.historical_unrealPnL)
        results['portfolio_value'] = stat_report(portfolio_value, 'portfolio_value')
    if hasattr(strat, 'historical_inventory'):
        arr = np.array(strat.historical_inventory)
        if arr.ndim == 2:
            for i, asset in enumerate(getattr(strat, 'assets', [])):
                results[f'inventory_{asset}'] = stat_report(arr[:, i], f'inventory_{asset}')
    # Add more checks as needed for other series
    return results
