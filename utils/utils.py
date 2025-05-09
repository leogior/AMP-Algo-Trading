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

    # Create 3 subplots (AUM, Realized, Unrealized)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Realized PnL",
            "Unrealized PnL",
            "AUM"
        )
    )


    # Add Realized & Unrealized PnL traces
    for name, strat in strategies.items():
        color = color_map[name]

        # Realized PnL — legend shown
        fig.add_trace(
            go.Scatter(
                x=OBData.OBData_[:, 0],
                y=strat.historical_pnl,
                mode='lines',
                name=name,
                line=dict(color=color),
                legendgroup=name,
                showlegend=True
            ),
            row=1, col=1
        )

        # Unrealized PnL — legend hidden
        fig.add_trace(
            go.Scatter(
                x=OBData.OBData_[:, 0],
                y=strat.historical_unrealPnL,
                mode='lines',
                name=name,
                line=dict(color=color),
                legendgroup=name,
                showlegend=False
            ),
            row=2, col=1
        )

        # AUM for strategy 50-200 (first plot)
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
            row=3, col=1
        )
    # Layout
    fig.update_layout(
        height=1000,
        width=1000,
        title_text=f"{strat_name} Strategies: AUM, Realized & Unrealized PnL",
        template='plotly_white',
        legend_title="Strategy"
    )

    fig.update_xaxes(title_text="Time Step", row=3, col=1)
    fig.update_yaxes(title_text="Realized PnL", row=1, col=1)
    fig.update_yaxes(title_text="Unrealized PnL", row=2, col=1)
    fig.update_yaxes(title_text="AUM", row=3, col=1)

    fig.show()

def compute_portfolio_value(strat):
    portfolio_values = []
    for t in range(len(strat.historical_AUM)):
        # Cash available at time t
        cash = strat.historical_AUM[t]
        # For each asset, get quantity at time t and price at time t
        asset_value = 0
        for asset in strat.assets:
            try:
                quantity = strat.historical_inventory[t][strat.assets.index(asset)]
            except (IndexError, KeyError):
                quantity = 0
            try:
                price = OBData.OBData_[t, OBData.assetIdx[asset]]
            except Exception:
                price = 0
            asset_value += quantity * price
        portfolio_values.append(cash + asset_value)
    return np.array(portfolio_values)

def compute_returns(portfolio_value_series):
    portfolio_value_series = np.array(portfolio_value_series)
    # Use log returns for stability
    returns = np.diff(np.log(portfolio_value_series[portfolio_value_series > 0]))
    return returns

def compute_volatility(returns):
    return np.std(returns)

def compute_sharpe(returns):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return mean_return / std_return if std_return != 0 else np.nan

def compute_max_drawdown(portfolio_values):
    portfolio_values = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return -drawdown.min() * 100  # as positive percentage

def peformance_metrics(strategies):
    # Compute stats
    data = []
    periods_per_year = 252  # Assuming daily returns

    for name, strat in strategies.items():
        portfolio_values = compute_portfolio_value(strat)
        if len(portfolio_values) < 2:
            continue  # skip incomplete data

        returns = compute_returns(portfolio_values)
        
        mean_return = np.mean(returns)
        volatility = compute_volatility(returns)
        # Annualize
        ann_mean_return = mean_return * periods_per_year * 100  # as percentage
        ann_volatility = volatility * np.sqrt(periods_per_year) * 100  # as percentage
        ann_sharpe = (mean_return * periods_per_year) / (volatility * np.sqrt(periods_per_year)) if volatility != 0 else np.nan
        max_dd = compute_max_drawdown(portfolio_values)  # as positive percentage
        total_pnl = portfolio_values[-1] - portfolio_values[0]  # Total change in portfolio value

        data.append({
            "Strategy": name,
            "Annualized Sharpe Ratio": ann_sharpe,
            "Annualized Volatility (%)": ann_volatility,
            "Annualized Mean Return (%)": ann_mean_return,
            "Max Drawdown (%)": max_dd,
            "Total PnL ($m)": total_pnl/1e6,
        })

    # Convert to DataFrame
    stats_df = pd.DataFrame(data)
    return stats_df

def plot_return_distributions(strategies):
    fig = go.Figure()
    for name, strat in strategies.items():
        portfolio_values = compute_portfolio_value(strat)
        if len(portfolio_values) < 2:
            continue
        returns = compute_returns(portfolio_values) * 100  # as percentage
        fig.add_trace(go.Histogram(
            x=returns,
            name=name,
            opacity=0.6,
            nbinsx=50
        ))
    fig.update_layout(
        barmode='overlay',
        title='Distribution of Periodic Returns by Strategy',
        xaxis_title='Return per Period (%)',
        yaxis_title='Count',
        template='plotly_white',
        legend_title='Strategy'
    )
    fig.update_traces(opacity=0.6)
    fig.show()

def per_asset_report(strategies):
    # For each strategy, compute the requested metrics with correct averaging
    rows = []
    for strat_name, strat in strategies.items():
        assets = strat.assets if hasattr(strat, 'assets') else OBData.assets
        pnl_per_asset = np.array(strat.historical_pnl_per_asset)
        inventory = np.array(strat.historical_inventory)
        trades = pd.DataFrame(strat.historical_trade, columns=["tradeIndex", "asset", "sendTime", "price", "quantity", "endTime", "status"])
        n_assets = len(assets)
        n_periods = len(pnl_per_asset)
        # For new averaging
        pnl_diffs = []
        variances = []
        num_trades_total = 0
        hit_ratios = []
        turnover_rates = []
        avg_holding_periods = []
        volatilities = []
        for i, asset in enumerate(assets):
            pnl_series = pnl_per_asset[:, i] if pnl_per_asset.shape[0] > 0 else np.zeros(n_periods)
            inv_series = inventory[:, i] if inventory.shape[0] > 0 else np.zeros(n_periods)
            asset_trades = trades[trades["asset"] == asset]
            pnl_diffs.append(pnl_series[-1] - pnl_series[0])
            variances.append(np.var(pnl_series))
            volatilities.append(np.std(pnl_series))
            num_trades = len(asset_trades)
            num_trades_total += num_trades
            holding_periods = []
            holding = False
            start = 0
            for t, qty in enumerate(inv_series):
                if not holding and qty != 0:
                    holding = True
                    start = t
                elif holding and qty == 0:
                    holding = False
                    holding_periods.append(t - start)
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0
            avg_holding_periods.append(avg_holding_period)
            profits = asset_trades["quantity"].astype(float) * (asset_trades["price"].astype(float))
            hit_ratio = (profits > 0).sum() / num_trades if num_trades > 0 else 0
            hit_ratios.append(hit_ratio)
            total_volume = asset_trades["quantity"].abs().sum() if num_trades > 0 else 0
            avg_inventory = np.mean(np.abs(inv_series)) if n_periods > 0 else 1
            turnover_rate = total_volume / avg_inventory if avg_inventory > 0 else 0
            turnover_rates.append(turnover_rate)
        avg_pnl = sum(pnl_diffs) / n_assets if n_assets > 0 else 0
        avg_variance = sum(variances) / n_assets if n_assets > 0 else 0
        avg_num_trades = num_trades_total / n_assets if n_assets > 0 else 0
        rows.append({
            "Strategy": strat_name,
            "Avg PnL": avg_pnl,
            "Variance": avg_variance,
            "Volatility": np.mean(volatilities),
            "Avg # Trades": avg_num_trades,
            "Avg Holding Period": np.mean(avg_holding_periods),
            "Hit Ratio": np.mean(hit_ratios),
            "Turnover Rate": np.mean(turnover_rates)
        })
    df = pd.DataFrame(rows)
    print(df)
    return df
