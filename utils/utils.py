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

def compute_returns(pnl_series):
    return np.diff(pnl_series)

def compute_volatility(returns):
    return np.std(returns)

def compute_sharpe(returns):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return mean_return / std_return if std_return != 0 else np.nan

def compute_max_drawdown(pnl_series):
    pnl_array = np.array(pnl_series)
    cumulative_max = np.maximum.accumulate(pnl_array)
    drawdowns = (pnl_array - cumulative_max)
    return drawdowns.min()

def peformance_metrics(strategies):

    # Compute stats
    data = []

    for name, strat in strategies.items():
        pnl = strat.historical_unrealPnL
        if len(pnl) < 2:
            continue  # skip incomplete data

        returns = compute_returns(pnl)
        
        sharpe = compute_sharpe(returns)
        max_drawdown = compute_max_drawdown(pnl)
        total_pnl = pnl[-1]  # Final unrealized PnL value

        data.append({
            "Strategy": name,
            "Sharpe Ratio": sharpe,
            "Max Drawdown ($m)": max_drawdown/1e8,
            "Total PnL ($m)": total_pnl/1e8,
        })

    # Convert to DataFrame
    stats_df = pd.DataFrame(data)
    return stats_df
