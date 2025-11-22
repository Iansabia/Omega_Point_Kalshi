"""
Solara Dashboard for Prediction Market ABM.

Multi-page interactive dashboard with real-time updates:
- Page 1: Market overview (price, volume, spread)
- Page 2: Agent behavior (wealth distribution, positions)
- Page 3: Performance (Sharpe, drawdown, P&L)
- Page 4: Order book visualization (depth chart, heatmap)
"""
import solara
from mesa.visualization import SolaraViz, make_plot_component
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional

from ..models.market_model import PredictionMarketModel


@solara.component
def MarketPriceChart(model: PredictionMarketModel):
    """Page 1: Market overview - price, volume, spread."""
    # Force update on model step
    update_counter = solara.use_reactive(0)

    def _update():
        update_counter.value += 1

    # Get data from datacollector
    model_data = model.datacollector.get_model_vars_dataframe()

    if len(model_data) == 0:
        return solara.Markdown("No data available yet. Run simulation to see charts.")

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Market Price vs Fundamental Value', 'Trading Volume', 'Bid-Ask Spread'),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25]
    )

    # Price chart
    fig.add_trace(
        go.Scatter(
            x=model_data.index,
            y=model_data['market_price'],
            mode='lines',
            name='Market Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=model_data.index,
            y=model_data['fundamental_value'],
            mode='lines',
            name='Fundamental Value',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )

    # Volume chart
    fig.add_trace(
        go.Bar(
            x=model_data.index,
            y=model_data['total_volume'],
            name='Volume',
            marker_color='green'
        ),
        row=2, col=1
    )

    # Spread chart
    fig.add_trace(
        go.Scatter(
            x=model_data.index,
            y=model_data['bid_ask_spread'],
            mode='lines',
            name='Spread',
            line=dict(color='orange', width=2),
            fill='tozeroy'
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_xaxes(title_text="Step", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Spread", row=3, col=1)

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Market Overview",
        title_x=0.5
    )

    return solara.FigurePlotly(fig)


@solara.component
def AgentBehaviorChart(model: PredictionMarketModel):
    """Page 2: Agent behavior - wealth distribution, positions."""
    update_counter = solara.use_reactive(0)

    # Get agent data
    agent_data = model.datacollector.get_agent_vars_dataframe()

    if len(agent_data) == 0:
        return solara.Markdown("No agent data available yet.")

    # Get latest step data
    latest_step = agent_data.index.get_level_values('Step').max()
    latest_data = agent_data.xs(latest_step, level='Step')

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Wealth Distribution by Agent Type',
            'Position Distribution',
            'Wealth Histogram',
            'Top 10 Agents by Wealth'
        ),
        specs=[
            [{"type": "box"}, {"type": "box"}],
            [{"type": "histogram"}, {"type": "bar"}]
        ]
    )

    # Wealth by agent type (box plot)
    for agent_type in latest_data['agent_type'].unique():
        type_data = latest_data[latest_data['agent_type'] == agent_type]
        fig.add_trace(
            go.Box(
                y=type_data['wealth'],
                name=agent_type,
                boxmean='sd'
            ),
            row=1, col=1
        )

    # Position by agent type (box plot)
    for agent_type in latest_data['agent_type'].unique():
        type_data = latest_data[latest_data['agent_type'] == agent_type]
        fig.add_trace(
            go.Box(
                y=type_data['position'],
                name=agent_type,
                showlegend=False
            ),
            row=1, col=2
        )

    # Wealth histogram
    fig.add_trace(
        go.Histogram(
            x=latest_data['wealth'],
            nbinsx=30,
            name='Wealth Distribution',
            marker_color='blue'
        ),
        row=2, col=1
    )

    # Top 10 agents by wealth
    top_agents = latest_data.nlargest(10, 'wealth')
    fig.add_trace(
        go.Bar(
            x=[f"Agent {i}" for i in range(10)],
            y=top_agents['wealth'].values,
            name='Top 10 Agents',
            marker_color='green',
            text=top_agents['agent_type'].values,
            textposition='auto'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_yaxes(title_text="Wealth", row=1, col=1)
    fig.update_yaxes(title_text="Position", row=1, col=2)
    fig.update_xaxes(title_text="Wealth", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Agent Rank", row=2, col=2)
    fig.update_yaxes(title_text="Wealth", row=2, col=2)

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Agent Behavior Analysis",
        title_x=0.5
    )

    return solara.FigurePlotly(fig)


@solara.component
def PerformanceChart(model: PredictionMarketModel):
    """Page 3: Performance metrics - Sharpe, drawdown, P&L."""
    update_counter = solara.use_reactive(0)

    model_data = model.datacollector.get_model_vars_dataframe()

    if len(model_data) == 0:
        return solara.Markdown("No performance data available yet.")

    # Calculate returns
    prices = model_data['market_price'].values
    returns = np.diff(prices) / prices[:-1]
    returns = np.insert(returns, 0, 0)  # Add 0 for first step

    # Calculate cumulative returns
    cumulative_returns = (1 + pd.Series(returns)).cumprod() - 1

    # Calculate rolling Sharpe ratio (20-period window)
    rolling_sharpe = pd.Series(returns).rolling(window=20).apply(
        lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
    )

    # Calculate drawdown
    cumulative = (1 + pd.Series(returns)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Cumulative Returns', 'Rolling Sharpe Ratio (20-period)', 'Drawdown %'),
        vertical_spacing=0.1
    )

    # Cumulative returns
    fig.add_trace(
        go.Scatter(
            x=model_data.index,
            y=cumulative_returns,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='green', width=2),
            fill='tozeroy'
        ),
        row=1, col=1
    )

    # Rolling Sharpe
    fig.add_trace(
        go.Scatter(
            x=model_data.index,
            y=rolling_sharpe,
            mode='lines',
            name='Sharpe Ratio',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )

    # Add horizontal line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=model_data.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_xaxes(title_text="Step", row=3, col=1)
    fig.update_yaxes(title_text="Return %", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)

    # Calculate summary statistics
    total_return = cumulative_returns.iloc[-1] * 100
    current_sharpe = rolling_sharpe.iloc[-1]
    max_drawdown = drawdown.min()

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Performance Metrics | Return: {total_return:.2f}% | Sharpe: {current_sharpe:.2f} | Max DD: {max_drawdown:.2f}%",
        title_x=0.5
    )

    return solara.FigurePlotly(fig)


@solara.component
def OrderBookChart(model: PredictionMarketModel):
    """Page 4: Order book visualization - depth chart, heatmap."""
    update_counter = solara.use_reactive(0)

    # Get current order book state
    order_book = model.order_book

    # Get depth data
    depth = order_book.get_depth(levels=10)

    if depth is None or (len(depth['bids']) == 0 and len(depth['asks']) == 0):
        return solara.Markdown("No order book data available. Orders will appear here when agents place them.")

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Order Book Depth (Top 10 Levels)', 'Order Book Imbalance'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )

    # Extract bid/ask data
    bid_prices = [level[0] for level in depth['bids']]
    bid_quantities = [level[1] for level in depth['bids']]
    ask_prices = [level[0] for level in depth['asks']]
    ask_quantities = [level[1] for level in depth['asks']]

    # Depth chart (bids)
    fig.add_trace(
        go.Bar(
            x=bid_prices,
            y=bid_quantities,
            name='Bids',
            marker_color='green',
            orientation='v'
        ),
        row=1, col=1
    )

    # Depth chart (asks)
    fig.add_trace(
        go.Bar(
            x=ask_prices,
            y=ask_quantities,
            name='Asks',
            marker_color='red',
            orientation='v'
        ),
        row=1, col=1
    )

    # Add mid price line
    mid_price = order_book.get_mid_price()
    if mid_price:
        fig.add_vline(
            x=mid_price,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Mid: {mid_price:.3f}",
            row=1, col=1
        )

    # Order book imbalance over time
    model_data = model.datacollector.get_model_vars_dataframe()
    if len(model_data) > 0 and 'step' in model_data.columns:
        # Calculate imbalance (would need to be added to datacollector)
        # For now, use spread as proxy
        fig.add_trace(
            go.Scatter(
                x=model_data.index,
                y=model_data['bid_ask_spread'],
                mode='lines',
                name='Spread (Imbalance Proxy)',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_xaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Quantity", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1)

    # Calculate summary statistics
    total_bid_volume = sum(bid_quantities) if bid_quantities else 0
    total_ask_volume = sum(ask_quantities) if ask_quantities else 0
    imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Order Book | Bid Volume: {total_bid_volume:.0f} | Ask Volume: {total_ask_volume:.0f} | Imbalance: {imbalance:.2%}",
        title_x=0.5
    )

    return solara.FigurePlotly(fig)


# Model parameters for interactive dashboard
model_params = {
    "n_noise_traders": {
        "type": "SliderInt",
        "value": 100,
        "label": "Noise Traders",
        "min": 10,
        "max": 500,
        "step": 10
    },
    "n_informed_traders": {
        "type": "SliderInt",
        "value": 20,
        "label": "Informed Traders",
        "min": 5,
        "max": 100,
        "step": 5
    },
    "n_arbitrageurs": {
        "type": "SliderInt",
        "value": 10,
        "label": "Arbitrageurs",
        "min": 1,
        "max": 50,
        "step": 5
    },
    "n_market_makers": {
        "type": "SliderInt",
        "value": 5,
        "label": "Market Makers",
        "min": 1,
        "max": 20,
        "step": 1
    },
    "initial_price": {
        "type": "SliderFloat",
        "value": 0.5,
        "label": "Initial Price",
        "min": 0.1,
        "max": 0.9,
        "step": 0.05
    },
    "fundamental_value": {
        "type": "SliderFloat",
        "value": 0.5,
        "label": "Fundamental Value",
        "min": 0.1,
        "max": 0.9,
        "step": 0.05
    }
}


def agent_portrayal(agent):
    """Define how agents are portrayed in spatial visualizations."""
    return {
        "size": 50,
        "color": {
            "NoiseTrader": "blue",
            "InformedTrader": "green",
            "Arbitrageur": "red",
            "MarketMakerAgent": "purple",
            "HomerAgent": "orange",
            "LLMAgent": "yellow"
        }.get(agent.__class__.__name__, "gray")
    }


def create_model_from_params(params):
    """Create PredictionMarketModel from dashboard parameters."""
    agent_config = {}

    if params.get("n_noise_traders", 0) > 0:
        agent_config['noise_trader'] = {
            'count': params["n_noise_traders"],
            'wealth': 1000
        }

    if params.get("n_informed_traders", 0) > 0:
        agent_config['informed_trader'] = {
            'count': params["n_informed_traders"],
            'wealth': 10000,
            'information_quality': 0.8
        }

    if params.get("n_arbitrageurs", 0) > 0:
        agent_config['arbitrageur'] = {
            'count': params["n_arbitrageurs"],
            'wealth': 50000,
            'detection_speed': 0.9
        }

    if params.get("n_market_makers", 0) > 0:
        agent_config['market_maker'] = {
            'count': params["n_market_makers"],
            'wealth': 100000,
            'risk_param': 0.1
        }

    market_config = {
        'initial_price': params.get("initial_price", 0.5),
        'tick_size': 0.01
    }

    return PredictionMarketModel(
        agent_config=agent_config,
        config={'market': market_config},
        seed=42
    )


# Create SolaraViz page
page = SolaraViz(
    create_model_from_params,
    components=[
        MarketPriceChart,
        AgentBehaviorChart,
        PerformanceChart,
        OrderBookChart
    ],
    model_params=model_params,
    name="Prediction Market ABM Dashboard"
)


if __name__ == "__main__":
    # Run the dashboard
    page  # Solara will automatically serve this
