"""
Simple working Solara dashboard for Prediction Market ABM.
Compatible with Mesa 3.3+ SolaraViz.
"""

import pandas as pd
import plotly.graph_objects as go
import solara
from mesa.visualization import SolaraViz

from src.models.market_model import PredictionMarketModel


def make_plot(model):
    """Create market price plot."""
    # Get data
    if not hasattr(model, "datacollector") or model.datacollector is None:
        return go.Figure()

    df = model.datacollector.get_model_vars_dataframe()

    if len(df) == 0:
        return go.Figure()

    # Create figure
    fig = go.Figure()

    # Add price line
    fig.add_trace(
        go.Scatter(x=df.index, y=df["market_price"], mode="lines", name="Market Price", line=dict(color="blue", width=2))
    )

    # Add fundamental value line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["fundamental_value"],
            mode="lines",
            name="Fundamental Value",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    fig.update_layout(title="Market Price vs Fundamental Value", xaxis_title="Step", yaxis_title="Price", height=400)

    return fig


def make_volume_plot(model):
    """Create volume plot."""
    if not hasattr(model, "datacollector") or model.datacollector is None:
        return go.Figure()

    df = model.datacollector.get_model_vars_dataframe()

    if len(df) == 0:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Bar(x=df.index, y=df["total_volume"], name="Volume", marker_color="green"))

    fig.update_layout(title="Trading Volume", xaxis_title="Step", yaxis_title="Volume", height=400)

    return fig


def make_spread_plot(model):
    """Create spread plot."""
    if not hasattr(model, "datacollector") or model.datacollector is None:
        return go.Figure()

    df = model.datacollector.get_model_vars_dataframe()

    if len(df) == 0:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["bid_ask_spread"], mode="lines", name="Spread", line=dict(color="orange", width=2), fill="tozeroy"
        )
    )

    fig.update_layout(title="Bid-Ask Spread", xaxis_title="Step", yaxis_title="Spread", height=400)

    return fig


def make_wealth_plot(model):
    """Create wealth distribution plot."""
    if not hasattr(model, "datacollector") or model.datacollector is None:
        return go.Figure()

    agent_df = model.datacollector.get_agent_vars_dataframe()

    if len(agent_df) == 0:
        return go.Figure()

    # Get latest step
    latest_step = agent_df.index.get_level_values("Step").max()
    latest_data = agent_df.xs(latest_step, level="Step")

    fig = go.Figure()

    # Create histogram
    fig.add_trace(go.Histogram(x=latest_data["wealth"], nbinsx=30, name="Wealth Distribution", marker_color="blue"))

    fig.update_layout(title="Agent Wealth Distribution", xaxis_title="Wealth ($)", yaxis_title="Count", height=400)

    return fig


# Model parameters
model_params = {
    "n_noise_traders": {"type": "SliderInt", "value": 50, "label": "Noise Traders", "min": 10, "max": 200, "step": 10},
    "n_informed_traders": {"type": "SliderInt", "value": 10, "label": "Informed Traders", "min": 5, "max": 50, "step": 5},
    "n_market_makers": {"type": "SliderInt", "value": 2, "label": "Market Makers", "min": 1, "max": 10, "step": 1},
    "initial_price": {"type": "SliderFloat", "value": 0.5, "label": "Initial Price", "min": 0.1, "max": 0.9, "step": 0.05},
}


def model_constructor(n_noise_traders=50, n_informed_traders=10, n_market_makers=2, initial_price=0.5):
    """Construct model from parameters."""
    agent_config = {}

    if n_noise_traders > 0:
        agent_config["noise_trader"] = {"count": n_noise_traders, "wealth": 1000}

    if n_informed_traders > 0:
        agent_config["informed_trader"] = {"count": n_informed_traders, "wealth": 10000, "information_quality": 0.8}

    if n_market_makers > 0:
        agent_config["market_maker"] = {"count": n_market_makers, "wealth": 100000, "risk_param": 0.1}

    market_config = {"initial_price": initial_price, "tick_size": 0.01}

    return PredictionMarketModel(agent_config=agent_config, config={"market": market_config}, seed=42)


# Create the page
page = SolaraViz(
    model_constructor,
    model_params=model_params,
    measures=[make_plot, make_volume_plot, make_spread_plot, make_wealth_plot],
    name="Prediction Market ABM",
)
