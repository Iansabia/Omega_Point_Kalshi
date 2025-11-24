"""
Standalone Solara dashboard for Prediction Market ABM.
Pure Solara implementation without Mesa's SolaraViz.
"""

import pandas as pd
import plotly.graph_objects as go
import solara
from plotly.subplots import make_subplots

from src.models.market_model import PredictionMarketModel


# Global model state
@solara.component
def Page():
    """Main dashboard page."""

    # State variables
    n_noise = solara.use_reactive(50)
    n_informed = solara.use_reactive(10)
    n_makers = solara.use_reactive(2)
    initial_price = solara.use_reactive(0.5)

    model = solara.use_reactive(None)
    is_running = solara.use_reactive(False)
    current_step = solara.use_reactive(0)

    def create_model():
        """Create new model with current parameters."""
        agent_config = {}

        if n_noise.value > 0:
            agent_config["noise_trader"] = {"count": n_noise.value, "wealth": 1000}

        if n_informed.value > 0:
            agent_config["informed_trader"] = {"count": n_informed.value, "wealth": 10000, "information_quality": 0.8}

        if n_makers.value > 0:
            agent_config["market_maker"] = {"count": n_makers.value, "wealth": 100000, "risk_param": 0.1}

        market_config = {"initial_price": initial_price.value, "tick_size": 0.01}

        new_model = PredictionMarketModel(agent_config=agent_config, config={"market": market_config}, seed=42)
        model.value = new_model
        current_step.value = 0
        is_running.value = False

    def step_model():
        """Run one step of the simulation."""
        if model.value is not None:
            model.value.step()
            current_step.value += 1

    def reset_model():
        """Reset the model."""
        create_model()

    def toggle_running():
        """Toggle continuous running."""
        is_running.value = not is_running.value

    # Auto-step when running
    def auto_step():
        if is_running.value and model.value is not None:
            step_model()

    solara.lab.use_task(auto_step, dependencies=[is_running.value, current_step.value])

    # Create model on first load
    if model.value is None:
        create_model()

    # Layout
    with solara.Column(style={"padding": "20px"}):
        # Title
        solara.Markdown("# 📊 Prediction Market ABM Dashboard")

        # Controls
        with solara.Card("Controls", elevation=2):
            with solara.Column():
                # Sliders
                solara.SliderInt("Noise Traders", value=n_noise, min=10, max=200, step=10)
                solara.SliderInt("Informed Traders", value=n_informed, min=5, max=50, step=5)
                solara.SliderInt("Market Makers", value=n_makers, min=1, max=10, step=1)
                solara.SliderFloat("Initial Price", value=initial_price, min=0.1, max=0.9, step=0.05)

                # Buttons
                with solara.Row():
                    solara.Button("Reset", on_click=reset_model, color="primary")
                    solara.Button("Step", on_click=step_model, disabled=model.value is None)
                    solara.Button(
                        "Pause" if is_running.value else "Play",
                        on_click=toggle_running,
                        color="success" if not is_running.value else "warning",
                        disabled=model.value is None,
                    )

                # Status
                if model.value is not None:
                    solara.Markdown(
                        f"**Step:** {current_step.value} | **Agents:** {len(list(model.value.agents))} | **Price:** ${model.value.current_price:.3f}"
                    )

        # Charts
        if model.value is not None and current_step.value > 0:
            with solara.Card("Market Overview", elevation=2):
                MarketCharts(model.value)

            with solara.Card("Agent Analysis", elevation=2):
                AgentCharts(model.value)


@solara.component
def MarketCharts(model):
    """Display market-level charts."""
    df = model.datacollector.get_model_vars_dataframe()

    if len(df) == 0:
        solara.Markdown("*No data yet. Click Step or Play to run the simulation.*")
        return

    # Create subplot figure
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Market Price vs Fundamental", "Trading Volume", "Bid-Ask Spread", "Cumulative Returns"),
    )

    # Price chart
    fig.add_trace(
        go.Scatter(x=df.index, y=df["market_price"], mode="lines", name="Market Price", line=dict(color="blue", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["fundamental_value"],
            mode="lines",
            name="Fundamental",
            line=dict(color="red", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Volume chart
    fig.add_trace(go.Bar(x=df.index, y=df["total_volume"], name="Volume", marker_color="green"), row=1, col=2)

    # Spread chart
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["bid_ask_spread"], mode="lines", name="Spread", line=dict(color="orange", width=2), fill="tozeroy"
        ),
        row=2,
        col=1,
    )

    # Returns chart
    if len(df) > 1:
        returns = df["market_price"].pct_change().fillna(0)
        cum_returns = (1 + returns).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=df.index, y=cum_returns, mode="lines", name="Returns", line=dict(color="purple", width=2), fill="tozeroy"
            ),
            row=2,
            col=2,
        )

    fig.update_layout(height=600, showlegend=False)
    solara.FigurePlotly(fig)


@solara.component
def AgentCharts(model):
    """Display agent-level charts."""
    agent_df = model.datacollector.get_agent_vars_dataframe()

    if len(agent_df) == 0:
        solara.Markdown("*No agent data yet.*")
        return

    # Get latest step
    latest_step = agent_df.index.get_level_values("Step").max()
    latest_data = agent_df.xs(latest_step, level="Step")

    # Create subplot figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Wealth Distribution", "Wealth by Agent Type"))

    # Wealth histogram
    fig.add_trace(go.Histogram(x=latest_data["wealth"], nbinsx=30, name="Wealth", marker_color="blue"), row=1, col=1)

    # Wealth by type (box plot)
    for agent_type in latest_data["agent_type"].unique():
        type_data = latest_data[latest_data["agent_type"] == agent_type]
        fig.add_trace(go.Box(y=type_data["wealth"], name=agent_type), row=1, col=2)

    fig.update_layout(height=400, showlegend=True)
    solara.FigurePlotly(fig)

    # Summary stats
    with solara.Row():
        with solara.Column():
            solara.Markdown(f"**Avg Wealth:** ${latest_data['wealth'].mean():.2f}")
            solara.Markdown(f"**Std Dev:** ${latest_data['wealth'].std():.2f}")
        with solara.Column():
            solara.Markdown(f"**Total Positions:** {latest_data['position'].sum():.0f}")
            solara.Markdown(f"**Agents:** {len(latest_data)}")
