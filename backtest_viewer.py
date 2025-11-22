"""
Interactive Backtest Results Viewer with Monte Carlo Simulations.

Features:
- Visual exploration of backtest results
- Monte Carlo simulations on NFL games
- Interactive charts and analysis
- Performance metrics dashboard

Usage:
    solara run backtest_viewer.py
    Then open: http://localhost:8765
"""
import solara
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path


@solara.component
def Page():
    """Main backtest viewer page."""

    with solara.Column(style={"padding": "20px"}):
        solara.Markdown("# 📊 Backtest Results & Monte Carlo Analysis")

        # Tabs for different views
        with solara.Card():
            tabs = ["Backtest Results", "Monte Carlo Simulation", "Trade Analysis", "Performance Metrics"]
            selected_tab = solara.use_reactive(0)

            with solara.Row():
                for i, tab in enumerate(tabs):
                    solara.Button(
                        tab,
                        on_click=lambda i=i: selected_tab.set(i),
                        color="primary" if selected_tab.value == i else "default"
                    )

            # Show selected tab
            if selected_tab.value == 0:
                BacktestResultsView()
            elif selected_tab.value == 1:
                MonteCarloView()
            elif selected_tab.value == 2:
                TradeAnalysisView()
            elif selected_tab.value == 3:
                PerformanceMetricsView()


@solara.component
def BacktestResultsView():
    """Display backtest results from CSV files."""

    # Load data
    equity_file = Path("backtest_equity_curve.csv")

    if not equity_file.exists():
        with solara.Column():
            solara.Warning("No backtest results found. Run a backtest first:")
            solara.Code("python run_backtest.py --games 100")
        return

    df = pd.read_csv(equity_file)

    with solara.Column():
        # Summary cards
        with solara.Row():
            with solara.Card("Capital", elevation=2):
                initial = df['capital'].iloc[0] if len(df) > 0 else 0
                final = df['capital'].iloc[-1] if len(df) > 0 else 0
                pnl = final - initial

                solara.Markdown(f"### ${final:,.2f}")
                color = "green" if pnl >= 0 else "red"
                pnl_text = f"P&L: ${pnl:,.2f}" if pnl >= 0 else f"P&L: -${abs(pnl):,.2f}"
                solara.Markdown(f"**{pnl_text}**")

            with solara.Card("Total Return", elevation=2):
                if len(df) > 0 and df['capital'].iloc[0] != 0:
                    ret = ((df['capital'].iloc[-1] / df['capital'].iloc[0]) - 1) * 100
                    solara.Markdown(f"### {ret:.1f}%")
                    status = "Profit" if ret >= 0 else "Loss"
                    solara.Markdown(f"**{status}**")

            with solara.Card("Win Rate", elevation=2):
                if len(df) > 0:
                    wins = len(df[df['pnl'] > 0])
                    total = len(df)
                    win_rate = (wins / total * 100) if total > 0 else 0
                    solara.Markdown(f"### {win_rate:.1f}%")
                    solara.Markdown(f"{wins} wins / {total} games")

        # Charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Equity Curve',
                'Per-Game P&L Distribution',
                'Drawdown',
                'Cumulative Return'
            )
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(x=df['game'], y=df['capital'], mode='lines',
                      name='Capital', line=dict(color='blue', width=2)),
            row=1, col=1
        )

        # P&L distribution
        fig.add_trace(
            go.Histogram(x=df['pnl'], nbinsx=30, name='P&L',
                        marker_color='green'),
            row=1, col=2
        )

        # Drawdown
        if len(df) > 0:
            equity = df['capital'].values
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max * 100

            fig.add_trace(
                go.Scatter(x=df['game'], y=drawdown, mode='lines',
                          name='Drawdown', line=dict(color='red', width=2),
                          fill='tozeroy'),
                row=2, col=1
            )

        # Cumulative return
        if len(df) > 0 and df['capital'].iloc[0] != 0:
            cum_return = (df['capital'] / df['capital'].iloc[0] - 1) * 100
            fig.add_trace(
                go.Scatter(x=df['game'], y=cum_return, mode='lines',
                          name='Return', line=dict(color='purple', width=2)),
                row=2, col=2
            )

        fig.update_layout(height=800, showlegend=False)
        solara.FigurePlotly(fig)


@solara.component
def MonteCarloView():
    """Run Monte Carlo simulations on NFL game scenarios."""

    # Parameters
    n_sims = solara.use_reactive(1000)
    n_games = solara.use_reactive(50)
    win_prob = solara.use_reactive(0.55)
    avg_win = solara.use_reactive(100.0)
    avg_loss = solara.use_reactive(80.0)

    # State for results
    results = solara.use_reactive(None)

    def run_monte_carlo():
        """Run Monte Carlo simulation."""
        simulations = []

        for _ in range(n_sims.value):
            capital = 10000
            equity_curve = [capital]

            for _ in range(n_games.value):
                # Random outcome based on win probability
                if np.random.random() < win_prob.value:
                    pnl = np.random.normal(avg_win.value, avg_win.value * 0.3)
                else:
                    pnl = -np.random.normal(avg_loss.value, avg_loss.value * 0.3)

                capital += pnl
                equity_curve.append(capital)

            simulations.append({
                'final_capital': capital,
                'return': (capital / 10000 - 1) * 100,
                'equity_curve': equity_curve
            })

        results.value = simulations

    with solara.Column():
        solara.Markdown("## Monte Carlo Simulation")
        solara.Markdown("*Simulate thousands of possible outcomes to estimate probability of success*")

        # Parameters
        with solara.Card("Simulation Parameters"):
            solara.SliderInt("Number of Simulations", value=n_sims, min=100, max=10000, step=100)
            solara.SliderInt("Games per Simulation", value=n_games, min=10, max=200, step=10)
            solara.SliderFloat("Win Probability", value=win_prob, min=0.3, max=0.8, step=0.01)
            solara.InputFloat("Average Win ($)", value=avg_win)
            solara.InputFloat("Average Loss ($)", value=avg_loss)

            solara.Button("Run Simulation", on_click=run_monte_carlo, color="primary")

        # Results
        if results.value is not None:
            sims = results.value

            # Summary stats
            returns = [s['return'] for s in sims]
            positive = len([r for r in returns if r > 0])
            negative = len([r for r in returns if r <= 0])

            with solara.Row():
                with solara.Card("Probability of Profit"):
                    prob_profit = positive / len(returns) * 100
                    solara.Markdown(f"### {prob_profit:.1f}%")
                    solara.Markdown(f"**{positive:,} / {len(returns):,} sims profitable**")

                with solara.Card("Expected Return"):
                    exp_return = np.mean(returns)
                    solara.Markdown(f"### {exp_return:.1f}%")
                    solara.Markdown(f"**Median: {np.median(returns):.1f}%**")

                with solara.Card("Risk (Std Dev)"):
                    std = np.std(returns)
                    solara.Markdown(f"### {std:.1f}%")
                    solara.Markdown(f"95% CI: [{np.percentile(returns, 2.5):.1f}%, {np.percentile(returns, 97.5):.1f}%]")

            # Charts
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Return Distribution',
                    'Sample Equity Curves (100 sims)',
                    'Cumulative Probability',
                    'Risk of Ruin'
                )
            )

            # Return distribution
            fig.add_trace(
                go.Histogram(x=returns, nbinsx=50, name='Returns',
                            marker_color='blue'),
                row=1, col=1
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)

            # Sample equity curves
            for i in range(min(100, len(sims))):
                fig.add_trace(
                    go.Scatter(y=sims[i]['equity_curve'], mode='lines',
                              line=dict(width=0.5), showlegend=False,
                              line_color='blue' if sims[i]['return'] > 0 else 'red'),
                    row=1, col=2
                )

            # Cumulative probability
            sorted_returns = sorted(returns)
            cumprob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns) * 100
            fig.add_trace(
                go.Scatter(x=sorted_returns, y=cumprob, mode='lines',
                          name='Cumulative', line=dict(color='purple', width=2)),
                row=2, col=1
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=1)

            # Risk of ruin (probability of losing X%)
            loss_levels = np.arange(-100, 0, 5)
            prob_loss = [len([r for r in returns if r <= level]) / len(returns) * 100
                        for level in loss_levels]
            fig.add_trace(
                go.Scatter(x=loss_levels, y=prob_loss, mode='lines',
                          name='Risk', line=dict(color='red', width=2),
                          fill='tozeroy'),
                row=2, col=2
            )

            fig.update_layout(height=800, showlegend=False)
            fig.update_xaxes(title_text="Return (%)", row=1, col=1)
            fig.update_xaxes(title_text="Game", row=1, col=2)
            fig.update_xaxes(title_text="Return (%)", row=2, col=1)
            fig.update_xaxes(title_text="Loss Level (%)", row=2, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_yaxes(title_text="Capital ($)", row=1, col=2)
            fig.update_yaxes(title_text="Probability (%)", row=2, col=1)
            fig.update_yaxes(title_text="Probability (%)", row=2, col=2)

            solara.FigurePlotly(fig)


@solara.component
def TradeAnalysisView():
    """Analyze individual trades."""

    trades_file = Path("backtest_trades.csv")

    if not trades_file.exists():
        solara.Warning("No trade data found. Run a backtest first.")
        return

    # Load sample of trades (file is large)
    try:
        df = pd.read_csv(trades_file, nrows=10000)
    except:
        solara.Error("Error loading trade data")
        return

    with solara.Column():
        solara.Markdown("## Trade Analysis")
        solara.Markdown(f"*Showing first 10,000 of {len(df):,} trades*")

        # Summary
        with solara.Row():
            with solara.Card("Total Trades"):
                solara.Markdown(f"### {len(df):,}")

            with solara.Card("Avg Trade Size"):
                avg_size = df['quantity'].mean()
                solara.Markdown(f"### {avg_size:.1f}")

            with solara.Card("Avg Price"):
                avg_price = df['price'].mean()
                solara.Markdown(f"### ${avg_price:.3f}")

        # Charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Price Distribution',
                'Quantity Distribution',
                'Trades Over Time',
                'Buy vs Sell'
            )
        )

        # Price distribution
        fig.add_trace(
            go.Histogram(x=df['price'], nbinsx=50, name='Price'),
            row=1, col=1
        )

        # Quantity distribution
        fig.add_trace(
            go.Histogram(x=df['quantity'], nbinsx=50, name='Quantity'),
            row=1, col=2
        )

        # Trades over time
        trades_per_game = df.groupby('game_id').size()
        fig.add_trace(
            go.Scatter(x=trades_per_game.index, y=trades_per_game.values,
                      mode='lines', name='Trades'),
            row=2, col=1
        )

        # Buy vs Sell
        side_counts = df['side'].value_counts()
        fig.add_trace(
            go.Bar(x=side_counts.index, y=side_counts.values, name='Side'),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=False)
        solara.FigurePlotly(fig)


@solara.component
def PerformanceMetricsView():
    """Display detailed performance metrics."""

    equity_file = Path("backtest_equity_curve.csv")

    if not equity_file.exists():
        solara.Warning("No backtest results found.")
        return

    df = pd.read_csv(equity_file)

    if len(df) == 0:
        return

    # Calculate metrics
    returns = df['return'].values
    capital = df['capital'].values

    # Sharpe ratio
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    sortino = np.sqrt(252) * returns.mean() / downside_returns.std() if len(downside_returns) > 0 else 0

    # Max drawdown
    running_max = np.maximum.accumulate(capital)
    drawdown = (capital - running_max) / running_max
    max_dd = drawdown.min() * 100

    # Win/loss stats
    wins = len(returns[returns > 0])
    losses = len(returns[returns <= 0])
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    avg_win = returns[returns > 0].mean() * 100 if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() * 100 if len(returns[returns < 0]) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Calmar ratio
    total_return = (capital[-1] / capital[0] - 1) * 100
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0

    with solara.Column():
        solara.Markdown("## Performance Metrics")

        # Risk-adjusted metrics
        with solara.Card("Risk-Adjusted Returns"):
            cols = solara.Columns([1, 1, 1])

            with cols[0]:
                solara.Markdown("### Sharpe Ratio")
                solara.Markdown(f"**{sharpe:.3f}**")
                if sharpe > 2.0:
                    solara.Success("Excellent (> 2.0)")
                elif sharpe > 1.0:
                    solara.Info("Good (> 1.0)")
                elif sharpe > 0.5:
                    solara.Warning("Marginal (> 0.5)")
                else:
                    solara.Error("Poor (< 0.5)")

            with cols[1]:
                solara.Markdown("### Sortino Ratio")
                solara.Markdown(f"**{sortino:.3f}**")
                solara.Markdown("*Accounts for downside risk only*")

            with cols[2]:
                solara.Markdown("### Calmar Ratio")
                solara.Markdown(f"**{calmar:.3f}**")
                solara.Markdown("*Return / Max Drawdown*")

        # Drawdown analysis
        with solara.Card("Drawdown Analysis"):
            cols = solara.Columns([1, 1, 1])

            with cols[0]:
                solara.Markdown("### Max Drawdown")
                solara.Markdown(f"**{max_dd:.2f}%**")
                if abs(max_dd) < 10:
                    solara.Success("Excellent (< 10%)")
                elif abs(max_dd) < 20:
                    solara.Warning("Acceptable (< 20%)")
                else:
                    solara.Error("High (> 20%)")

            with cols[1]:
                solara.Markdown("### Avg Drawdown")
                avg_dd = drawdown[drawdown < 0].mean() * 100 if len(drawdown[drawdown < 0]) > 0 else 0
                solara.Markdown(f"**{avg_dd:.2f}%**")

            with cols[2]:
                solara.Markdown("### Recovery Factor")
                recovery = total_return / abs(max_dd) if max_dd != 0 else 0
                solara.Markdown(f"**{recovery:.2f}**")

        # Win/Loss analysis
        with solara.Card("Win/Loss Analysis"):
            cols = solara.Columns([1, 1, 1, 1])

            with cols[0]:
                solara.Markdown("### Win Rate")
                solara.Markdown(f"**{win_rate:.1f}%**")
                solara.Markdown(f"{wins} / {wins + losses} games")

            with cols[1]:
                solara.Markdown("### Avg Win")
                solara.Markdown(f"**{avg_win:.2f}%**")

            with cols[2]:
                solara.Markdown("### Avg Loss")
                solara.Markdown(f"**{avg_loss:.2f}%**")

            with cols[3]:
                solara.Markdown("### Profit Factor")
                solara.Markdown(f"**{profit_factor:.2f}**")
                if profit_factor > 2.0:
                    solara.Success("Excellent (> 2.0)")
                elif profit_factor > 1.5:
                    solara.Info("Good (> 1.5)")
                elif profit_factor > 1.0:
                    solara.Warning("Marginal (> 1.0)")
                else:
                    solara.Error("Losing (< 1.0)")

        # Overall recommendation
        with solara.Card("Overall Assessment"):
            score = 0
            if sharpe > 1.0:
                score += 1
            if abs(max_dd) < 15:
                score += 1
            if win_rate > 55:
                score += 1
            if profit_factor > 1.5:
                score += 1

            if score >= 3:
                solara.Success("✅ GOOD - Strategy shows promise! Consider paper trading.")
            elif score >= 2:
                solara.Warning("⚠️ MARGINAL - Needs optimization before live trading.")
            else:
                solara.Error("❌ POOR - Do NOT trade! Requires significant improvement.")
