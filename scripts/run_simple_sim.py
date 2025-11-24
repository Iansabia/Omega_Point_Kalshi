"""
Simple simulation runner to test the prediction market model.

Run with: python run_simple_sim.py
"""

import sys

sys.path.insert(0, "/Users/jaredmarcus/projects/omega_point/Omega_Point_Kalshi")

import matplotlib.pyplot as plt
import pandas as pd

from src.models.market_model import PredictionMarketModel

# Configure agents
agent_config = {
    "noise_trader": {"count": 50, "wealth": 1000},
    "informed_trader": {"count": 10, "wealth": 10000, "information_quality": 0.8},
    "market_maker": {"count": 2, "wealth": 100000, "risk_param": 0.1},
}

market_config = {"initial_price": 0.5, "tick_size": 0.01}

# Create model
print("Creating prediction market model...")
model = PredictionMarketModel(agent_config=agent_config, config={"market": market_config}, seed=42)

print(f"Model initialized with {len(list(model.agents))} agents")
print(f"Initial price: {model.current_price}")
print(f"Fundamental value: {model.fundamental_value}")
print("\nRunning simulation for 100 steps...")

# Run simulation
num_steps = 100
for i in range(num_steps):
    model.step()
    if (i + 1) % 10 == 0:
        print(f"  Step {i + 1}/{num_steps} - Price: {model.current_price:.3f}")

print("\n✅ Simulation complete!")

# Get results
model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()

# Display summary statistics
print("\n" + "=" * 60)
print("SIMULATION RESULTS")
print("=" * 60)

print(f"\nMarket Statistics:")
print(f"  Final Price: ${model_data['market_price'].iloc[-1]:.3f}")
print(f"  Price Range: ${model_data['market_price'].min():.3f} - ${model_data['market_price'].max():.3f}")
print(f"  Total Volume: {model_data['total_volume'].sum():.0f}")
print(f"  Avg Spread: {model_data['bid_ask_spread'].mean():.4f}")

# Agent statistics
latest_step = agent_data.index.get_level_values("Step").max()
latest_agents = agent_data.xs(latest_step, level="Step")

print(f"\nAgent Statistics:")
print(f"  Total Agents: {len(latest_agents)}")
print(f"  Avg Wealth: ${latest_agents['wealth'].mean():.2f}")
print(f"  Wealth Std Dev: ${latest_agents['wealth'].std():.2f}")
print(f"  Total Positions: {latest_agents['position'].sum():.0f}")

# Top performers
print(f"\nTop 5 Agents by Wealth:")
top_5 = latest_agents.nlargest(5, "wealth")[["wealth", "position", "agent_type"]]
for idx, row in top_5.iterrows():
    agent_id = idx
    print(f"  Agent {agent_id}: ${row['wealth']:.2f} (pos: {row['position']:.0f}, type: {row['agent_type']})")

# Create simple plots
print("\nGenerating plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Price chart
axes[0, 0].plot(model_data.index, model_data["market_price"], label="Market Price", linewidth=2)
axes[0, 0].axhline(y=model.fundamental_value, color="r", linestyle="--", label="Fundamental Value")
axes[0, 0].set_title("Market Price vs Fundamental Value")
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Price")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Volume chart
axes[0, 1].bar(model_data.index, model_data["total_volume"], color="green", alpha=0.6)
axes[0, 1].set_title("Trading Volume")
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("Volume")
axes[0, 1].grid(True, alpha=0.3)

# Spread chart
axes[1, 0].fill_between(model_data.index, model_data["bid_ask_spread"], alpha=0.5, color="orange")
axes[1, 0].set_title("Bid-Ask Spread")
axes[1, 0].set_xlabel("Step")
axes[1, 0].set_ylabel("Spread")
axes[1, 0].grid(True, alpha=0.3)

# Wealth distribution
latest_agents["wealth"].hist(bins=30, ax=axes[1, 1], color="blue", alpha=0.6)
axes[1, 1].set_title("Wealth Distribution (Final Step)")
axes[1, 1].set_xlabel("Wealth ($)")
axes[1, 1].set_ylabel("Count")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("simulation_results.png", dpi=150, bbox_inches="tight")
print("✅ Plots saved to simulation_results.png")

# Save data
model_data.to_csv("model_data.csv")
agent_data.to_csv("agent_data.csv")
print("✅ Data saved to model_data.csv and agent_data.csv")

print("\n" + "=" * 60)
print("✨ All done! Check the generated files:")
print("   - simulation_results.png (charts)")
print("   - model_data.csv (market data)")
print("   - agent_data.csv (agent data)")
print("=" * 60)
