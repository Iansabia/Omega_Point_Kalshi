import os
import logging
import yaml
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Import the market model
from src.models.market_model import PredictionMarketModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_dir: str = "config") -> Dict[str, Any]:
    """Load configuration files."""
    config = {}
    config_path = Path(config_dir)

    files = ["base_config.yaml", "agent_profiles.yaml", "market_config.yaml"]
    for f in files:
        path = config_path / f
        if path.exists():
            with open(path, "r") as file:
                loaded = yaml.safe_load(file)
                if loaded:  # Check if file is not empty
                    config[path.stem.replace('_config', '')] = loaded
                    logger.info(f"Loaded config: {f}")
        else:
            logger.warning(f"Config file not found: {f}")

    return config

def run_simulation(model: PredictionMarketModel, steps: int, verbose: bool = False):
    """
    Run the simulation for a specified number of steps.

    Args:
        model: The market model instance
        steps: Number of simulation steps to run
        verbose: Whether to print detailed progress
    """
    logger.info(f"Starting simulation for {steps} steps...")

    for i in range(steps):
        model.step()

        if verbose and (i + 1) % 100 == 0:
            logger.info(f"Step {i + 1}/{steps} - Price: {model.current_price:.4f}, "
                       f"Spread: {model.get_spread():.4f}, "
                       f"Volume: {model.calculate_volume():.2f}")

    logger.info(f"Simulation complete after {steps} steps")

def save_results(model: PredictionMarketModel, output_dir: str = "results"):
    """
    Save simulation results to files.

    Args:
        model: The completed market model
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save model-level data
    model_data = model.datacollector.get_model_vars_dataframe()
    model_data.to_csv(output_path / "model_data.csv")
    logger.info(f"Saved model data to {output_path / 'model_data.csv'}")

    # Save agent-level data
    agent_data = model.datacollector.get_agent_vars_dataframe()
    if not agent_data.empty:
        agent_data.to_csv(output_path / "agent_data.csv")
        logger.info(f"Saved agent data to {output_path / 'agent_data.csv'}")

    # Save summary statistics
    summary = {
        "total_steps": model.step_count,
        "final_price": model.current_price,
        "total_volume": model.calculate_volume(),
        "total_trades": len(model.matching_engine.trades),
        "final_spread": model.get_spread(),
        "llm_cost": model.cumulative_llm_cost,
        "num_agents": len(list(model.schedule.agents))
    }

    with open(output_path / "summary.yaml", "w") as f:
        yaml.dump(summary, f)
    logger.info(f"Saved summary to {output_path / 'summary.yaml'}")

    # Print summary
    logger.info("=" * 60)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 60)
    for key, value in summary.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 60)

def print_agent_summary(model: PredictionMarketModel):
    """Print summary of agent performance."""
    logger.info("=" * 60)
    logger.info("AGENT PERFORMANCE SUMMARY")
    logger.info("=" * 60)

    agent_stats = {}
    for agent in model.schedule.agents:
        agent_type = agent.__class__.__name__
        if agent_type not in agent_stats:
            agent_stats[agent_type] = {
                'count': 0,
                'total_wealth': 0,
                'total_position': 0,
                'total_trades': 0,
                'total_pnl': 0
            }

        stats = agent_stats[agent_type]
        stats['count'] += 1
        stats['total_wealth'] += agent.wealth
        stats['total_position'] += agent.position
        stats['total_trades'] += len(agent.trade_history)
        stats['total_pnl'] += agent.calculate_pnl(model.current_price)

    for agent_type, stats in agent_stats.items():
        count = stats['count']
        logger.info(f"\n{agent_type} (n={count}):")
        logger.info(f"  Avg Wealth: ${stats['total_wealth']/count:.2f}")
        logger.info(f"  Avg Position: {stats['total_position']/count:.2f}")
        logger.info(f"  Avg Trades: {stats['total_trades']/count:.1f}")
        logger.info(f"  Avg PnL: ${stats['total_pnl']/count:.2f}")

    logger.info("=" * 60)

def main():
    """Main entry point for the Prediction Market ABM."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Prediction Market ABM Simulation")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config-dir", type=str, default="config", help="Configuration directory")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PREDICTION MARKET AGENT-BASED MODEL")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(args.config_dir)
    logger.info(f"Configuration loaded: {list(config.keys())}")

    # Override with command-line arguments
    if 'base' not in config:
        config['base'] = {}
    if 'simulation' not in config['base']:
        config['base']['simulation'] = {}

    config['base']['simulation']['seed'] = args.seed
    config['base']['simulation']['steps'] = args.steps

    # Initialize Market Model
    logger.info(f"Initializing market model with seed={args.seed}...")
    model = PredictionMarketModel(
        config=config.get('base', {}),
        agent_config=config.get('agent_profiles', {}),
        seed=args.seed
    )

    logger.info(f"Model initialized with {len(list(model.schedule.agents))} agents")

    # Run Simulation
    try:
        run_simulation(model, steps=args.steps, verbose=args.verbose)

        # Print agent performance
        print_agent_summary(model)

        # Save results
        save_results(model, output_dir=args.output_dir)

        logger.info("Simulation completed successfully!")

    except Exception as e:
        logger.error(f"Simulation failed with error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
