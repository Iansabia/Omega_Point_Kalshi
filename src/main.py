import os
import logging
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_dir: str = "config"):
    """Load configuration files."""
    config = {}
    config_path = Path(config_dir)
    
    files = ["base_config.yaml", "agent_profiles.yaml", "market_config.yaml"]
    for f in files:
        path = config_path / f
        if path.exists():
            with open(path, "r") as file:
                config[path.stem] = yaml.safe_load(file)
                logger.info(f"Loaded config: {f}")
        else:
            logger.warning(f"Config file not found: {f}")
            
    return config

def main():
    """Main entry point for the Prediction Market ABM."""
    logger.info("Starting Prediction Market ABM...")
    
    # Load configuration
    config = load_config()
    logger.info(f"Configuration loaded: {list(config.keys())}")
    
    # TODO: Initialize Market Model
    # TODO: Run Simulation
    
    logger.info("Simulation complete.")

if __name__ == "__main__":
    main()
