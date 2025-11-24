# Kalshi Omega Point

Agent-Based Model (ABM) for prediction market trading on Kalshi. This system simulates and executes trading strategies using multi-agent modeling with real-time market integration.

## Features

- **Agent-Based Modeling**: Multiple agent types (informed traders, noise traders, market makers, arbitrageurs, LLM agents)
- **Order Book Simulation**: Full limit order book with realistic matching engine
- **Risk Management**: Integrated position limits, stop losses, and Kelly Criterion sizing
- **Backtesting**: Event-driven backtesting with real historical data from Kalshi
- **Real-Time Trading**: Live integration with Kalshi API for paper and live trading
- **Visualization**: Interactive Solara dashboards for real-time monitoring

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Kalshi_Omega_Point

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the environment template and configure your credentials:

```bash
cp .env.template .env
# Edit .env with your API keys
```

Get your Kalshi API credentials:
1. Visit https://kalshi.com/profile/api-keys
2. Create a new API key and download the private key file
3. Update `.env` with your credentials

### 3. Run a Simulation

```bash
# Simple simulation
python scripts/run_simple_sim.py

# Backtest on historical data
python scripts/run_backtest.py --games 100 --agents 50

# Launch interactive dashboard
python scripts/dashboard_simple.py
```

## Project Structure

```
.
├── src/
│   ├── agents/           # Trading agent implementations
│   ├── backtesting/      # Backtesting engine and metrics
│   ├── data/             # Data fetching and processing
│   ├── execution/        # Order execution and API clients
│   ├── models/           # Market models and simulations
│   ├── orderbook/        # Order book and matching engine
│   ├── risk/             # Risk management system
│   └── visualization/    # Dashboard and monitoring
├── tests/                # Test suite
├── scripts/              # Runnable scripts
├── config/               # Configuration files
├── docs/                 # Documentation
├── results/              # Backtest results and outputs
└── data/                 # Historical data cache

```

## Documentation

### Getting Started
- [Quick Start Guide](docs/guides/QUICKSTART.md)
- [Backtesting Guide](docs/guides/REAL_DATA_BACKTEST_GUIDE.md)
- [Paper Trading Setup](docs/guides/PAPER_TRADING_SETUP.md)

### Technical Documentation
- [API Reference](docs/API_REFERENCE.md)
- [Risk Management](docs/RISK_MANAGEMENT_STATUS.md)
- [Validation Report](docs/VALIDATION_REPORT.md)
- [Backtest Analysis](docs/BACKTEST_ANALYSIS.md)
- [Progress Summary](docs/PROGRESS_SUMMARY.md)

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_agents.py
```

## Code Quality

This project uses automated code formatting:

```bash
# Format code
black .
isort .

# Run linting
flake8 .
mypy src/
```

## Architecture

The system is built on several key components:

1. **Market Model**: Mesa-based agent simulation framework
2. **Order Book**: High-performance limit order book with price-time priority
3. **Agents**: Multiple agent types with different trading strategies
4. **Risk Manager**: Real-time risk controls and position management
5. **Execution**: Kalshi API integration with order routing
6. **Backtesting**: Event-driven backtesting on historical data

## Trading Agents

- **Noise Traders**: Random traders providing liquidity
- **Informed Traders**: Trade based on information signals
- **Market Makers**: Provide two-sided liquidity
- **Arbitrageurs**: Exploit price inefficiencies
- **LLM Agents**: Use language models for decision making (Gemini integration)

## Development Status

Current status: **87% production ready**

See [PROGRESS_SUMMARY.md](docs/PROGRESS_SUMMARY.md) for detailed status.

## License

MIT License - see LICENSE file for details

## Contributing

This is a research/trading project. Contact the maintainer for collaboration opportunities.
