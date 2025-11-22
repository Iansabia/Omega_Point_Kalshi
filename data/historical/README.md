# Historical Data Cache

This directory stores cached historical market data from Kalshi API for backtesting.

## Directory Structure

```
data/historical/
├── markets/          # Market metadata CSV files
│   ├── nfl_settled_2023.csv
│   ├── nfl_settled_2024.csv
│   └── nfl_settled_all.csv
├── candlesticks/     # Price history per market
│   ├── HIGHFB-24SEP15-B-KC.csv
│   ├── HIGHFB-24SEP22-B-SF.csv
│   └── ...
└── README.md         # This file
```

## Files

### Market Files (`markets/`)
- **Format**: CSV with market metadata
- **Columns**: ticker, series_ticker, title, status, result, event_ticker, etc.
- **Purpose**: Index of all markets for a season/status

### Candlestick Files (`candlesticks/`)
- **Format**: CSV with OHLC price data
- **Columns**: start_ts, end_ts, open, high, low, close, volume
- **Purpose**: Price history for backtesting
- **Interval**: 60 minutes (configurable)

## Usage

### Download Data
```bash
# Download 2024 season
python scripts/download_historical_data.py --season 2024

# Download all seasons
python scripts/download_historical_data.py --all

# Test with limited markets
python scripts/download_historical_data.py --season 2024 --max-markets 10
```

### Use in Backtest
```bash
# Run backtest with real data
python run_backtest.py --use-real-data --season 2024 --games 50

# Run with synthetic data (default)
python run_backtest.py --games 100
```

### Programmatic Access
```python
from src.data.kalshi_historical import KalshiHistoricalDataFetcher

fetcher = KalshiHistoricalDataFetcher()

# Fetch markets
markets = fetcher.fetch_nfl_markets(season=2024, status="settled")

# Download candlesticks
candlesticks = fetcher.download_candlesticks(markets)

# Load formatted backtest data
scenarios = fetcher.load_historical_backtest_data(season=2024)
```

## Data Source

- **API**: Kalshi API v2
- **Endpoint**: `https://trading-api.kalshi.com/trade-api/v2`
- **Series**: HIGHFB (NFL high scorer markets)
- **Documentation**: https://docs.kalshi.com

## Cache Management

### Check Cache Stats
```python
fetcher = KalshiHistoricalDataFetcher()
stats = fetcher.get_cache_stats()
print(stats)
```

### Clear Cache
```python
fetcher = KalshiHistoricalDataFetcher()
fetcher.clear_cache()
```

Or manually:
```bash
rm -rf data/historical/markets/*.csv
rm -rf data/historical/candlesticks/*.csv
```

## Notes

- Data is cached to avoid repeated API calls
- Use `--clear-cache` flag to force re-download
- Rate limiting: 0.2s delay between candlestick requests
- Pagination: Automatically handles large result sets
- Data is stored in CSV format for easy inspection

## .gitignore

This directory is listed in `.gitignore` to avoid committing large data files.
Download data locally as needed.
