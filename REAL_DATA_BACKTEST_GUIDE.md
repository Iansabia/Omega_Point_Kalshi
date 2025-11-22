# 📊 Real Historical Data Backtesting Guide

Complete guide to backtesting your trading strategy on **real Kalshi historical data**.

---

## 🎯 Overview

You now have two backtesting modes:

1. **Synthetic Data** (default) - Simulated markets for quick testing
2. **Real Historical Data** - Actual Kalshi NFL markets from 2023-2024

---

## 🚀 Quick Start

### Step 1: Set Up Kalshi Credentials

```bash
# Copy template
cp .env.template .env

# Edit .env with your credentials
nano .env
```

Add your Kalshi credentials:
```bash
KALSHI_EMAIL=your_email@example.com
KALSHI_PASSWORD=your_password
KALSHI_BASE_URL=https://demo-api.kalshi.co  # or production URL
```

### Step 2: Download Historical Data

```bash
# Download 2024 NFL season data
python scripts/download_historical_data.py --season 2024

# Or download all available seasons
python scripts/download_historical_data.py --all
```

**Output:**
```
================================================================================
KALSHI HISTORICAL DATA DOWNLOADER
================================================================================

📥 Loading REAL historical data for 2024 season...
🌐 Fetching settled NFL markets from Kalshi API...
✅ Found 156 markets

📥 Downloading candlesticks for 156 markets...
   Interval: 60 minutes
   🌐 Downloaded 10/156
   🌐 Downloaded 20/156
   ...
✅ Downloaded 156 candlestick datasets

💾 Cache Statistics:
   Market files: 1
   Candlestick files: 156
   Total size: 12.5 MB
   Location: data/historical

✅ Data ready for backtesting!
```

### Step 3: Run Backtest with Real Data

```bash
# Run backtest on real 2024 data
python run_backtest.py --use-real-data --season 2024 --games 50
```

**Output:**
```
🎯 Using REAL Kalshi historical data (2024 season)

================================================================================
RUNNING HISTORICAL BACKTEST
================================================================================

Configuration:
  Games: 50
  Initial Capital: $10,000.00
  Transaction Cost: 10 bps
  Agents: 42

1️⃣  Generating historical scenarios...
📥 Loading REAL historical data for 2024 season...
📂 Loading cached markets from data/historical/markets/nfl_settled_2024.csv
   📂 Loaded 10/50 from cache
   📂 Loaded 20/50 from cache
   ...
✅ Loaded 50 historical game scenarios
   Average price path length: 48.2 candlesticks

2️⃣  Simulating trading on historical data...
   Processing game 10/50...
   Processing game 20/50...
   ...
   ✅ Completed 50 games

3️⃣  Calculating performance metrics...

================================================================================
BACKTEST RESULTS
================================================================================

📊 TRADING STATISTICS
  Total Games: 50
  Total Trades: 8,543
  Trades per Game: 170.9

💰 CAPITAL & RETURNS
  Initial Capital: $10,000.00
  Final Capital: $12,450.00
  Total P&L: $2,450.00
  Total Return: 24.50%

📈 RISK-ADJUSTED METRICS
  Sharpe Ratio: 1.245
  Sortino Ratio: 1.678
  Calmar Ratio: 2.156
  Max Drawdown: -11.36%

🎯 WIN/LOSS ANALYSIS
  Win Rate: 62.00%
  Wins: 31
  Losses: 19
  Avg Win: 8.45%
  Avg Loss: -4.23%
  Profit Factor: 2.00

💡 RECOMMENDATION
================================================================================
✅ GOOD - Strategy shows promise!
   → Sharpe > 1.0 and Max DD < 15%
   → Safe to proceed to paper trading
```

---

## 📝 Detailed Instructions

### Download Options

#### Download Specific Season
```bash
python scripts/download_historical_data.py --season 2024
```

#### Download All Seasons
```bash
python scripts/download_historical_data.py --all
```

#### Test with Limited Markets
```bash
python scripts/download_historical_data.py --season 2024 --max-markets 10
```

#### Change Candlestick Interval
```bash
# 1-minute candles (high granularity, large files)
python scripts/download_historical_data.py --season 2024 --interval 1

# 1-hour candles (recommended balance)
python scripts/download_historical_data.py --season 2024 --interval 60

# Daily candles (low granularity, small files)
python scripts/download_historical_data.py --season 2024 --interval 1440
```

#### Clear Cache and Re-download
```bash
python scripts/download_historical_data.py --season 2024 --clear-cache
```

### Backtest Options

#### Real Data Backtest
```bash
# 50 games from 2024
python run_backtest.py --use-real-data --season 2024 --games 50

# 100 games from 2023
python run_backtest.py --use-real-data --season 2023 --games 100

# Custom agent count
python run_backtest.py --use-real-data --season 2024 --agents 20
```

#### Synthetic Data Backtest (Default)
```bash
# Still works as before
python run_backtest.py --games 100
```

---

## 📂 Data Structure

```
data/historical/
├── markets/
│   ├── nfl_settled_2023.csv     # 2023 season markets
│   ├── nfl_settled_2024.csv     # 2024 season markets
│   └── nfl_settled_all.csv      # All seasons
└── candlesticks/
    ├── HIGHFB-24SEP15-B-KC.csv  # Chiefs vs Ravens 9/15
    ├── HIGHFB-24SEP22-B-SF.csv  # 49ers game 9/22
    └── ... (one file per market)
```

### Market File Columns
- `ticker`: Market ticker (e.g., HIGHFB-24SEP15-B-KC)
- `series_ticker`: Series (e.g., HIGHFB)
- `title`: Market description
- `status`: Market status (settled, closed, open)
- `result`: Outcome (yes/no)
- `event_ticker`: Event identifier

### Candlestick File Columns
- `start_ts`: Candle start timestamp
- `end_ts`: Candle end timestamp
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

---

## 🔍 Verify Real Data

### Check What's Cached
```python
from src.data.kalshi_historical import KalshiHistoricalDataFetcher

fetcher = KalshiHistoricalDataFetcher()
stats = fetcher.get_cache_stats()

print(f"Markets: {stats['market_files']}")
print(f"Candlesticks: {stats['candlestick_files']}")
print(f"Size: {stats['total_size_mb']:.2f} MB")
```

### Inspect Market Data
```bash
# View market metadata
cat data/historical/markets/nfl_settled_2024.csv | head -10

# View candlestick data
cat data/historical/candlesticks/HIGHFB-24SEP15-B-KC.csv | head -10
```

---

## ⚙️ Programmatic Usage

### Load Historical Data
```python
from src.data.kalshi_historical import KalshiHistoricalDataFetcher

# Initialize fetcher
fetcher = KalshiHistoricalDataFetcher(cache_dir="data/historical")

# Fetch markets
markets = fetcher.fetch_nfl_markets(
    season=2024,
    status="settled",
    use_cache=True
)

print(f"Found {len(markets)} markets")

# Download candlesticks
candlesticks = fetcher.download_candlesticks(
    markets,
    period_interval=60,  # 1-hour candles
    use_cache=True,
    max_markets=50
)

print(f"Downloaded {len(candlesticks)} price histories")
```

### Load Backtest-Ready Data
```python
# Get formatted scenarios
scenarios = fetcher.load_historical_backtest_data(
    season=2024,
    max_games=50
)

for scenario in scenarios[:3]:
    print(f"Ticker: {scenario['ticker']}")
    print(f"Title: {scenario['title']}")
    print(f"Initial Price: ${scenario['initial_price']:.2f}")
    print(f"Outcome: {'YES' if scenario['outcome'] == 1 else 'NO'}")
    print(f"Price Points: {len(scenario['price_path'])}")
    print(f"Real Data: {scenario['is_real_data']}")
    print()
```

### Run Backtest Programmatically
```python
from run_backtest import HistoricalBacktest

# Configure backtest
backtest = HistoricalBacktest(
    n_games=50,
    initial_capital=10000,
    use_real_data=True,
    season=2024
)

# Run
results = backtest.run_backtest()

# Analyze
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

---

## 🎯 Comparison: Real vs Synthetic Data

| Feature | Synthetic Data | Real Historical Data |
|---------|---------------|---------------------|
| **Speed** | ⚡ Very Fast | 🐢 Slower (API calls) |
| **Accuracy** | 🎲 Approximate | ✅ Exact historical |
| **Cost** | 💰 Free | 💰 Free (API access required) |
| **Setup** | ✅ None | ⚙️ Credentials + Download |
| **Repeatability** | ❌ Random | ✅ Identical each run |
| **Market Realism** | 🤔 Simulated | 💯 Actual behavior |
| **Use Case** | Quick testing | Final validation |

---

## 📊 Example Workflow

```bash
# 1. Quick test with synthetic data
python run_backtest.py --games 10 --agents 20

# 2. Download real data
python scripts/download_historical_data.py --season 2024 --max-markets 20

# 3. Validate on small real dataset
python run_backtest.py --use-real-data --season 2024 --games 20

# 4. Full backtest if initial results good
python scripts/download_historical_data.py --season 2024  # Get all markets
python run_backtest.py --use-real-data --season 2024 --games 100

# 5. View results
solara run backtest_viewer.py  # Open http://localhost:8765
```

---

## ⚠️ Important Notes

### Rate Limits
- Kalshi API has rate limits
- Download script adds 0.2s delay between requests
- For large datasets, expect 5-10 minutes download time

### Data Availability
- Historical data depends on Kalshi API
- Not all markets may have complete candlestick data
- Some markets may be missing or have gaps

### Authentication
- Demo accounts have $10,000 virtual money
- Production accounts use real money (be careful!)
- Tokens expire every 30 minutes (auto-renewed)

### Cache Management
- Data is cached automatically
- Use `--clear-cache` to force re-download
- Cache size grows with more seasons (~10-50 MB/season)

---

## 🐛 Troubleshooting

### "No markets found"
```bash
# Check your credentials
cat .env | grep KALSHI

# Verify API access
python test_kalshi_connection.py

# Try different season
python scripts/download_historical_data.py --season 2023
```

### "Authentication failed"
```bash
# Check credentials in .env
# Make sure no extra spaces or quotes
# Verify account has API access enabled
```

### "No candlestick data"
```bash
# Some markets may not have historical data
# Try different interval (60 or 1440)
# Check market was actually settled
```

### "Import error"
```bash
# Make sure you're in venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 📈 Next Steps

1. **Download historical data** for 2024 season
2. **Run backtest** on real data
3. **Compare results** to synthetic backtest
4. **Analyze performance** in viewer (http://localhost:8766)
5. **Optimize strategy** based on real market behavior
6. **Re-backtest** until profitable (Sharpe > 1.0)
7. **Paper trade** on Kalshi demo account
8. **Live trade** only after 30+ days of profitable paper trading

---

## 💡 Tips

- Start with `--max-markets 10` to test quickly
- Use 60-minute interval for balance of detail and speed
- Compare real vs synthetic to understand model accuracy
- Focus on Sharpe > 1.0 and Max DD < 15% for real data
- Real data may show lower returns than synthetic (more realistic)

---

**Ready to backtest on real data?**

```bash
python scripts/download_historical_data.py --season 2024
python run_backtest.py --use-real-data --season 2024 --games 50
```

Good luck! 🍀
