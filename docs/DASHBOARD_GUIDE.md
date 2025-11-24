# 🏈 Monday Night Football Trading Dashboard

## Quick Start

### 1. Make Sure You Have a New API Key

Before running the dashboard, ensure you've generated a new Kalshi API key (see `GENERATE_NEW_KEY.md`).

### 2. Launch the Dashboard

```bash
PYTHONPATH=. ./venv/bin/python3 scripts/run_dashboard_trading.py
```

### 3. Open in Browser

```
http://localhost:8000
```

---

## Dashboard Features

### 🏈 Game State (Top Left)
- **Live Score**: Real-time score updates from ESPN
- **Quarter & Clock**: Current game time
- **Possession**: Which team has the ball
- **Field Position**: Yardline (0-100)
- **Down & Distance**: Current down and yards to go

### 💰 Market State (Top Right)
- **Bid/Ask/Mid**: Current Kalshi market prices
- **Spread**: Bid-ask spread
- **Volume**: Trading volume

### 🤖 Model Prediction (Bottom Left)
- **Win Probability**: Model's prediction vs market price
- **Edge**: How much the model disagrees with the market
- **Signal Indicator**: BUY/SELL signal when edge > threshold

### 📊 Performance (Bottom Right)
- **Total Trades**: Number of trades executed
- **Win Rate**: Percentage of profitable trades
- **Total P&L**: Profit/Loss for the session
- **Balance**: Current account balance

### ⚙️ Controls
- **Paper Trading**: Toggle between paper and live trading
- **Auto Trading**: Enable/disable automatic trade execution
- **Min Edge**: Minimum edge required for a trade (default 10%)
- **Max Position**: Maximum position size per trade

### 📈 Recent Trades
- Shows last 20 trades
- Color coded: Green (BUY), Red (SELL)
- Timestamp for each trade

### 🔌 Circuit Breakers
- **Kalshi API**: Status of Kalshi API circuit breaker
- **ESPN API**: Status of ESPN API circuit breaker
- Color coded: Green (CLOSED), Red (OPEN), Yellow (HALF_OPEN)

---

## How It Works

### Data Flow

```
ESPN API (2s polling)
    ↓
Game State → Dashboard
    ↓
Kalshi WebSocket (real-time)
    ↓
Market Prices → Dashboard
    ↓
Win Probability Model
    ↓
Arbitrage Detector
    ↓
Risk Manager
    ↓
Trade Execution → Dashboard
```

### Real-Time Updates

The dashboard uses **WebSocket** for real-time updates:
- Game state updates every 2 seconds (ESPN poll interval)
- Market prices update in real-time (Kalshi WebSocket)
- Model predictions recalculated on every game state change
- Trade notifications appear instantly

---

## Configuration

### Trading Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Min Edge** | 10% | 0-100% | Minimum edge to trigger a trade |
| **Max Position** | 100 | 10-1000 | Maximum contracts per trade |
| **Paper Trading** | ON | ON/OFF | Safe mode (no real money) |
| **Auto Trading** | OFF | ON/OFF | Automatic trade execution |

### Recommended Settings for First Game

```
Paper Trading: ON
Auto Trading: OFF (manual approval)
Min Edge: 10% (conservative)
Max Position: 10 (start small)
```

---

## Usage Scenarios

### Scenario 1: Paper Trading (Recommended First)

1. **Launch dashboard** with paper trading ON
2. **Watch the game** and observe signals
3. **Manually approve trades** when you see good opportunities
4. **Review performance** after the game

**Goal**: Learn how the system behaves without risking money

### Scenario 2: Semi-Automatic Trading

1. **Paper trading OFF**, **Auto trading OFF**
2. Dashboard shows BUY/SELL signals
3. **You click "Execute"** to approve each trade
4. System places orders on Kalshi

**Goal**: Have final control over every trade

### Scenario 3: Fully Automatic Trading (Advanced)

1. **Paper trading OFF**, **Auto trading ON**
2. System executes trades automatically when:
   - Edge > Min Edge threshold
   - Risk checks pass
   - Circuit breakers are closed
3. **Monitor dashboard** for issues

**Goal**: Hands-off trading (use with caution!)

---

## What to Watch For

### 🟢 Good Signals
- **Large edge** (>15%): Model strongly disagrees with market
- **Clear game momentum**: Touchdown, interception, turnover
- **Tight spread** (<5¢): Easy to execute trade
- **Fresh data**: Both ESPN and Kalshi updates recent

### 🔴 Bad Signals
- **Small edge** (<10%): Not worth the risk
- **Wide spread** (>10¢): Hard to get filled
- **Stale data**: ESPN or Kalshi not updating
- **Circuit breaker open**: API issues

### ⚠️ Warning Signs
- **Circuit breaker opens**: API is down, stop trading
- **Many losing trades**: Model may be off, reduce position size
- **Large spread**: Market is illiquid, be careful
- **Late in game**: Less time for mean reversion

---

## Keyboard Shortcuts (Future)

| Key | Action |
|-----|--------|
| `Space` | Start/Stop trading |
| `R` | Refresh data |
| `P` | Toggle paper trading |
| `A` | Toggle auto trading |

---

## Troubleshooting

### Dashboard won't load

```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill existing process
kill -9 <PID>

# Restart dashboard
PYTHONPATH=. ./venv/bin/python3 scripts/run_dashboard_trading.py
```

### No game data showing

1. Check ESPN API is working:
   ```bash
   PYTHONPATH=. ./venv/bin/python3 scripts/test_espn_api.py
   ```
2. Verify game is scheduled for tonight
3. Check console for error messages

### No market prices showing

1. Check Kalshi API credentials in `.env`
2. Verify API key is not revoked
3. Check circuit breaker status on dashboard
4. Look for market that matches tonight's game

### WebSocket disconnected

- Dashboard will automatically reconnect every 5 seconds
- Check your internet connection
- Verify firewall isn't blocking WebSocket

---

## Advanced Features

### Custom Signals

You can manually trigger trades from the dashboard:

1. Click "Execute Trade" button
2. Enter trade details:
   - Ticker
   - Side (BUY/SELL)
   - Quantity
   - Price
3. System executes immediately (bypasses model)

### Export Performance Data

```python
# Query audit log for tonight's trades
from src.execution.audit_log import audit_logger
import time

tonight = time.time() - 3600 * 4  # Last 4 hours
trades = audit_logger.wal.query(
    event_type="TRADE",
    start_time=tonight
)

# Export to CSV
import csv
with open('trades.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=trades[0].to_dict().keys())
    writer.writeheader()
    for trade in trades:
        writer.writerow(trade.to_dict())
```

---

## Dashboard API

The dashboard exposes a REST API for integration:

### GET /api/state
```bash
curl http://localhost:8000/api/state
```

Returns current dashboard state (game, market, model, trades)

### GET /api/circuit-breaker
```bash
curl http://localhost:8000/api/circuit-breaker
```

Returns circuit breaker status for all APIs

### GET /api/audit-log?limit=50
```bash
curl http://localhost:8000/api/audit-log?limit=50
```

Returns last 50 audit log entries

### POST /api/config
```bash
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"min_edge": 0.15, "auto_trading": true}'
```

Update dashboard configuration

---

## Performance Tips

### For Best Results

1. **Start early**: Launch dashboard 15 minutes before kickoff
2. **Monitor actively**: Watch for the first 30 minutes to see how model performs
3. **Start conservative**: Use paper trading or small positions first
4. **Adjust thresholds**: If too many/few signals, adjust min edge
5. **Watch spread**: Only trade when spread < 5¢ for easy fills

### When to Stop Trading

- Circuit breaker opens (API issues)
- Losing streak of 3+ trades
- Game is in 4th quarter with <5 minutes (too volatile)
- You're uncomfortable with position size
- Market becomes illiquid (spread > 10¢)

---

## Example Session

**7:30 PM**: Launch dashboard, verify connection to ESPN and Kalshi

**8:00 PM**: Kickoff, game state starts updating

**8:15 PM**: First touchdown, model shows 75% home win but market at 85%
- Signal: SELL (edge = 10%)
- You approve the trade manually
- Order placed on Kalshi

**8:30 PM**: Market corrects to 76%, close to model
- No new signals (edge < 10%)

**8:45 PM**: Interception by away team, model now shows 55% away win, market at 70%
- Signal: BUY away team (edge = 15%)
- Automatic trade executed

**10:00 PM**: Game ends, review performance
- Total trades: 5
- Winning trades: 4
- Win rate: 80%
- Total P&L: +$127.50

**10:15 PM**: Export trades to CSV for analysis

---

## Safety Features

The dashboard includes multiple safety features:

1. **Circuit Breakers**: Automatically stop trading if APIs fail
2. **Risk Manager**: Position limits, daily loss limits
3. **Paper Trading Mode**: Test without real money
4. **Manual Approval**: Option to approve each trade
5. **Audit Log**: Complete trail of all actions

---

## Next Steps

1. **Tonight**: Run paper trading and observe
2. **Tomorrow**: Review audit log and performance
3. **Next Game**: Try semi-automatic with small positions
4. **Later**: Tune parameters based on results

---

**Support**: See `SECURITY_INCIDENT_REMEDIATION.md` and `docs/CIRCUIT_BREAKER_AUDIT_LOG_INTEGRATION.md`

**Good luck! 🏈🚀**
