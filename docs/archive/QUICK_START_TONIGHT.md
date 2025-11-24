# 🏈 Quick Start for Tonight's Game

## Setup (Do This First!)

### 1. Generate New API Key

⚠️ **CRITICAL**: You need a new Kalshi API key (old one was revoked)

```bash
# Follow this guide:
cat GENERATE_NEW_KEY.md

# Quick version:
# 1. Go to https://kalshi.com → Settings → API Keys → Create New Key
# 2. Download private key
# 3. Move to secure location:
mkdir -p ~/.ssh/kalshi
mv ~/Downloads/kalshi_private_key_new.pem ~/.ssh/kalshi/kalshi_private_key.pem
chmod 600 ~/.ssh/kalshi/kalshi_private_key.pem

# 4. Update .env:
nano .env
# Add:
# KALSHI_API_KEY_ID="your_new_key_id"
# KALSHI_PRIVATE_KEY_PATH="/Users/jaredmarcus/.ssh/kalshi/kalshi_private_key.pem"
```

### 2. Test API Key

```bash
PYTHONPATH=. ./venv/bin/python3 -c "
from src.execution.kalshi_client import KalshiClient
client = KalshiClient()
print(client.get_balance())
"
```

Expected: `{'balance': 10000, ...}` ✅

---

## Launch Dashboard (30 minutes before kickoff)

### Option 1: Dashboard + Live Trading (Recommended)

```bash
PYTHONPATH=. ./venv/bin/python3 scripts/run_dashboard_trading.py
```

Then open: **http://localhost:8000**

### Option 2: Dashboard Only (No Trading)

```bash
PYTHONPATH=. ./venv/bin/python3 src/dashboard/dashboard_server.py
```

---

## Dashboard Overview

### What You'll See

```
┌─────────────────────────────────────────────────────────┐
│  🏈 Monday Night Football Trading Dashboard             │
│  Status: 🟢 Connected                                   │
├─────────────────────────────────────────────────────────┤
│  Game State          │  Market State                    │
│  CAR 14 - 17 SF     │  Bid: 45¢  Ask: 48¢             │
│  Q2 8:45            │  Mid: 46.5¢  Spread: 3¢          │
│  Possession: CAR    │                                   │
│  Field: 35 yd       │  Model Prediction                │
│  3rd & 7            │  Home WP: 55%  Market: 46.5%     │
│                     │  Edge: 8.5%                       │
│                     │  🟡 NO SIGNAL (edge < 10%)       │
├─────────────────────────────────────────────────────────┤
│  Performance        │  Controls                         │
│  Trades: 3          │  [x] Paper Trading               │
│  Win Rate: 66.7%    │  [ ] Auto Trading                │
│  P&L: +$45.50       │  Min Edge: 10%                   │
│                     │  [▶️ Start] [⏹️ Stop]            │
└─────────────────────────────────────────────────────────┘
```

---

## How to Trade Tonight

### Step 1: Start in Paper Mode (Safe)

1. Launch dashboard
2. **Paper Trading: ON** (default)
3. **Auto Trading: OFF** (manual approval)
4. Watch the game and observe signals

**You won't lose any money in paper mode** ✅

### Step 2: When You See a Signal

The dashboard will show:
```
🟢 BUY SIGNAL - Edge: 12.3%
```
or
```
🔴 SELL SIGNAL - Edge: 15.7%
```

**What this means:**
- Model thinks home team has X% chance to win
- Market is pricing it at Y%
- Edge = |X - Y| = opportunity

### Step 3: Execute Trade (If You Want)

**Paper Mode**: Trade is simulated, no real money

**Live Mode**: Trade is executed on Kalshi

Click **"Execute Trade"** button to place order

### Step 4: Monitor Performance

Watch the **Performance** panel:
- Total Trades
- Win Rate
- P&L (Profit & Loss)

---

## Trading Scenarios

### Scenario A: Just Watch (Recommended First Time)

```bash
# Launch dashboard
PYTHONPATH=. ./venv/bin/python3 scripts/run_dashboard_trading.py

# Settings:
Paper Trading: ON
Auto Trading: OFF

# Just observe:
- How often do signals appear?
- What's the typical edge?
- How does model perform?
```

**Goal**: Learn how the system works without risk

### Scenario B: Paper Trade with Manual Approval

```bash
# Same as Scenario A, but click "Execute" when you see good signals

# Good signal criteria:
- Edge > 12% (strong disagreement)
- Spread < 5¢ (easy to execute)
- Fresh data (< 5 seconds old)
- Clear momentum shift in game
```

**Goal**: Practice decision-making

### Scenario C: Live Trading (Use Caution!)

```bash
# Launch dashboard
# Change settings:
Paper Trading: OFF ⚠️
Auto Trading: OFF (or ON for fully automatic)
Max Position: 10 (start small!)

# Trade real money on Kalshi
```

**Goal**: Actual trading (only if comfortable!)

---

## What to Watch For

### 🟢 Good Signs

- **Large edge (>15%)**: Model strongly disagrees with market
  - Example: Model 75%, Market 55% → **20% edge!**
- **Touchdown/turnover**: Big momentum shift
- **Tight spread (<5¢)**: Easy to get filled
- **Circuit breakers all green**: APIs working

### 🔴 Warning Signs

- **Small edge (<10%)**: Not worth the risk
- **Wide spread (>10¢)**: Hard to execute
- **Circuit breaker open**: API is down, **STOP TRADING**
- **Losing streak (3+)**: Model may be off today
- **Late in Q4 (<5 min)**: Too volatile

---

## Dashboard Controls

### Configuration Panel

| Setting | Default | What It Does |
|---------|---------|--------------|
| **Paper Trading** | ON | Safe mode, no real money |
| **Auto Trading** | OFF | Automatic vs manual execution |
| **Min Edge** | 10% | Minimum edge to trigger signal |
| **Max Position** | 100 | Max contracts per trade |

### Buttons

- **▶️ Start Trading**: Begin monitoring for signals
- **⏹️ Stop Trading**: Pause trading
- **🔄 Refresh**: Reload data from APIs
- **💾 Save Config**: Save your settings

---

## Example Trading Session

**7:45 PM**: Launch dashboard
```bash
PYTHONPATH=. ./venv/bin/python3 scripts/run_dashboard_trading.py
```

**8:00 PM**: Kickoff, observe game state updating

**8:15 PM**: First touchdown!
- Score: CAR 0 - 7 SF
- Model: 65% SF win
- Market: 75% SF win
- Edge: 10% → **NO SIGNAL** (at threshold)

**8:22 PM**: CAR interception returned for TD!
- Score: CAR 7 - 7 SF
- Model: 52% SF win (nearly even)
- Market: 65% SF win (overreacting)
- Edge: 13% → **🔴 SELL SIGNAL**
- Action: Click "Execute" to sell SF win at 65¢

**8:45 PM**: Market corrects to 54¢
- You bought at 65¢, market now 54¢
- Potential to close for profit or wait

**10:00 PM**: Game ends SF wins 24-21
- Your sell at 65¢ was correct (SF only won by 3)
- Review performance in dashboard

---

## Troubleshooting

### Dashboard won't start

```bash
# Check for port conflict
lsof -i :8000

# Kill existing process
kill -9 <PID>

# Restart
PYTHONPATH=. ./venv/bin/python3 scripts/run_dashboard_trading.py
```

### No API key error

```bash
# Check .env file
cat .env | grep KALSHI

# Should see:
# KALSHI_API_KEY_ID=...
# KALSHI_PRIVATE_KEY_PATH=...

# If missing, see GENERATE_NEW_KEY.md
```

### Circuit breaker is RED

- **Kalshi API down**: Check https://kalshi.com
- **ESPN API down**: Rare, wait 60 seconds for recovery
- **Fix**: Dashboard will auto-reconnect

### No signals appearing

- **Edge too small**: Market is pricing game correctly
- **Threshold too high**: Lower min edge to 8%
- **Game not close**: Model only signals when game is competitive

---

## Safety Checklist

Before trading with real money:

- [ ] ✅ New Kalshi API key generated and tested
- [ ] ✅ Dashboard launches successfully
- [ ] ✅ Game state updating from ESPN
- [ ] ✅ Market prices updating from Kalshi
- [ ] ✅ Circuit breakers all GREEN
- [ ] ✅ Tried paper trading first
- [ ] ✅ Understand what "edge" means
- [ ] ✅ Know how to stop trading
- [ ] ✅ Comfortable with max position size

---

## Quick Reference

### Launch Commands

```bash
# Dashboard + Trading
PYTHONPATH=. ./venv/bin/python3 scripts/run_dashboard_trading.py

# Just Dashboard
PYTHONPATH=. ./venv/bin/python3 src/dashboard/dashboard_server.py

# Test ESPN API
PYTHONPATH=. ./venv/bin/python3 scripts/test_espn_api.py

# Test Kalshi API
PYTHONPATH=. ./venv/bin/python3 -c "from src.execution.kalshi_client import KalshiClient; print(KalshiClient().get_balance())"
```

### Key Files

- `DASHBOARD_GUIDE.md` - Full dashboard documentation
- `GENERATE_NEW_KEY.md` - How to get new API key
- `SECURITY_INCIDENT_REMEDIATION.md` - Security incident details
- `docs/CIRCUIT_BREAKER_AUDIT_LOG_INTEGRATION.md` - Production safeguards

---

## Tonight's Game

**Panthers @ 49ers**
- Date: Monday, November 24, 2025
- Time: 8:15 PM EST
- Network: ESPN

**Dashboard will show:**
- Live score and game state
- Kalshi market for "49ers to win"
- Model win probability
- Trading signals when edge > 10%

---

## Have Fun! 🏈

**Remember:**
- Start with paper trading
- Only trade what you can afford to lose
- Stop if circuit breaker opens
- Review performance after the game

**Questions?** Check `DASHBOARD_GUIDE.md` for detailed help.

**Good luck!** 🚀
