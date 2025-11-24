# Monday Night Football Paper Trading - TONIGHT!

## Game: Carolina Panthers @ San Francisco 49ers
**Kickoff**: 8:15 PM ET (November 24, 2025)

---

## Quick Start (3 Steps)

### Step 1: Get Sportradar API Key (5 minutes)

1. Go to: https://developer.sportradar.com/
2. Sign up for free account
3. Get **NFL v7 Trial API key**
4. Add to `.env`:
   ```bash
   SPORTRADAR_API_KEY=your_trial_key_here
   ```

### Step 2: Test Setup (2 minutes)

```bash
# Test if you can find tonight's game
python scripts/find_tonights_game.py
```

**Expected output**:
```
✅ FOUND GAME!
Sportradar Game ID: sr:match:xxxxx
✅ FOUND MARKET!
Kalshi Ticker: KXMVENFLSINGLEGAME-S2025-xxxxx
```

### Step 3: Start Paper Trading (Before 8:15 PM)

```bash
# Run this 15 minutes before kickoff
python scripts/run_paper_trading_mnf.py
```

**That's it!** The system will:
- ✅ Auto-find game and market
- ✅ Connect to live NFL data (Sportradar)
- ✅ Connect to live market prices (Kalshi WebSocket)
- ✅ Trade automatically in paper mode (NO REAL MONEY)
- ✅ Log everything to file

---

## What You'll See

### During Setup:
```
================================================
MONDAY NIGHT FOOTBALL - PAPER TRADING
================================================
Game: CAR @ SF
Mode: PAPER TRADING (No Real Money)

🔧 Initializing trading engine...
✅ Win probability model loaded
✅ Arbitrage detector ready
✅ Risk manager initialized

📍 Registering game...
✅ Game registered: CAR @ SF

🚀 Starting paper trading...
✅ Sportradar connected (polling every 2s)
✅ Kalshi WebSocket connected
```

### During Game:
```
[19:32:15] INFO - NFL Update: SF 7, CAR 3, Q1 8:45
[19:32:15] INFO - Kalshi Price: $0.72 (SF to win)
[19:32:16] INFO - Model WP: 0.68 (68% SF wins)
[19:32:16] INFO - Edge: -4% (below threshold)

[19:45:22] INFO - NFL Update: SF 14, CAR 10, Q2 3:12
[19:45:22] INFO - Kalshi Price: $0.88 (SF to win)
[19:45:23] INFO - Model WP: 0.75 (75% SF wins)
[19:45:23] INFO - 🎯 SIGNAL: SELL @ $0.88 (Edge: -13%)
[19:45:23] INFO - 📝 PAPER TRADE: SELL 114 contracts @ $0.88

[19:50:15] INFO - Kalshi Price: $0.78
[19:50:15] INFO - 📉 CLOSE SELL @ $0.78 (P&L: $+11.40)
```

---

## Controls

- **Stop trading**: Press `Ctrl+C` (stops immediately)
- **View logs**: Check `logs/paper_trading_mnf_YYYYMMDD_HHMMSS.log`
- **Monitor**: Watch console output in real-time

---

## What to Expect

### Typical Game:
- **20-40 signals** generated (opportunities detected)
- **5-15 actual trades** (after risk filtering)
- **Most active during**: Score changes, turnovers, red zone plays

### Best Signals:
- After touchdowns (momentum spike)
- After turnovers (market overreacts)
- During 4th quarter (time pressure)
- On big plays (50+ yard gains)

---

## Safety Features

✅ **Paper Trading Only** - No real money at risk
✅ **Risk Limits** - Max position sizes enforced
✅ **Data Freshness** - Won't trade on stale data (>10s old)
✅ **Spread Limits** - Won't trade if spread too wide (>10%)
✅ **Edge Requirements** - Only trades with min 10% edge
✅ **Kill Switch** - Ctrl+C stops immediately

---

## Troubleshooting

### Problem: "Could not find game"
**Solution**:
- Check `SPORTRADAR_API_KEY` in `.env`
- Verify it's a valid trial key
- Test: `curl "https://api.sportradar.us/nfl/official/trial/v7/en/games/2025/REG/12/schedule.json?api_key=YOUR_KEY"`

### Problem: "Could not find market"
**Solution**:
- Kalshi market may not be open yet
- Try running closer to game time (after 7:00 PM ET)
- Check kalshi.com to see if markets are available

### Problem: "No signals generated"
**Reasons**:
- Market is efficient (no arbitrage opportunities)
- Min edge too high (can lower to 0.05 in code)
- Spread too wide (market has low liquidity)
- Wait for game events (TDs, turnovers)

### Problem: Rate limit errors
**Solution**:
- Sportradar trial: 1 req/sec limit
- Script already handles this (polls every 2s)
- Should work fine

---

## After the Game

### View Results:
```bash
# Check log file
cat logs/paper_trading_mnf_*.log

# Or search for specific events
grep "SIGNAL" logs/paper_trading_mnf_*.log
grep "PAPER TRADE" logs/paper_trading_mnf_*.log
```

### Analysis Questions:
1. How many signals were generated?
2. How many passed risk filters?
3. What was the win rate?
4. When did most signals occur?
5. Were edges real or model error?

---

## Manual Mode (If Auto-Find Fails)

If the auto-discovery doesn't work, you can manually specify:

```bash
# Step 1: Find IDs manually
python scripts/find_tonights_game.py

# Step 2: Copy the IDs and run
python scripts/run_paper_trading_mnf.py \
    --game-id "sr:match:xxxxx" \
    --ticker "KXMVENFLSINGLEGAME-xxxxx" \
    --home "SF" \
    --away "CAR"
```

---

## System Architecture (What's Running)

```
┌─────────────────────────────────────────┐
│      Paper Trading Engine (You)        │
├─────────────────────────────────────────┤
│                                         │
│  Sportradar ──────┐                    │
│  (NFL Data)       │                    │
│  Every 2s         ├──> Event Correlator│
│                   │    (Sync Streams)  │
│  Kalshi WS ───────┘                    │
│  (Prices)              │               │
│  Real-time             ▼               │
│                   Win Prob Model       │
│                   (<5ms inference)     │
│                        │               │
│                        ▼               │
│                   Arbitrage Detector   │
│                   (Find Edges >10%)    │
│                        │               │
│                        ▼               │
│                   Risk Manager         │
│                   (Safety Checks)      │
│                        │               │
│                        ▼               │
│                   Paper Trade Logger   │
│                   (Log Everything)     │
│                                         │
└─────────────────────────────────────────┘
```

---

## Timeline for Tonight

### 7:00 PM - 8:00 PM ET:
- ✅ Get Sportradar API key
- ✅ Test setup with `find_tonights_game.py`
- ✅ Verify game found

### 8:00 PM ET (15 min before kickoff):
- ✅ Run `python scripts/run_paper_trading_mnf.py`
- ✅ Verify connections
- ✅ Wait for kickoff

### 8:15 PM - 11:30 PM ET (During game):
- ✅ System trades automatically
- ✅ Monitor console output
- ✅ Watch for signals

### After 11:30 PM (Post-game):
- ✅ Press Ctrl+C to stop
- ✅ Review final statistics
- ✅ Analyze logs
- ✅ Plan improvements

---

## Success Criteria

After tonight, you should know:
- ✅ Does your system work end-to-end?
- ✅ Do arbitrage opportunities exist?
- ✅ Is your model accurate?
- ✅ How fast is execution?
- ✅ What's the signal quality?

---

## Next Steps After Tonight

Based on tonight's results:

1. **If good signals** → Run on more games, collect data
2. **If no signals** → Lower edge threshold, tune parameters
3. **If model wrong** → Review predictions vs actual outcomes
4. **If too slow** → Optimize execution speed

---

## Need Help?

**Before game**:
- Test: `python scripts/find_tonights_game.py`
- Check: `.env` has `SPORTRADAR_API_KEY`

**During game**:
- Watch console for errors
- Check `logs/` directory for detailed logs

**After game**:
- Review logs in `logs/` directory
- Analyze what happened

---

**Good luck! 🏈**

The system is fully built and ready - tonight is your first real validation!
