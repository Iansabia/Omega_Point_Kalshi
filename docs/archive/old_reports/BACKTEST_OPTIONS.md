# Backtesting Options for Your Strategy

## Problem: Limited Historical Data

Kalshi's API **does not provide** historical tick-by-tick orderbook data or candlesticks for settled markets. You can see this data in the web UI for your own trades, but it's not available programmatically via API.

This means traditional backtesting on historical games is **not possible** without first collecting live data.

---

## ✅ Solution 1: Demo Backtest (AVAILABLE NOW)

**What**: Synthetic game with realistic momentum patterns
**How**: `python scripts/backtest_demo.py`

**Output Example**:
```
📈 Min 10: SIGNAL BUY @ $0.29 (Model: 44%, Market: 29%, Edge: +15%)
📉 Min 15: CLOSE BUY @ $0.47 (P&L: $+54.74)

Total P&L: $+307.48
Win Rate: 100%
3 trades executed
```

**Pros**:
- Shows exactly how your strategy works
- Demonstrates signal generation and P&L calculation
- Available immediately

**Cons**:
- Not real data
- Can't validate if actual edges exist

---

## ✅ Solution 2: Live Data Recording (RECOMMENDED)

**What**: Record live games then backtest on that data
**When**: Next NFL game (weekends)

### Step 1: Create Data Recorder

```python
# scripts/record_live_game.py
import asyncio
import json
from datetime import datetime
from src.data.sportradar_client import SportradarClient
from src.execution.kalshi_websocket import KalshiWebSocket

async def record_game(game_id, ticker):
    """Record live game data every second."""

    data = []

    async def on_nfl_update(gid, state):
        data.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'nfl',
            'data': state
        })

    async def on_price_update(tick, price):
        data.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'kalshi',
            'ticker': tick,
            'data': price
        })

    # Start recording
    sportradar = SportradarClient()
    kalshi_ws = KalshiWebSocket()

    await kalshi_ws.connect()
    await kalshi_ws.subscribe_orderbook(ticker)

    await asyncio.gather(
        sportradar.poll_live_games([game_id], on_nfl_update, interval=2.0),
        kalshi_ws.listen(on_price_update)
    )

    # Save after game ends
    with open(f'data/recorded/{ticker}.json', 'w') as f:
        json.dump(data, f)
```

### Step 2: Run During Live Game

```bash
# Sunday during NFL game
python scripts/record_live_game.py --game-id sr:match:... --ticker KXMVENFL...
```

### Step 3: Backtest on Recorded Data

```bash
python scripts/backtest_single_game.py --data data/recorded/KXMVENFL...json
```

**Pros**:
- Real game state + real market prices
- Can replay and analyze multiple times
- Validates actual arbitrage opportunities

**Cons**:
- Must wait for live games (weekends)
- Requires Sportradar API key
- Need to run recorder during games

---

## ✅ Solution 3: Live Paper Trading (BEST VALIDATION)

**What**: Run strategy live but don't execute real trades
**When**: Next NFL game

```python
from src.live_trading.live_trading_engine import LiveTradingEngine

engine = LiveTradingEngine(
    paper_trading=True,  # No real money
    min_edge=0.10
)

# Register today's games
engine.register_game(
    sportradar_game_id="sr:match:...",
    kalshi_ticker="KXMVENFLSINGLEGAME-...",
    home_team="BAL",
    away_team="KC"
)

await engine.start()  # Runs until game ends
```

**Pros**:
- Real-time validation
- See actual signal frequency
- Test execution speed
- Validate model predictions live

**Cons**:
- Must run during live games
- Can't replay/analyze afterwards (unless you also record)

---

## 📊 What You Can Learn From Each Method

### Demo Backtest ✅
- ✅ Strategy mechanics
- ✅ P&L calculation
- ✅ Signal filtering
- ❌ Real edge validation
- ❌ Actual market behavior

### Recorded Data Backtest ✅✅
- ✅ Real market prices
- ✅ Real game events
- ✅ Actual edge opportunities
- ✅ Can replay multiple times
- ❌ Must collect data first

### Live Paper Trading ✅✅✅
- ✅ Everything above
- ✅ Execution timing validation
- ✅ Risk management testing
- ✅ System stability
- ❌ Can't replay
- ❌ Must run during games

---

## 🎯 Recommended Approach

**Phase 1: Now → Next Game**
1. Run demo backtest to understand mechanics ✅ DONE
2. Review code and strategy
3. Get Sportradar API key
4. Set up recording script

**Phase 2: Next NFL Game (This Weekend?)**
1. Run live paper trading
2. Simultaneously record all data
3. Monitor signals in real-time
4. Take notes on what happens

**Phase 3: After Game**
1. Analyze paper trading results
2. Replay recorded data
3. Test different parameters
4. Refine strategy

**Phase 4: After 3-5 Games**
1. Evaluate signal quality
2. Check win rate on real data
3. Decide if edges actually exist
4. Consider live trading (small size)

---

## 🚨 Important Note

You mentioned you can see previous betting slips and orderbooks on Kalshi's website. Unfortunately:

1. **Web UI data ≠ API data**: The website shows this data, but API doesn't expose it for settled markets
2. **Your trades only**: You can only see markets YOU traded, not all historical markets
3. **No bulk export**: Kalshi doesn't provide a way to export all historical data

**Workaround**: If you have specific markets you traded, you might be able to manually extract that data from the web UI, but it would be tedious.

---

## 📝 Summary

**For backtesting, your best option is**:

1. ✅ **Run demo now** (shows how it works)
2. ⏳ **Record next live game** (gets real data)
3. ⏳ **Paper trade 2+ weeks** (validates strategy)
4. ⏳ **Then consider live trading**

The system is fully built and ready - you just need live game data to validate it on!

---

## Next Steps

Want me to create:
1. **Data recording script** for live games?
2. **Enhanced paper trading logger** for detailed analysis?
3. **Playback script** to replay recorded games?

Let me know which would be most helpful!
