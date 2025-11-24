# 📝 Paper Trading Setup Guide

## Step 1: Create Kalshi Demo Account

### Option A: Use Demo Environment (Recommended - Free)
Kalshi has a demo environment at: **https://demo.kalshi.com**

1. Go to https://demo.kalshi.com
2. Sign up for a free account
3. You'll get virtual money to trade with ($10,000 demo balance)
4. No real money required!

### Option B: Use Production with Small Amount
1. Go to https://kalshi.com
2. Sign up for real account
3. Deposit minimum amount ($10-$50 for testing)

---

## Step 2: Get API Credentials

### For Demo Environment:
1. Log in to https://demo.kalshi.com
2. Go to **Profile → API Keys** (https://demo.kalshi.com/profile/api-keys)
3. Click **Create API Key**
4. Download the private key file (.pem) - save it securely!
5. Copy your API Key ID

### For Production:
1. Log in to https://kalshi.com
2. Go to **Profile → API Keys** (https://kalshi.com/profile/api-keys)
3. Same process as demo

---

## Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Kalshi API Credentials
KALSHI_API_KEY_ID=your_api_key_id_here
KALSHI_PRIVATE_KEY_PATH=/path/to/your/private_key.pem

# Use demo environment
KALSHI_BASE_URL=https://demo-api.kalshi.co

# For production (comment out for demo):
# KALSHI_BASE_URL=https://api.elections.kalshi.com
```

---

## Step 4: Test Connection

Run the test script:

```bash
source venv/bin/activate
python test_kalshi_connection.py
```

This will:
- ✅ Test authentication
- ✅ Fetch your account balance
- ✅ List available markets
- ✅ Get market data for a sample market

---

## Step 5: Run Paper Trading

Once connection works:

```bash
source venv/bin/activate
python run_paper_trading.py
```

This will:
- Connect to your Kalshi demo account
- Run the ABM simulation
- Generate trading signals from agents
- Place orders on demo markets
- Track performance

---

## Current Implementation Status

### ✅ **Already Built:**
- Kalshi client (`src/execution/kalshi_client.py`)
- Signal generator (`src/execution/signal_generator.py`)
- Risk manager (`src/execution/risk_manager.py`)
- Order router (`src/execution/order_router.py`)
- Audit logging (`src/execution/audit_log.py`)
- Circuit breakers (`src/execution/circuit_breaker.py`)

### 🔧 **Need to Build:**
- Paper trading orchestrator (connects everything)
- Market selection logic (which markets to trade)
- Position tracking
- Performance monitoring
- Live dashboard (optional)

---

## Safety Features

✅ **Demo Account = Zero Risk**
- No real money
- Can't lose anything
- Perfect for testing

✅ **Risk Controls Built-In:**
- Position limits
- Drawdown stops
- Kill switch
- Circuit breakers for API failures
- Audit logging of all trades

✅ **Monitoring:**
- Real-time alerts (Slack/Email)
- Performance tracking
- Error logging

---

## Recommended Approach

**Week 1: Demo Trading**
- Set up demo account
- Test with virtual money
- Validate strategies
- Monitor for errors

**Week 2-4: Monitor Performance**
- Track daily P&L
- Calculate Sharpe ratio
- Measure win rate
- Identify issues

**After 30 Days:**
- Review results
- If Sharpe > 0.8 and drawdown < 10%
- Consider small real money ($50-$100)

---

## Next Steps

1. **Right now:** Create demo account at https://demo.kalshi.com
2. **Get API credentials:** Generate API key and download private key file
3. **Update .env:** Add your API key ID and private key path
4. **Test connection:** Run `python scripts/test_kalshi_connection.py`
5. **Start paper trading:** Run `python scripts/run_backtest.py`

---

## Troubleshooting

**Can't log in to demo?**
- Demo might require invitation/waitlist
- Try production with $10 deposit instead
- Email Kalshi support for demo access

**API errors?**
- Check credentials in .env
- Verify API key is valid
- Check you're using correct base URL (demo vs prod)

**No markets available?**
- Demo might have limited markets
- Try production for full market access
- Focus on NFL/sports markets

---

## Support

- **Kalshi Docs:** https://trading-api.kalshi.com/docs
- **Kalshi Support:** support@kalshi.com
- **Demo Issues:** Mention you're testing API integration

---

**Ready to start?** Let me know when you have your Kalshi credentials!
