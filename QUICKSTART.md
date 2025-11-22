# 🚀 Omega Point Quick Start Guide

## Option 1: Run Dashboard Locally (Without Docker) ⚡

**Fastest way to see the simulation in action!**

### Step 1: Activate Virtual Environment
```bash
source venv/bin/activate
```

### Step 2: Run the Dashboard
```bash
solara run test_dashboard.py
```

### Step 3: Open Browser
The terminal will show:
```
Solara server is starting at http://localhost:8765
```

Open your browser to: **http://localhost:8765**

### What You'll See:
- **Page 1:** Market price vs fundamental value, volume, spread
- **Page 2:** Agent wealth distribution, positions, top performers
- **Page 3:** Performance metrics (returns, Sharpe ratio, drawdown)
- **Page 4:** Order book depth chart and imbalance

### Interactive Controls:
- Use sliders to adjust agent counts
- Click "Step" to run 1 simulation step
- Click "Play" to run continuous simulation
- Adjust initial price and fundamental value
- Watch real-time charts update

---

## Option 2: Run Full Stack with Docker 🐳

**For production-grade environment with database, monitoring, etc.**

### Step 1: Install Docker Desktop
1. Download: https://www.docker.com/products/docker-desktop/
2. Install and start Docker Desktop
3. Verify: `docker --version`

### Step 2: Start Full Stack
```bash
cd /Users/jaredmarcus/projects/omega_point/Omega_Point_Kalshi
docker compose up
```

### Step 3: Access Services
- **Grafana:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090
- **Database:** localhost:5432
- **Redis:** localhost:6379

### Services Running:
- ✅ trader: Prediction market simulation
- ✅ database: TimescaleDB for tick data
- ✅ redis: Caching and queue
- ✅ grafana: Monitoring dashboards
- ✅ prometheus: Metrics collection

---

## Run Tests

### All Tests
```bash
source venv/bin/activate
pytest
```

### With Coverage
```bash
pytest --cov=src --cov-report=html
```

View coverage report: `open htmlcov/index.html`

### Specific Test File
```bash
pytest tests/test_market_model.py -v
```

---

## Run Backtest

```bash
source venv/bin/activate
python src/backtesting/backtest_engine.py
```

---

## Project Structure

```
Omega_Point_Kalshi/
├── src/
│   ├── agents/              # 6 agent types (noise, informed, arbitrageur, etc.)
│   ├── models/              # Jump diffusion, sentiment, microstructure
│   ├── orderbook/           # Heap-based order book + matching engine
│   ├── execution/           # Kalshi/Polymarket clients, risk manager
│   ├── backtesting/         # Event-driven backtester
│   ├── data/                # NFL data, Sportradar integration
│   └── visualization/       # Solara dashboard, monitoring, alerts
├── tests/                   # 80 tests, 72.50% coverage
├── docker/                  # Production Docker config
├── config/                  # YAML configuration files
└── notebooks/               # Jupyter analysis notebooks
```

---

## Current Implementation Status

### ✅ **COMPLETE (74%)**
- [x] Agent-Based Modeling (Mesa 3.3+)
- [x] 6 Agent Types (Noise, Informed, Arbitrageur, Market Maker, Homer, LLM)
- [x] Mathematical Models (Jump Diffusion, Sentiment, Microstructure)
- [x] Order Book Engineering (Heap-based, O(log n))
- [x] Matching Engine (MARKET, LIMIT, FOK, IOC)
- [x] Kalshi + Polymarket Integration
- [x] Backtesting Framework
- [x] Solara Dashboard (4 pages)
- [x] Monitoring & Alerting System
- [x] Circuit Breakers for APIs
- [x] Audit Logging (Write-Ahead Log)
- [x] Production Docker Config
- [x] 80 Tests (72.50% coverage)

### 🟡 **IN PROGRESS (20%)**
- [ ] Grafana Dashboards
- [ ] Performance Profiling
- [ ] Paper Trading Setup

### ⏳ **PENDING (6%)**
- [ ] 95% Test Coverage
- [ ] Stylized Facts Validation
- [ ] 30-Day Paper Trading
- [ ] Production Launch

---

## Key Features

### 🤖 **Agent Types**
1. **Noise Traders** - Random walk, contrarian, trend following
2. **Informed Traders** - Information acquisition, strategic trading
3. **Arbitrageurs** - Detect mispricing, capital constrained
4. **Market Makers** - Avellaneda-Stoikov, inventory management
5. **Homer Agents** - Loyalty bias with decay/reinforcement
6. **LLM Agents** - Gemini Flash 2.0, hybrid rule-based

### 📊 **Mathematical Models**
- **Jump Diffusion** - Merton & Kou models, MLE calibration
- **Sentiment** - FinBERT + VADER, panic coefficient, CSAD
- **Microstructure** - Kyle's Lambda, Glosten-Milgrom, spread models
- **Behavioral Biases** - Recency, herding, gambler's fallacy

### 🔧 **Production Features**
- **Circuit Breakers** - Prevent cascading failures
- **Audit Logging** - SHA256 checksums, tamper detection
- **Monitoring** - Structured logging, multi-channel alerts
- **Health Checks** - All services monitored
- **Resource Limits** - CPU/memory constraints

---

## Environment Variables

Create `.env` file:
```bash
# API Keys
GEMINI_API_KEY=your_key_here
KALSHI_EMAIL=your_email
KALSHI_PASSWORD=your_password
SPORTRADAR_API_KEY=your_key_here

# Monitoring (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
ALERT_EMAIL=your_email@example.com
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Database (Docker only)
DATABASE_URL=postgresql://user:pass@localhost:5432/trading
REDIS_URL=redis://localhost:6379
```

---

## Troubleshooting

### Dashboard won't start
```bash
# Make sure venv is activated
source venv/bin/activate

# Install missing dependencies
pip install solara altair mesa

# Try again
solara run test_dashboard.py
```

### Tests failing
```bash
# Activate venv
source venv/bin/activate

# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest -v
```

### Docker issues
```bash
# Check Docker is running
docker --version

# Use new syntax (not docker-compose)
docker compose up

# View logs
docker compose logs trader

# Restart services
docker compose restart
```

---

## Next Steps

### Immediate (Today)
1. ✅ Run dashboard: `solara run test_dashboard.py`
2. ✅ Run tests: `pytest`
3. ✅ Check coverage: `pytest --cov=src`

### Short-term (This Week)
1. [ ] Set up Kalshi demo account
2. [ ] Run comprehensive backtests
3. [ ] Start paper trading

### Medium-term (Next 2 Weeks)
1. [ ] Complete Grafana dashboards
2. [ ] 30-day paper trading validation
3. [ ] Performance optimization

### Long-term (Next Month)
1. [ ] Production launch with real money
2. [ ] Expand to golf markets
3. [ ] Scale to multiple markets

---

## Support

- **Documentation:** See `/docs` folder
- **Tests:** See `/tests` folder
- **Examples:** See `/notebooks` folder
- **Issues:** Report at GitHub

---

## Performance Benchmarks

- **Order Matching:** O(log n)
- **Simulation Speed:** 1000+ agents @ interactive speeds
- **Test Coverage:** 72.50% (80 tests)
- **API Latency:** <500ms (with circuit breakers)

---

## License

[Your license here]

---

**Built with:**
- Mesa 3.3+ (Agent-Based Modeling)
- Solara (Interactive Dashboard)
- Plotly (Charts)
- Docker (Production Deployment)
- TimescaleDB (Time-Series Data)
- Grafana (Monitoring)
