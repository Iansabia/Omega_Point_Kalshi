# Omega Point ABM - API Reference

**Version:** 1.0
**Last Updated:** November 22, 2025

## Table of Contents
- [Jump-Diffusion Model](#jump-diffusion-model)
- [Sentiment Model](#sentiment-model)
- [Market Microstructure](#market-microstructure)
- [Behavioral Biases](#behavioral-biases)
- [Prediction Market Model](#prediction-market-model)
- [Agent Types](#agent-types)

---

## Jump-Diffusion Model

**Module:** `src.models.jump_diffusion`
**Class:** `JumpDiffusionModel`

### Initialization

```python
from src.models.jump_diffusion import JumpDiffusionModel

# Default parameters
model = JumpDiffusionModel()

# Custom parameters
model = JumpDiffusionModel(params={
    'sigma': 0.35,          # Diffusion volatility
    'lambda_base': 5,       # Jump rate per contract lifetime
    'eta_up': 20,           # Upward jump rate parameter
    'eta_down': 12,         # Downward jump rate parameter
    'p_up': 0.4,            # Probability of upward jump
    'mu_jump': 0.0,         # Mean jump size
    'sigma_jump': 0.15      # Jump size volatility
})
```

### Methods

#### `simulate_path(S0, T, steps, simulations=1)`
Simulate price paths using Merton Jump Diffusion.

**Parameters:**
- `S0` (float): Initial price
- `T` (float): Total time horizon
- `steps` (int): Number of time steps
- `simulations` (int): Number of simulation paths (default: 1)

**Returns:**
- `np.ndarray`: Shape `(steps + 1, simulations)` with price paths

**Example:**
```python
# Simulate 1000 paths over 100 days
paths = model.simulate_path(
    S0=0.5,
    T=100/252,  # 100 trading days
    steps=100,
    simulations=1000
)

# paths[0, :] = initial prices (all 0.5)
# paths[-1, :] = final prices
```

#### `merton_jump_diffusion(S_t, dt)`
Single-step Merton model update.

**Formula:** `dP_t = μ(S_t)dt + σ(S_t)dW_t + J(Z_t)dN_t`

**Parameters:**
- `S_t` (float): Current price
- `dt` (float): Time step

**Returns:**
- `float`: Next price after one step

#### `kou_double_exponential(S_t, dt)`
Single-step Kou model with asymmetric jumps.

**Parameters:**
- `S_t` (float): Current price
- `dt` (float): Time step

**Returns:**
- `float`: Next price after one step

#### `calibrate_mle(historical_prices)`
Maximum Likelihood Estimation calibration.

**Parameters:**
- `historical_prices` (np.array): Array of historical prices

**Returns:**
- `bool`: True if calibration successful

**Side Effects:**
- Updates `self.params` with calibrated values

**Example:**
```python
prices = np.array([0.45, 0.46, 0.47, 0.48, 0.50, 0.51])
success = model.calibrate_mle(prices)
if success:
    print(f"Calibrated sigma: {model.params['sigma']}")
```

#### `calibrate_method_of_moments(historical_prices)`
Method of Moments calibration.

**Parameters:**
- `historical_prices` (np.array): Array of historical prices

**Returns:**
- `dict`: Empirical moments `{'mean', 'var', 'skew', 'kurt'}`

#### `calibrate_mcmc(historical_prices, iterations=1000)`
Bayesian MCMC calibration using Metropolis-Hastings.

**Parameters:**
- `historical_prices` (np.array): Array of historical prices
- `iterations` (int): Number of MCMC iterations

**Returns:**
- `dict`: Calibrated parameters `{'sigma', 'lambda'}`

#### `liquidity_adjusted_intensity(liquidity_t)`
Adjust jump intensity based on liquidity.

**Formula:** `λ(t) = λ_base × f(liquidity_t)`

**Parameters:**
- `liquidity_t` (float): Current market liquidity

**Returns:**
- `float`: Adjusted jump intensity (capped at 5x)

---

## Sentiment Model

**Module:** `src.models.sentiment_model`
**Class:** `SentimentModel`

### Initialization

```python
from src.models.sentiment_model import SentimentModel

model = SentimentModel()
```

### Constants

```python
ALPHA = 3.5    # Panic coefficient parameter
BETA = 1.2     # Panic coefficient parameter
GAMMA = -2.0   # Panic coefficient parameter
```

### Methods

#### `analyze_sentiment_finbert(text)`
Analyze sentiment using FinBERT model.

**Parameters:**
- `text` (str): Text to analyze

**Returns:**
- `float`: Sentiment score in [-1, 1] range
  - Positive values: Bullish sentiment
  - Negative values: Bearish sentiment
  - Near 0: Neutral

**Example:**
```python
text = "The team looks incredible this season!"
score = model.analyze_sentiment_finbert(text)
print(f"Sentiment: {score:.2f}")  # Expected: 0.80-0.95
```

**Note:** Falls back to VADER if transformers library unavailable.

#### `analyze_sentiment_vader(text)`
Analyze sentiment using VADER lexicon.

**Parameters:**
- `text` (str): Text to analyze

**Returns:**
- `float`: Compound sentiment score in [-1, 1]

**Example:**
```python
text = "Terrible performance, complete disaster"
score = model.analyze_sentiment_vader(text)
print(f"Sentiment: {score:.2f}")  # Expected: -0.80 to -0.95
```

**Convenience Method:**
```python
# Use either FinBERT or VADER automatically
score = model.analyze_sentiment_finbert(text)  # Auto-fallback to VADER
```

#### `aggregate_sentiment(n_pos, n_neg)`
Aggregate sentiment from positive/negative counts.

**Formula:** `S = (N_pos - N_neg) / (N_pos + N_neg)`

**Parameters:**
- `n_pos` (int): Number of positive signals
- `n_neg` (int): Number of negative signals

**Returns:**
- `float`: Aggregated sentiment in [-1, 1]

**Example:**
```python
# 70 positive tweets, 30 negative
agg_sent = model.aggregate_sentiment(70, 30)
print(agg_sent)  # Output: 0.4
```

#### `calculate_panic_coefficient(csad_t, volatility_t, sentiment_t)`
Calculate market panic coefficient.

**Formula:** `Panic_t = exp(α×CSAD_t + β×Volatility_t + γ×Sentiment_t)`

**Parameters:**
- `csad_t` (float): Cross-sectional absolute deviation
- `volatility_t` (float): Market volatility
- `sentiment_t` (float): Market sentiment

**Returns:**
- `float`: Panic coefficient (>1 indicates elevated panic)

**Example:**
```python
# High stress scenario
panic = model.calculate_panic_coefficient(
    csad_t=0.05,        # High dispersion
    volatility_t=0.4,   # High volatility
    sentiment_t=-0.8    # Negative sentiment
)
print(f"Panic level: {panic:.2f}")  # Expected: >2.0
```

#### `calculate_csad(returns, market_return)`
Calculate Cross-Sectional Absolute Deviation.

**Formula:** `CSAD_t = (1/N) × Σ|R_i,t - R_m,t|`

**Parameters:**
- `returns` (np.ndarray): Individual agent/asset returns
- `market_return` (float): Market return

**Returns:**
- `float`: CSAD value

**Example:**
```python
agent_returns = np.array([0.02, 0.03, -0.01, 0.04, 0.02])
market_ret = 0.02

csad = model.calculate_csad(agent_returns, market_ret)
print(f"CSAD: {csad:.4f}")
```

#### `update_herding_history(returns, market_return)`
Update rolling history for herding detection.

**Parameters:**
- `returns` (np.ndarray): Current returns array
- `market_return` (float): Current market return

**Side Effects:**
- Appends to `self.csad_history` and `self.market_return_history`
- Maintains last 100 periods

#### `detect_herding(min_periods=30)`
Detect herding behavior using regression analysis.

**Regression:** `CSAD_t = α + γ₁|R_m,t| + γ₂(R_m,t)² + ε_t`

**Herding Indicator:** γ₂ < 0 and statistically significant (t < -1.96)

**Parameters:**
- `min_periods` (int): Minimum periods required (default: 30)

**Returns:**
- `dict`: Regression results
  ```python
  {
      'alpha': float,           # Intercept
      'gamma1': float,          # Linear coefficient
      'gamma2': float,          # Quadratic coefficient
      't_statistic': float,     # T-stat for gamma2
      'is_herding': bool,       # True if herding detected
      'r_squared': float,       # Model fit
      'n_periods': int          # Number of periods used
  }
  ```

**Example:**
```python
# After collecting data over time
for t in range(100):
    model.update_herding_history(returns_t, market_ret_t)

# Detect herding
results = model.detect_herding()
if results['is_herding']:
    print(f"Herding detected! γ₂ = {results['gamma2']:.4f}")
```

---

## Market Microstructure

**Module:** `src.models.microstructure`
**Class:** `MicrostructureModel`

### Initialization

```python
from src.models.microstructure import MicrostructureModel

ms = MicrostructureModel()
```

### Constants

```python
KYLE_LAMBDA = 1.5  # Default for prediction markets
ETA = 0.314        # Almgren-Chriss parameter
GAMMA = 0.142      # Almgren-Chriss exponent
```

### Methods

#### `calculate_kyle_lambda(sigma_v, sigma_u)`
Calculate Kyle's lambda for price impact.

**Formula:** `λ = 0.5 × √(Σ_v/Σ_u)`

**Parameters:**
- `sigma_v` (float): Volatility of fundamental value
- `sigma_u` (float): Volatility of noise trading

**Returns:**
- `float`: Kyle's lambda

**Interpretation:**
- Higher λ → Higher price impact → Less liquid market
- Lower λ → Lower price impact → More liquid market

**Example:**
```python
# Illiquid market
lambda_illiq = ms.calculate_kyle_lambda(sigma_v=100, sigma_u=10)
print(f"Illiquid λ: {lambda_illiq:.2f}")  # ~1.58

# Liquid market
lambda_liq = ms.calculate_kyle_lambda(sigma_v=10, sigma_u=100)
print(f"Liquid λ: {lambda_liq:.2f}")  # ~0.16
```

#### `calculate_spread(order_processing_cost, inventory_cost, adverse_selection_cost)`
Calculate bid-ask spread components.

**Formula:** `Spread = Order_Processing + Inventory + Adverse_Selection`

**Parameters:**
- `order_processing_cost` (float): Fixed transaction cost
- `inventory_cost` (float): Inventory holding cost
- `adverse_selection_cost` (float): Information asymmetry cost

**Returns:**
- `float`: Total spread

**Example:**
```python
spread = ms.calculate_spread(
    order_processing_cost=0.0005,  # 5 bps
    inventory_cost=0.001,           # 10 bps
    adverse_selection_cost=0.002   # 20 bps
)
print(f"Total spread: {spread:.4f}")  # 0.0035 (35 bps)
```

#### `calculate_price_impact_sqrt(quantity, lambda_param=None)`
Calculate price impact using square root law.

**Formula:** `ΔP = λ × √Q`

**Parameters:**
- `quantity` (float): Order quantity
- `lambda_param` (float, optional): Kyle's lambda (uses default if None)

**Returns:**
- `float`: Price impact

**Example:**
```python
# Square root law: 4x quantity = 2x impact
impact_100 = ms.calculate_price_impact_sqrt(100, lambda_param=1.5)
impact_400 = ms.calculate_price_impact_sqrt(400, lambda_param=1.5)

print(f"Q=100: {impact_100:.2f}")  # 15.0
print(f"Q=400: {impact_400:.2f}")  # 30.0
print(f"Ratio: {impact_400/impact_100:.2f}")  # 2.0
```

#### `calculate_almgren_chriss_impact(quantity, daily_volume, volatility)`
Calculate market impact using Almgren-Chriss model.

**Formula:** `Impact = η × σ × (Q/V)^γ`

**Parameters:**
- `quantity` (float): Order quantity
- `daily_volume` (float): Average daily volume
- `volatility` (float): Price volatility

**Returns:**
- `float`: Market impact as decimal

**Example:**
```python
impact = ms.calculate_almgren_chriss_impact(
    quantity=1000,
    daily_volume=50000,
    volatility=0.3
)
print(f"Market impact: {impact:.4f}")
```

---

## Behavioral Biases

**Module:** `src.models.behavioral_biases`
**Class:** `BehavioralBiases`

### Initialization

```python
from src.models.behavioral_biases import BehavioralBiases

biases = BehavioralBiases()
```

### Attributes

```python
recency_weight = 0.7       # Overweights recent data
herding_coefficient = 0.2  # Herding strength
```

### Methods

#### `apply_recency_bias(historical_returns)`
Apply recency bias to returns.

**Parameters:**
- `historical_returns` (list): List of historical returns

**Returns:**
- `float`: Weighted average with recency bias

**Formula:** `weighted_return = 0.7 × recent + 0.3 × past_avg`

**Example:**
```python
returns = [0.01, 0.02, 0.03, 0.04, 0.10]  # Last one is recent
biased_return = biases.apply_recency_bias(returns)
print(f"Biased return: {biased_return:.4f}")  # Heavily weights 0.10
```

#### `calculate_loyalty_adjustment(fundamental_value, loyalty_strength, is_preferred_outcome)`
Adjust value perception based on loyalty (homer bias).

**Parameters:**
- `fundamental_value` (float): True fundamental value
- `loyalty_strength` (float): Loyalty strength in [0.5, 0.9]
- `is_preferred_outcome` (bool): True if outcome favors loyal team

**Returns:**
- `float`: Adjusted perceived value

**Example:**
```python
# Fan overvalues their team winning
perceived = biases.calculate_loyalty_adjustment(
    fundamental_value=0.6,
    loyalty_strength=0.9,
    is_preferred_outcome=True
)
print(f"Fan perception: {perceived:.2f}")  # > 0.6 (overvalued)
```

#### `detect_gamblers_fallacy(recent_outcomes, target_outcome)`
Detect gambler's fallacy thinking.

**Parameters:**
- `recent_outcomes` (list): Recent outcomes (e.g., ['WIN', 'WIN', 'WIN'])
- `target_outcome` (str): Outcome to check

**Returns:**
- `bool`: True if gambler's fallacy applies

**Logic:** After streak of same outcome, agent irrationally expects reversal

---

## Prediction Market Model

**Module:** `src.models.market_model`
**Class:** `PredictionMarketModel`

### Initialization

The model uses **configuration-based initialization**:

```python
from src.models.market_model import PredictionMarketModel

# Load configurations
import yaml

with open('config/base_config.yaml') as f:
    config = yaml.safe_load(f)

with open('config/agent_profiles.yaml') as f:
    agent_config = yaml.safe_load(f)

# Initialize model
model = PredictionMarketModel(
    config=config,
    agent_config=agent_config,
    seed=42
)
```

### Configuration Structure

**`config/base_config.yaml`:**
```yaml
market:
  initial_price: 0.5
  ticker: "NFL_CHI_GB_2025W10"
  fundamental_value: 0.6  # True probability

simulation:
  max_steps: 1000
  collection_interval: 1
```

**`config/agent_profiles.yaml`:**
```yaml
noise_trader:
  count: 50
  initial_wealth: 1000
  trade_probability: 0.1

informed_trader:
  count: 20
  initial_wealth: 5000
  information_quality_range: [0.6, 0.9]

arbitrageur:
  count: 10
  initial_wealth: 10000
  detection_speed_range: [0.7, 1.0]

market_maker:
  count: 5
  initial_wealth: 20000
  spread_target: 0.02

homer_agent:
  count: 30
  initial_wealth: 1000
  loyalty_strength_range: [0.5, 0.9]

llm_agent:
  count: 5
  initial_wealth: 5000
  model: "gemini-2.0-flash"
```

### Attributes

```python
model.current_price         # Current market price
model.fundamental_value     # True value/probability
model.current_ticker        # Market ticker symbol
model.order_book            # OrderBook instance
model.matching_engine       # MatchingEngine instance
model.agents                # AgentSet (Mesa 3.0+)
model.datacollector         # DataCollector instance
model.cumulative_llm_cost   # Total LLM API cost
model.step_count            # Current step number
```

### Methods

#### `step()`
Execute one simulation step.

**Process:**
1. All agents observe market
2. Agents make decisions
3. Orders submitted to order book
4. Matching engine executes trades
5. Data collected
6. Step counter incremented

**Example:**
```python
# Run simulation for 100 steps
for i in range(100):
    model.step()

    if i % 10 == 0:
        print(f"Step {i}: Price = {model.current_price:.4f}")
```

#### `calculate_volume()`
Calculate total order book volume.

**Returns:**
- `int`: Total number of orders in order book

#### `get_spread()`
Get current bid-ask spread.

**Returns:**
- `float`: Spread between best ask and best bid

**Example:**
```python
spread = model.get_spread()
print(f"Current spread: {spread:.4f}")
```

### Data Collection

Access collected data:

```python
# Get model-level data
model_data = model.datacollector.get_model_vars_dataframe()
print(model_data['market_price'])  # Price time series
print(model_data['total_volume'])  # Volume time series

# Get agent-level data
agent_data = model.datacollector.get_agent_vars_dataframe()
print(agent_data.groupby('agent_type')['wealth'].mean())
```

---

## Agent Types

All agents inherit from `BaseTrader` abstract class.

### Common Agent Attributes

```python
agent.unique_id        # Unique agent ID (auto-assigned)
agent.model            # Reference to market model
agent.wealth           # Current wealth
agent.position         # Current position (contracts held)
agent.trade_history    # List of past trades
```

### Common Agent Methods

```python
agent.observe_market()     # Read current market state
agent.make_decision()      # Generate trading signal
agent.submit_orders()      # Place orders
agent.execute_trade()      # Update portfolio
agent.get_portfolio_value()  # Calculate total value
agent.calculate_pnl()      # Calculate P&L
agent.check_risk_limits()  # Verify risk constraints
```

### 1. Noise Traders

**Types:**
- `RandomNoiseTrader`: Random trades (10% probability per step)
- `ContrarianTrader`: Trade against recent returns
- `TrendFollower`: Moving average crossover strategy

**Behavioral Biases:**
- Recency bias (0.7 weight on recent)
- Overconfidence (trade size multiplier 1.2-1.5)

### 2. Informed Traders

**Attributes:**
- `information_quality` ∈ [0.5, 1.0]: Signal accuracy

**Behavior:**
- Acquire information about fundamental value
- Trade when signal diverges from price (>2% threshold)
- Strategic order splitting to minimize impact

### 3. Arbitrageurs

**Attributes:**
- `detection_speed` ∈ [0.7, 1.0]: Speed of detecting mispricing

**Behavior:**
- Detect price divergence from fundamental
- Execute when spread > minimum threshold
- Close positions as prices converge

### 4. Market Makers

**Based on:** Avellaneda-Stoikov framework

**Attributes:**
- `target_inventory`: Target position (usually 0)
- `risk_param`: Risk aversion parameter

**Behavior:**
- Provide continuous bid/ask quotes
- Adjust spreads based on inventory
- Manage inventory toward target

### 5. Homer Agents

**Attributes:**
- `loyal_asset`: Team/outcome they favor
- `loyalty_strength` ∈ [0.5, 0.9]: Strength of bias

**Behavior:**
- Overvalue preferred outcome
- Loyalty decays 1% per step
- Reinforced 5% on positive outcomes

### 6. LLM Agents

**Attributes:**
- `llm_model`: Model name (e.g., "gemini-2.0-flash")
- `cost_per_call`: API cost tracking

**Behavior:**
- Hybrid: 70% rule-based, 30% LLM decisions
- Uses LLM for complex market conditions
- Context caching for cost efficiency
- Batch processing when possible

---

## Usage Examples

### Complete Simulation Example

```python
import yaml
from src.models.market_model import PredictionMarketModel

# Load configurations
with open('config/base_config.yaml') as f:
    config = yaml.safe_load(f)

with open('config/agent_profiles.yaml') as f:
    agent_config = yaml.safe_load(f)

# Create model
model = PredictionMarketModel(
    config=config,
    agent_config=agent_config,
    seed=42
)

# Run simulation
print(f"Starting simulation: {model.current_ticker}")
print(f"Initial price: {model.current_price:.4f}")

for step in range(1000):
    model.step()

    if step % 100 == 0:
        price = model.current_price
        spread = model.get_spread()
        volume = model.calculate_volume()

        print(f"Step {step}: Price={price:.4f}, "
              f"Spread={spread:.4f}, Volume={volume}")

# Analyze results
model_data = model.datacollector.get_model_vars_dataframe()

print(f"\nFinal Results:")
print(f"Final price: {model_data['market_price'].iloc[-1]:.4f}")
print(f"Fundamental: {model.fundamental_value:.4f}")
print(f"Price efficiency: "
      f"{1 - abs(model_data['market_price'].iloc[-1] - model.fundamental_value):.2%}")
print(f"Total LLM cost: ${model.cumulative_llm_cost:.2f}")
```

### Testing with Custom Agents

```python
# Create minimal model for testing
model = PredictionMarketModel(
    config={'market': {'initial_price': 0.5}},
    agent_config={
        'informed_trader': {'count': 10, 'initial_wealth': 5000},
        'noise_trader': {'count': 20, 'initial_wealth': 1000}
    },
    seed=42
)

# Run short simulation
for _ in range(50):
    model.step()

# Check agent performance
agent_data = model.datacollector.get_agent_vars_dataframe()
informed_wealth = agent_data[agent_data['agent_type'] == 'InformedTrader']['wealth'].mean()
noise_wealth = agent_data[agent_data['agent_type'] == 'RandomNoiseTrader']['wealth'].mean()

print(f"Informed traders avg wealth: ${informed_wealth:.2f}")
print(f"Noise traders avg wealth: ${noise_wealth:.2f}")
```

---

## Error Handling

### Common Errors

**1. Configuration Missing:**
```python
# Error
model = PredictionMarketModel()  # No config provided

# Solution
model = PredictionMarketModel(
    config={'market': {'initial_price': 0.5}},
    agent_config={}
)
```

**2. Invalid Parameters:**
```python
# Error
model = JumpDiffusionModel(params={'sigma': -0.1})  # Negative volatility

# Will default to base parameters or raise error
```

**3. Insufficient Data:**
```python
sentiment = SentimentModel()
result = sentiment.detect_herding()  # Not enough data

# Check result
if not result['is_herding'] and 'message' in result:
    print(result['message'])  # "Insufficient data: 0/30"
```

---

## Performance Tips

1. **Use Numba for hot paths** (already implemented in matching engine)
2. **Batch LLM calls** (automatic in hybrid agents)
3. **Limit data collection frequency** for large simulations
4. **Use vectorized numpy operations** in custom calculations

---

## Next Steps

- See `VALIDATION_REPORT.md` for test results
- See `config/examples/` for configuration templates
- See `tests/test_validation_suite.py` for usage examples
- See `Checklist.md` for implementation roadmap

---

**Documentation Version:** 1.0
**Generated:** November 22, 2025
**Maintained By:** Omega Point Development Team
