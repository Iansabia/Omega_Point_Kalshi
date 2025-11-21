import os
import json
import time
from typing import Dict, Any
from src.agents.base_agent import BaseTrader
from src.orderbook.order import Order, OrderType
import uuid

# Placeholder for google.genai
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

class LLMAgent(BaseTrader):
    """
    Trader agent driven by an LLM (Gemini Flash 2.0).
    """
    
    def __init__(self, unique_id: int, model, initial_wealth: float = 10000.0, risk_profile: str = "balanced"):
        super().__init__(unique_id, model, initial_wealth=initial_wealth)
        self.risk_profile = risk_profile
        self.cumulative_cost = 0.0
        self.client = None
        self.cached_content = None

        if genai and os.getenv("GEMINI_API_KEY"):
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Initialize cache if needed (mocked for now)
            # self._initialize_cache()

    def _initialize_cache(self):
        """
        Implement context caching for static agent profiles.
        """
        if not self.client:
            return
            
        agent_profile = f"You are a {self.risk_profile} trader. Your goal is to maximize profit while managing risk."
        try:
            # Mocking the cache creation as it requires specific API version/setup
            # self.cached_content = self.client.caches.create(...)
            pass
        except Exception as e:
            print(f"Cache init failed: {e}")

    def observe_market(self):
        """
        Update internal state based on market observations.
        """
        # In a real agent, this would look at order book depth, recent trades, etc.
        pass

    def make_decision(self):
        """
        Decide using LLM or fallback rules.
        """
        market_state = {
            "price": self.model.current_price,
            "spread": self.model.get_spread(),
            "volatility": 0.05, # Mock
            "wealth": self.wealth,
            "position": self.position
        }
        
        if self.should_use_llm(market_state):
            self._llm_decision(market_state)
        else:
            self._rule_based_decision(market_state)

    def should_use_llm(self, market_state: Dict) -> bool:
        """
        Determine if LLM call is worth the cost/latency.
        """
        # Use rules for simple cases
        if market_state['volatility'] < 0.1:
            return False
        # Use LLM for complex scenarios
        return True

    def _rule_based_decision(self, market_state: Dict):
        """Fallback rule-based logic: Simple Trend Following."""
        price = market_state['price']
        
        # Simple logic: if price is low (vs 0.5), buy. If high, sell.
        # This is a mean-reversion strategy for the fallback.
        if price < 0.4 and self.wealth > price:
            # Buy
            qty = 1
            order = Order(
                order_id=str(uuid.uuid4()),
                side='BUY',
                price=price, # Limit at current
                quantity=qty,
                timestamp=time.time(),
                trader_id=self.trader_id,
                order_type=OrderType.LIMIT
            )
            self.submit_orders([order])
        elif price > 0.6 and self.position > 0:
            # Sell
            qty = 1
            order = Order(
                order_id=str(uuid.uuid4()),
                side='SELL',
                price=price,
                quantity=qty,
                timestamp=time.time(),
                trader_id=self.trader_id,
                order_type=OrderType.LIMIT
            )
            self.submit_orders([order])

    def _llm_decision(self, market_state: Dict):
        """Call Gemini API with retry logic and error handling."""
        if not self.client:
            return
            
        prompt = self._construct_prompt(market_state)
        max_retries = 3
        backoff = 1.0
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                duration = time.time() - start_time
                
                # Parse response
                decision = json.loads(response.text)
                
                # Validate output format
                if not self._validate_decision_format(decision):
                    raise ValueError("Malformed LLM response format")
                    
                self._execute_llm_action(decision)
                return # Success
                
            except Exception as e:
                print(f"LLM Error (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2 # Exponential backoff
                else:
                    # Fallback after all retries fail
                    self._rule_based_decision(market_state)

    def _validate_decision_format(self, decision: Dict) -> bool:
        """Check if decision dict has required keys."""
        required = {"action", "quantity"}
        return required.issubset(decision.keys()) and decision["action"] in ["BUY", "SELL", "HOLD"]

    def _construct_prompt(self, market_state: Dict) -> str:
        return f"""
        Market State: {json.dumps(market_state)}
        Decide action (BUY/SELL/HOLD) and quantity.
        """

    def _execute_llm_action(self, decision: Dict):
        action = decision.get("action")
        quantity = decision.get("quantity", 0)
        
        if action in ["BUY", "SELL"] and quantity > 0:
            price = self.model.current_price # Market order or limit?
            # Let's assume market order for simplicity or limit at current price
            order = Order(
                order_id=str(uuid.uuid4()),
                side=action,
                price=price,
                quantity=quantity,
                timestamp=time.time(),
                trader_id=self.trader_id,
                order_type=OrderType.MARKET
            )
            self.submit_orders([order])
