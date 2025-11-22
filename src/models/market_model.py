import mesa
from mesa import Model
from mesa.datacollection import DataCollector
from typing import Dict, Any, List, Optional
import random
import numpy as np
import yaml
import logging

# Import our components
from src.orderbook.orderbook import OrderBook
from src.orderbook.matching_engine import MatchingEngine

# Import agent types
from src.agents.noise_trader import NoiseTrader
from src.agents.informed_trader import InformedTrader
from src.agents.arbitrageur import Arbitrageur
from src.agents.market_maker_agent import MarketMakerAgent
from src.agents.homer_agent import HomerAgent
from src.agents.llm_agent import LLMAgent

logger = logging.getLogger(__name__)

class PredictionMarketModel(mesa.Model):
    """
    Agent-based model for a prediction market.
    """

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 agent_config: Optional[Dict[str, Any]] = None,
                 seed: int = None):
        """
        Initialize the prediction market model.

        Args:
            config: Base configuration dictionary
            agent_config: Agent profiles configuration dictionary
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)

        # Load configurations
        self.config = config or {}
        self.agent_config = agent_config or {}

        # Extract market parameters
        market_config = self.config.get('market', {})
        self.current_price = market_config.get('initial_price', 0.5)
        self.fundamental_value = self.current_price  # Track true value
        self.current_ticker = market_config.get('ticker', 'MARKET')  # Current market ticker

        # Mesa 3.3+ doesn't use schedulers - agents are managed directly in Model
        # Agents are automatically tracked in self.agents

        # Initialize market mechanisms
        self.order_book = OrderBook()
        self.matching_engine = MatchingEngine(self.order_book)

        # Link matching engine to this model for callbacks
        self.matching_engine.model = self

        # Market state tracking
        self.cumulative_llm_cost = 0.0
        self.step_count = 0

        # Initialize DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "market_price": "current_price",
                "fundamental_value": "fundamental_value",
                "total_volume": lambda m: m.calculate_volume(),
                "bid_ask_spread": lambda m: m.get_spread(),
                "llm_cost": "cumulative_llm_cost",
                "step": "step_count"
            },
            agent_reporters={
                "wealth": "wealth",
                "position": "position",
                "agent_type": lambda a: a.__class__.__name__
            }
        )

        # Phase 4.4: Batch Processing Queue
        self.pending_decisions = []  # List of (agent, prompt) tuples

        # Initialize agents if config provided
        if self.agent_config:
            self._initialize_agents()

    def _initialize_agents(self):
        """
        Initialize agents based on configuration.
        Mesa 3.3+ auto-assigns unique_ids to agents.
        """

        # Create Noise Traders
        if 'noise_trader' in self.agent_config:
            noise_config = self.agent_config['noise_trader']
            count = noise_config.get('count', 50)
            wealth_dist = noise_config.get('wealth_distribution', {})
            risk_limits = noise_config.get('risk_limits', None)

            # Generate wealth based on distribution
            if wealth_dist.get('type') == 'lognormal':
                mean = wealth_dist.get('mean', 1000)
                sigma = wealth_dist.get('sigma', 0.5)
                wealth_values = np.random.lognormal(
                    mean=np.log(mean), sigma=sigma, size=count
                )
            else:
                wealth_values = [noise_config.get('wealth', 1000)] * count

            # Create agents with different strategies
            strategies = ['random', 'contrarian', 'trend']
            for i in range(count):
                strategy = strategies[i % len(strategies)]
                agent = NoiseTrader(
                    model=self,
                    strategy=strategy,
                    initial_wealth=wealth_values[i],
                    risk_limits=risk_limits
                )
                # Mesa 3.3+ auto-registers agents and assigns unique_id
                logger.debug(f"Created NoiseTrader {agent.unique_id} with strategy={strategy}, wealth={wealth_values[i]:.2f}")

        # Create Informed Traders
        if 'informed_trader' in self.agent_config:
            informed_config = self.agent_config['informed_trader']
            count = informed_config.get('count', 5)
            wealth = informed_config.get('wealth', 10000)
            info_quality = informed_config.get('information_quality', 0.8)
            risk_limits = informed_config.get('risk_limits', None)

            for i in range(count):
                agent = InformedTrader(
                    model=self,
                    initial_wealth=wealth,
                    information_quality=info_quality,
                    risk_limits=risk_limits
                )
                # Mesa 3.3+ auto-registers agents
                logger.debug(f"Created InformedTrader {agent.unique_id} with quality={info_quality}")

        # Create Arbitrageurs
        if 'arbitrageur' in self.agent_config:
            arb_config = self.agent_config['arbitrageur']
            count = arb_config.get('count', 3)
            wealth = arb_config.get('wealth', 50000)
            detection_speed = arb_config.get('detection_speed', 0.9)

            for i in range(count):
                agent = Arbitrageur(
                    model=self,
                    initial_wealth=wealth,
                    detection_speed=detection_speed
                )
                # Mesa 3.3+ auto-registers agents
                logger.debug(f"Created Arbitrageur {agent.unique_id}")

        # Create Market Makers
        if 'market_maker' in self.agent_config:
            mm_config = self.agent_config['market_maker']
            count = mm_config.get('count', 1)
            wealth = mm_config.get('wealth', 100000)
            target_inventory = mm_config.get('target_inventory', 0)
            risk_limits = mm_config.get('risk_limits', None)

            for i in range(count):
                agent = MarketMakerAgent(
                    model=self,
                    initial_wealth=wealth,
                    target_inventory=target_inventory,
                    risk_limits=risk_limits
                )
                # Mesa 3.3+ auto-registers agents
                logger.debug(f"Created MarketMaker {agent.unique_id}")

        # Create Homer Agents
        if 'homer_agent' in self.agent_config:
            homer_config = self.agent_config['homer_agent']
            count = homer_config.get('count', 10)
            wealth = homer_config.get('wealth', 2000)
            loyalty_strength = homer_config.get('loyalty_strength', 0.7)

            for i in range(count):
                agent = HomerAgent(
                    model=self,
                    initial_wealth=wealth,
                    loyalty_asset="YES",  # Could be randomized or configured
                    loyalty_strength=loyalty_strength
                )
                # Mesa 3.3+ auto-registers agents
                logger.debug(f"Created HomerAgent {agent.unique_id}")

        # Create LLM Agents
        if 'llm_agent' in self.agent_config:
            llm_config = self.agent_config['llm_agent']
            count = llm_config.get('count', 2)
            wealth = llm_config.get('wealth', 10000)
            risk_profile = llm_config.get('risk_profile', 'moderate')

            for i in range(count):
                agent = LLMAgent(
                    model=self,
                    initial_wealth=wealth,
                    risk_profile=risk_profile
                )
                # Mesa 3.3+ auto-registers agents
                logger.debug(f"Created LLMAgent {agent.unique_id}")

        logger.info(f"Initialized {len(list(self.agents))} agents total")

    def process_batch_decisions(self):
        """
        Process queued LLM decisions in a batch to save costs/time.
        """
        if not self.pending_decisions:
            return

        # In a real implementation, this would use the provider's Batch API.
        # For simulation, we iterate but could parallelize.
        
        # Example structure for Gemini Batch API (conceptual)
        # batch_prompts = [
        #     {"custom_id": str(a.unique_id), "body": {"contents": p}} 
        #     for a, p in self.pending_decisions
        # ]
        
        # Mock processing
        print(f"Processing batch of {len(self.pending_decisions)} LLM decisions...")
        
        for agent, prompt in self.pending_decisions:
            # Direct call for now as fallback
            # In production: client.batches.create(...)
            try:
                # Simulate LLM response
                decision = {
                    "action": "HOLD",
                    "quantity": 0,
                    "reasoning": "Batch processed decision"
                }
                agent._execute_llm_action(decision)
            except Exception as e:
                print(f"Batch error for agent {agent.unique_id}: {e}")
                
        self.pending_decisions.clear()

    def step(self):
        """
        Advance the model by one step.
        """
        self.step_count += 1

        # Agents observe market and make decisions (Mesa 3.3+ doesn't use schedulers)
        # Shuffle agents for random activation order
        agent_list = list(self.agents)
        self.random.shuffle(agent_list)
        for agent in agent_list:
            agent.step()

        # Process any batched LLM decisions
        if self.pending_decisions:
            self.process_batch_decisions()

        # Update market price based on last trade
        if self.matching_engine.trades:
            last_trade = self.matching_engine.trades[-1]
            self.current_price = last_trade.price

        # Or fallback to mid-price if no trades
        elif self.get_spread() > 0:
            self.current_price = self.order_book.get_mid_price()

        # Collect data
        self.datacollector.collect(self)

    def calculate_volume(self) -> float:
        """Calculate total trading volume for the step/run."""
        return sum(t.quantity for t in self.matching_engine.trades)

    def get_spread(self) -> float:
        """Calculate current bid-ask spread."""
        if self.order_book.bids and self.order_book.asks:
            best_bid = -self.order_book.bids[0][0]
            best_ask = self.order_book.asks[0][0]
            return best_ask - best_bid
        return 0.0
