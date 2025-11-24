class LLMPrompts:
    """
    Prompt templates and engineering for LLM agents.
    """

    SYSTEM_TEMPLATE = """
    You are a {risk_profile} trader in a financial market simulation.
    Trading philosophy: {philosophy}
    Risk tolerance: {tolerance}
    
    MARKET RULES:
    - Position limits: {limits}
    - Transaction costs: {costs}
    
    DECISION FRAMEWORK:
    1. Review current portfolio: {state}
    2. Analyze market signals: {data}
    3. Consider risk exposure
    4. Output decision with reasoning
    
    Format response as JSON:
    {{
      "reasoning": "step-by-step analysis",
      "action": "BUY|SELL|HOLD",
      "quantity": <number>,
      "confidence": <0-1>
    }}
    """

    @staticmethod
    def get_system_prompt(risk_profile: str) -> str:
        # Fill in defaults based on profile
        philosophy = "Growth" if risk_profile == "aggressive" else "Preservation"
        tolerance = "High" if risk_profile == "aggressive" else "Low"

        return LLMPrompts.SYSTEM_TEMPLATE.format(
            risk_profile=risk_profile,
            philosophy=philosophy,
            tolerance=tolerance,
            limits="Max 10% of portfolio",
            costs="0.1% per trade",
            state="{state}",  # Left for runtime formatting
            data="{data}",  # Left for runtime formatting
        )
