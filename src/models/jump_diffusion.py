import numpy as np
from typing import Dict, Any, Optional

class JumpDiffusionModel:
    """
    Implements Jump-Diffusion models for asset pricing in prediction markets.
    Includes Merton model and Kou double exponential model with asymmetric jumps.
    """
    
    # Prediction market defaults from research
    PARAMS = {
        'sigma': 0.35,  # Diffusion volatility
        'lambda_base': 5,  # Jump rate per contract lifetime
        'eta_up': 20,  # Upward jump rate parameter
        'eta_down': 12,  # Downward jump rate parameter
        'p_up': 0.4,  # Probability of upward jump
        'mu_jump': 0.0,  # Mean jump size
        'sigma_jump': 0.15  # Jump size volatility
    }

    def __init__(self, params: Optional[Dict[str, float]] = None):
        self.params = self.PARAMS.copy()
        if params:
            self.params.update(params)

    def merton_jump_diffusion(self, S_t: float, dt: float) -> float:
        """
        Implement Merton model: dP_t = μ(S_t)dt + σ(S_t)dW_t + J(Z_t)dN_t
        """
        mu = 0.0 # Drift assumed 0 for short term
        sigma = self.params['sigma']
        lambda_base = self.params['lambda_base']
        mu_jump = self.params['mu_jump']
        sigma_jump = self.params['sigma_jump']
        
        # Diffusion component
        dW = np.random.normal(0, np.sqrt(dt))
        diffusion = sigma * S_t * dW
        
        # Jump component
        # Poisson process for jump occurrence
        if np.random.random() < lambda_base * dt:
            # Log-normal jump size
            J = np.random.normal(mu_jump, sigma_jump)
            jump = S_t * (np.exp(J) - 1)
        else:
            jump = 0.0
            
        return S_t + diffusion + jump

    def kou_double_exponential(self, S_t: float, dt: float) -> float:
        """
        Implement Kou double exponential model with asymmetric jumps
        """
        sigma = self.params['sigma']
        lambda_base = self.params['lambda_base']
        p_up = self.params['p_up']
        eta_up = self.params['eta_up']
        eta_down = self.params['eta_down']
        
        # Diffusion
        dW = np.random.normal(0, np.sqrt(dt))
        diffusion = sigma * S_t * dW
        
        # Jump
        jump = 0.0
        if np.random.random() < lambda_base * dt:
            # Double exponential jump size
            if np.random.random() < p_up:
                # Upward jump: Exponential(eta_up)
                Y = np.random.exponential(1/eta_up)
            else:
                # Downward jump: -Exponential(eta_down)
                Y = -np.random.exponential(1/eta_down)
            
            jump = S_t * (np.exp(Y) - 1)
            
        return S_t + diffusion + jump

    def calibrate_mle(self, historical_prices: np.array):
        """
        Maximum Likelihood Estimation (MLE) for Merton Jump Diffusion.
        Uses scipy.optimize to minimize negative log-likelihood.
        """
        from scipy.optimize import minimize
        
        returns = np.diff(np.log(historical_prices))
        dt = 1.0 / 252.0 # Daily returns assumption
        
        def neg_log_likelihood(params):
            sigma, lambda_, mu_j, sigma_j = params
            if sigma <= 0 or lambda_ <= 0 or sigma_j <= 0:
                return 1e10
                
            # Density of Merton JD is sum of weighted normal densities
            # Approximation: Truncate series at N=10 jumps
            n_jumps = np.arange(0, 10)
            pdf = 0
            for k in n_jumps:
                p_k = (np.exp(-lambda_ * dt) * (lambda_ * dt)**k) / np.math.factorial(k)
                mu_k = 0.0 + k * mu_j # Drift 0
                var_k = dt * sigma**2 + k * sigma_j**2
                if var_k <= 0: continue
                
                # Normal PDF
                term = (1 / np.sqrt(2 * np.pi * var_k)) * np.exp(-(returns - mu_k)**2 / (2 * var_k))
                pdf += p_k * term
                
            # Avoid log(0)
            pdf = np.maximum(pdf, 1e-10)
            return -np.sum(np.log(pdf))

        # Initial guess: sigma=0.2, lambda=1, mu_j=0, sigma_j=0.1
        initial_guess = [0.2, 1.0, 0.0, 0.1]
        result = minimize(neg_log_likelihood, initial_guess, method='Nelder-Mead')
        
        if result.success:
            self.params['sigma'] = result.x[0]
            self.params['lambda_base'] = result.x[1]
            self.params['mu_jump'] = result.x[2]
            self.params['sigma_jump'] = result.x[3]
            return True
        return False

    def calibrate_method_of_moments(self, historical_prices: np.array):
        """
        Method of Moments calibration.
        Matches empirical mean, variance, skewness, and kurtosis to theoretical moments.
        """
        returns = np.diff(np.log(historical_prices))
        dt = 1.0 / 252.0
        
        mean_r = np.mean(returns)
        var_r = np.var(returns)
        skew_r = np.mean(((returns - mean_r) / np.sqrt(var_r))**3)
        kurt_r = np.mean(((returns - mean_r) / np.sqrt(var_r))**4)
        
        # Simplified mapping (heuristic for stability)
        # High kurtosis -> higher lambda/jump size
        self.params['sigma'] = np.sqrt(var_r / dt) * 0.8 # Assign 80% vol to diffusion
        
        excess_kurtosis = kurt_r - 3
        if excess_kurtosis > 0:
            # Rough approximation for jump parameters based on excess kurtosis
            self.params['lambda_base'] = max(1.0, excess_kurtosis * 10)
            self.params['sigma_jump'] = np.sqrt(var_r) * 2
        
        return {
            "mean": mean_r, "var": var_r, "skew": skew_r, "kurt": kurt_r
        }

    def calibrate_mcmc(self, historical_prices: np.array, iterations: int = 1000):
        """
        Bayesian MCMC using Metropolis-Hastings sampler (No external deps).
        Estimates sigma and lambda_base.
        """
        returns = np.diff(np.log(historical_prices))
        dt = 1.0 / 252.0
        
        current_sigma = self.params['sigma']
        current_lambda = self.params['lambda_base']
        
        def log_posterior(sigma, lam, data):
            if sigma <= 0 or lam <= 0: return -np.inf
            # Prior: Uniform > 0
            # Likelihood (Simplified to Normal for diffusion + rare large jumps)
            # This is a toy implementation for demonstration
            log_lik = -0.5 * np.sum((data / (sigma * np.sqrt(dt)))**2)
            return log_lik

        for i in range(iterations):
            # Propose new values
            prop_sigma = current_sigma + np.random.normal(0, 0.01)
            prop_lambda = current_lambda + np.random.normal(0, 0.1)
            
            curr_log = log_posterior(current_sigma, current_lambda, returns)
            prop_log = log_posterior(prop_sigma, prop_lambda, returns)
            
            if np.log(np.random.random()) < (prop_log - curr_log):
                current_sigma = prop_sigma
                current_lambda = prop_lambda
                
        self.params['sigma'] = current_sigma
        self.params['lambda_base'] = current_lambda
        return {'sigma': current_sigma, 'lambda': current_lambda}

    def liquidity_adjusted_intensity(self, liquidity_t: float) -> float:
        """
        Implement liquidity-adjusted jump intensity: λ(t) = λ_base × f(liquidity_t)
        """
        # Simple model: Lower liquidity -> Higher jump probability
        # f(L) = 1 + 1/L (normalized)
        lambda_base = self.params['lambda_base']
        if liquidity_t <= 0:
            return lambda_base * 2.0 # Max multiplier
            
        adjustment = 1.0 + (0.1 / liquidity_t)
        return lambda_base * min(adjustment, 5.0) # Cap at 5x

    def simulate_path(self, S0: float, T: float, steps: int, simulations: int = 1) -> np.ndarray:
        """
        Simulate price paths using Merton Jump Diffusion model.
        Returns: Array of shape (steps + 1, simulations)
        """
        dt = T / steps
        sigma = self.params['sigma']
        lambda_base = self.params['lambda_base']
        mu_jump = self.params['mu_jump']
        sigma_jump = self.params['sigma_jump']
        
        # Initialize paths
        paths = np.zeros((steps + 1, simulations))
        paths[0] = S0
        
        # Vectorized simulation
        for t in range(1, steps + 1):
            # Diffusion component: Normal(0, sqrt(dt))
            Z = np.random.standard_normal(simulations)
            diffusion = (0 - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            
            # Jump component: Poisson(lambda * dt)
            # Number of jumps in this step for each sim
            N = np.random.poisson(lambda_base * dt, simulations)
            
            # Total jump magnitude for this step
            # Sum of N log-normal jumps: We approximate if N > 0
            jump_magnitude = np.zeros(simulations)
            
            # Indices where jumps occur
            jump_indices = np.where(N > 0)[0]
            
            if len(jump_indices) > 0:
                # For each simulation with jumps, sum up random jump sizes
                # Simplified: Just take one aggregate jump if N=1 (most common)
                # or sum N normal variables for the log-return
                
                # J ~ Normal(mu_jump, sigma_jump)
                # Total jump effect in log price is sum of J_i
                
                # Vectorized approach for variable N is tricky, loop over max N or just loop indices
                # Optimization: Since lambda*dt is small, N is usually 0 or 1.
                
                for idx in jump_indices:
                    n_jumps = N[idx]
                    # Sum of n_jumps random variables from N(mu_jump, sigma_jump)
                    # Sum is N(n * mu, sqrt(n) * sigma)
                    total_jump = np.random.normal(n_jumps * mu_jump, np.sqrt(n_jumps) * sigma_jump)
                    jump_magnitude[idx] = total_jump

            # Update log price
            # S_t = S_{t-1} * exp(diffusion + jumps)
            # log(S_t) = log(S_{t-1}) + diffusion + jumps
            
            # We work with returns to avoid loop dependency if possible, but here we iterate t
            paths[t] = paths[t-1] * np.exp(diffusion + jump_magnitude)
            
        return paths
