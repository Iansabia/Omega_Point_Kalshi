"""
Circuit Breaker pattern for external API calls.

Prevents cascading failures by:
- Tracking failure rates
- Opening circuit after threshold breaches
- Allowing periodic retry attempts
- Falling back to cached data or degraded service

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests fail fast
- HALF_OPEN: Testing if service recovered, limited requests allowed
"""
import time
from typing import Callable, Any, Optional, Dict
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps
import threading


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before half-open
    expected_exception: type = Exception  # Exception type to catch
    success_threshold: int = 2  # Successful calls needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float = 0.0
    state_changes: int = 0


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Usage:
        breaker = CircuitBreaker(
            name="kalshi_api",
            failure_threshold=5,
            recovery_timeout=60
        )

        @breaker
        def call_api():
            return api.get_data()
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        timeout: float = 30.0,
        expected_exception: type = Exception,
        fallback: Optional[Callable] = None
    ):
        """Initialize circuit breaker."""
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            success_threshold=success_threshold,
            timeout=timeout
        )
        self.fallback = fallback

        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.lock = threading.Lock()

        # For logging
        self._state_change_callbacks: list[Callable] = []

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        with self.lock:
            self.stats.total_calls += 1

            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Retry in {self._time_until_retry():.0f}s"
                    )

            # Try the call
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result

            except self.config.expected_exception as e:
                self._on_failure()

                # Use fallback if available
                if self.fallback:
                    return self.fallback(*args, **kwargs)

                raise

    def _on_success(self):
        """Handle successful call."""
        self.stats.successful_calls += 1
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0

        # Transition from HALF_OPEN to CLOSED if enough successes
        if self.state == CircuitState.HALF_OPEN:
            if self.stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to_closed()

    def _on_failure(self):
        """Handle failed call."""
        self.stats.failed_calls += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = time.time()

        # Open circuit if failure threshold exceeded
        if self.stats.consecutive_failures >= self.config.failure_threshold:
            self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        return (time.time() - self.stats.last_failure_time) >= self.config.recovery_timeout

    def _time_until_retry(self) -> float:
        """Calculate seconds until retry is allowed."""
        elapsed = time.time() - self.stats.last_failure_time
        return max(0, self.config.recovery_timeout - elapsed)

    def _transition_to_open(self):
        """Transition to OPEN state."""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.stats.state_changes += 1
            self._notify_state_change(CircuitState.OPEN)

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        if self.state != CircuitState.HALF_OPEN:
            self.state = CircuitState.HALF_OPEN
            self.stats.state_changes += 1
            self.stats.consecutive_successes = 0
            self.stats.consecutive_failures = 0
            self._notify_state_change(CircuitState.HALF_OPEN)

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.stats.state_changes += 1
            self.stats.consecutive_failures = 0
            self._notify_state_change(CircuitState.CLOSED)

    def _notify_state_change(self, new_state: CircuitState):
        """Notify callbacks of state change."""
        for callback in self._state_change_callbacks:
            try:
                callback(self.name, new_state, self.stats)
            except Exception:
                pass  # Don't let callback errors affect circuit breaker

    def register_state_change_callback(self, callback: Callable):
        """Register callback for state changes."""
        self._state_change_callbacks.append(callback)

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        with self.lock:
            self._transition_to_closed()
            self.stats.consecutive_failures = 0
            self.stats.consecutive_successes = 0

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "success_rate": (
                self.stats.successful_calls / self.stats.total_calls
                if self.stats.total_calls > 0 else 0
            ),
            "consecutive_failures": self.stats.consecutive_failures,
            "consecutive_successes": self.stats.consecutive_successes,
            "state_changes": self.stats.state_changes,
            "time_since_last_failure": (
                time.time() - self.stats.last_failure_time
                if self.stats.last_failure_time > 0 else None
            )
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerRegistry:
    """Global registry for circuit breakers."""

    def __init__(self):
        """Initialize registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def register(self, breaker: CircuitBreaker):
        """Register a circuit breaker."""
        with self._lock:
            self._breakers[breaker.name] = breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global registry
registry = CircuitBreakerRegistry()


# Pre-configured circuit breakers for common APIs
kalshi_breaker = CircuitBreaker(
    name="kalshi_api",
    failure_threshold=5,
    recovery_timeout=60,
    success_threshold=2,
    timeout=30
)
registry.register(kalshi_breaker)

polymarket_breaker = CircuitBreaker(
    name="polymarket_api",
    failure_threshold=5,
    recovery_timeout=60,
    success_threshold=2,
    timeout=30
)
registry.register(polymarket_breaker)

sportradar_breaker = CircuitBreaker(
    name="sportradar_api",
    failure_threshold=3,
    recovery_timeout=120,
    success_threshold=2,
    timeout=60
)
registry.register(sportradar_breaker)

gemini_breaker = CircuitBreaker(
    name="gemini_api",
    failure_threshold=10,
    recovery_timeout=30,
    success_threshold=3,
    timeout=10
)
registry.register(gemini_breaker)


# State change logging callback
def log_circuit_state_change(name: str, new_state: CircuitState, stats: CircuitBreakerStats):
    """Log circuit breaker state changes."""
    try:
        from ..visualization.monitoring import log
        log.warning(
            "circuit_breaker_state_change",
            breaker=name,
            new_state=new_state.value,
            consecutive_failures=stats.consecutive_failures,
            total_calls=stats.total_calls,
            failed_calls=stats.failed_calls
        )
    except ImportError:
        print(f"Circuit breaker '{name}' transitioned to {new_state.value}")


# Register logging callback for all breakers
for breaker in [kalshi_breaker, polymarket_breaker, sportradar_breaker, gemini_breaker]:
    breaker.register_state_change_callback(log_circuit_state_change)


# Export public API
__all__ = [
    'CircuitBreaker',
    'CircuitBreakerOpenError',
    'CircuitState',
    'registry',
    'kalshi_breaker',
    'polymarket_breaker',
    'sportradar_breaker',
    'gemini_breaker'
]
