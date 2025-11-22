"""
Monitoring and Alerting System for Prediction Market ABM.

Features:
- Structured logging with structlog
- Alert thresholds for critical/warning/info events
- Integration with multiple notification channels (Slack, Email, PagerDuty)
- Performance metrics tracking
- Error rate monitoring
"""
import structlog
import logging
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import os
import json
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertLevel(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    metric: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    level: AlertLevel
    message_template: str
    cooldown_seconds: int = 300  # 5 minutes default
    last_triggered: float = 0.0


@dataclass
class Alert:
    """Alert instance."""
    level: AlertLevel
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: float = field(default_factory=time.time)


class AlertManager:
    """Manage alerts and notifications."""

    def __init__(self):
        """Initialize alert manager."""
        self.thresholds: List[AlertThreshold] = []
        self.alerts_history: List[Alert] = []
        self.notification_handlers: List[Callable[[Alert], None]] = []

        # Configure default thresholds
        self._configure_default_thresholds()

    def _configure_default_thresholds(self):
        """Set up default alert thresholds."""
        # CRITICAL alerts
        self.add_threshold(
            metric="drawdown",
            threshold=15.0,
            comparison="gt",
            level=AlertLevel.CRITICAL,
            message_template="CRITICAL: Drawdown exceeded {threshold}%! Current: {value:.2f}%"
        )

        self.add_threshold(
            metric="api_errors_per_minute",
            threshold=10.0,
            comparison="gt",
            level=AlertLevel.CRITICAL,
            message_template="CRITICAL: API errors spiking! {value:.0f} errors/min"
        )

        self.add_threshold(
            metric="position_limit_breach",
            threshold=1.0,
            comparison="gt",
            level=AlertLevel.CRITICAL,
            message_template="CRITICAL: Position limit breached! Value: {value:.0f}"
        )

        # WARNING alerts
        self.add_threshold(
            metric="slippage_bps",
            threshold=2.0,
            comparison="gt",
            level=AlertLevel.WARNING,
            message_template="WARNING: High slippage detected! {value:.2f} bps"
        )

        self.add_threshold(
            metric="latency_ms",
            threshold=500.0,
            comparison="gt",
            level=AlertLevel.WARNING,
            message_template="WARNING: High latency! {value:.0f}ms"
        )

        self.add_threshold(
            metric="fill_rate",
            threshold=0.8,
            comparison="lt",
            level=AlertLevel.WARNING,
            message_template="WARNING: Low fill rate! {value:.2%}"
        )

        # INFO alerts
        self.add_threshold(
            metric="large_trade",
            threshold=10000.0,
            comparison="gt",
            level=AlertLevel.INFO,
            message_template="INFO: Large trade executed: ${value:.0f}"
        )

        self.add_threshold(
            metric="regime_change",
            threshold=1.0,
            comparison="eq",
            level=AlertLevel.INFO,
            message_template="INFO: Market regime change detected"
        )

    def add_threshold(
        self,
        metric: str,
        threshold: float,
        comparison: str,
        level: AlertLevel,
        message_template: str,
        cooldown_seconds: int = 300
    ):
        """Add an alert threshold."""
        self.thresholds.append(
            AlertThreshold(
                metric=metric,
                threshold=threshold,
                comparison=comparison,
                level=level,
                message_template=message_template,
                cooldown_seconds=cooldown_seconds
            )
        )

    def check_threshold(self, metric: str, value: float) -> Optional[Alert]:
        """Check if a metric exceeds its threshold."""
        current_time = time.time()

        for threshold in self.thresholds:
            if threshold.metric != metric:
                continue

            # Check cooldown
            if current_time - threshold.last_triggered < threshold.cooldown_seconds:
                continue

            # Check threshold
            triggered = False
            if threshold.comparison == "gt" and value > threshold.threshold:
                triggered = True
            elif threshold.comparison == "lt" and value < threshold.threshold:
                triggered = True
            elif threshold.comparison == "eq" and abs(value - threshold.threshold) < 0.01:
                triggered = True

            if triggered:
                threshold.last_triggered = current_time
                message = threshold.message_template.format(
                    threshold=threshold.threshold,
                    value=value
                )

                alert = Alert(
                    level=threshold.level,
                    metric=metric,
                    value=value,
                    threshold=threshold.threshold,
                    message=message
                )

                self.alerts_history.append(alert)
                self._notify(alert)
                return alert

        return None

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler."""
        self.notification_handlers.append(handler)

    def _notify(self, alert: Alert):
        """Send notifications for an alert."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                log.error("notification_failed", error=str(e), alert=alert)

    def get_recent_alerts(self, minutes: int = 60) -> List[Alert]:
        """Get alerts from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        return [a for a in self.alerts_history if a.timestamp >= cutoff_time]


class SlackNotificationHandler:
    """Send alerts to Slack."""

    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize Slack handler."""
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")

    def __call__(self, alert: Alert):
        """Send alert to Slack."""
        if not self.webhook_url:
            return

        try:
            import requests

            color = {
                AlertLevel.CRITICAL: "#FF0000",
                AlertLevel.WARNING: "#FFA500",
                AlertLevel.INFO: "#0000FF"
            }[alert.level]

            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"{alert.level.value.upper()} Alert",
                    "text": alert.message,
                    "fields": [
                        {"title": "Metric", "value": alert.metric, "short": True},
                        {"title": "Value", "value": f"{alert.value:.2f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold:.2f}", "short": True},
                        {"title": "Time", "value": datetime.fromtimestamp(alert.timestamp).isoformat(), "short": True}
                    ]
                }]
            }

            requests.post(self.webhook_url, json=payload, timeout=5)
        except Exception as e:
            log.error("slack_notification_failed", error=str(e))


class EmailNotificationHandler:
    """Send alerts via email."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        recipient: Optional[str] = None
    ):
        """Initialize email handler."""
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.recipient = recipient or os.getenv("ALERT_EMAIL")

    def __call__(self, alert: Alert):
        """Send alert via email."""
        if not all([self.smtp_user, self.smtp_password, self.recipient]):
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = self.recipient
            msg['Subject'] = f"[{alert.level.value.upper()}] Prediction Market Alert: {alert.metric}"

            body = f"""
Alert Level: {alert.level.value.upper()}
Metric: {alert.metric}
Current Value: {alert.value:.2f}
Threshold: {alert.threshold:.2f}
Message: {alert.message}
Time: {datetime.fromtimestamp(alert.timestamp).isoformat()}

This is an automated alert from the Prediction Market ABM system.
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
            server.quit()

        except Exception as e:
            log.error("email_notification_failed", error=str(e))


class ConsoleNotificationHandler:
    """Print alerts to console (useful for development)."""

    def __call__(self, alert: Alert):
        """Print alert to console."""
        color_codes = {
            AlertLevel.CRITICAL: "\033[91m",  # Red
            AlertLevel.WARNING: "\033[93m",   # Yellow
            AlertLevel.INFO: "\033[94m"       # Blue
        }
        reset_code = "\033[0m"

        color = color_codes[alert.level]
        print(f"{color}[{alert.level.value.upper()}] {alert.message}{reset_code}")


# Configure structured logging
def configure_logging(log_level: str = "INFO", json_logs: bool = False):
    """Configure structured logging."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        # JSON output for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Pretty console output for development
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Global logger instance
configure_logging()
log = structlog.get_logger()


# Global alert manager
alert_manager = AlertManager()

# Add default handlers
alert_manager.add_notification_handler(ConsoleNotificationHandler())

# Add Slack/Email handlers if configured
if os.getenv("SLACK_WEBHOOK_URL"):
    alert_manager.add_notification_handler(SlackNotificationHandler())

if os.getenv("ALERT_EMAIL"):
    alert_manager.add_notification_handler(EmailNotificationHandler())


# Example usage functions
def log_order_execution(order_id: str, price: float, size: float, latency_ms: float, slippage_bps: float):
    """Log order execution with metrics."""
    log.info(
        "order_executed",
        order_id=order_id,
        price=price,
        size=size,
        latency_ms=latency_ms,
        slippage_bps=slippage_bps
    )

    # Check alert thresholds
    alert_manager.check_threshold("latency_ms", latency_ms)
    alert_manager.check_threshold("slippage_bps", slippage_bps)


def log_trade(trade_id: str, side: str, quantity: float, price: float, pnl: float):
    """Log trade execution."""
    log.info(
        "trade_executed",
        trade_id=trade_id,
        side=side,
        quantity=quantity,
        price=price,
        pnl=pnl
    )

    # Check for large trades
    trade_value = quantity * price
    alert_manager.check_threshold("large_trade", trade_value)


def log_performance(sharpe: float, drawdown: float, win_rate: float, pnl: float):
    """Log performance metrics."""
    log.info(
        "performance_update",
        sharpe=sharpe,
        drawdown=drawdown,
        win_rate=win_rate,
        pnl=pnl
    )

    # Check critical thresholds
    alert_manager.check_threshold("drawdown", abs(drawdown))


def log_api_error(api: str, endpoint: str, error: str, status_code: Optional[int] = None):
    """Log API error."""
    log.error(
        "api_error",
        api=api,
        endpoint=endpoint,
        error=error,
        status_code=status_code
    )


def log_system_metrics(cpu_percent: float, memory_percent: float, queue_depth: int):
    """Log system resource metrics."""
    log.info(
        "system_metrics",
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        queue_depth=queue_depth
    )


# Export public API
__all__ = [
    'log',
    'alert_manager',
    'AlertLevel',
    'Alert',
    'configure_logging',
    'log_order_execution',
    'log_trade',
    'log_performance',
    'log_api_error',
    'log_system_metrics'
]
