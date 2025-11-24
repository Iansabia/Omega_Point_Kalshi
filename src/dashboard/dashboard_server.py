"""
Real-time Trading Dashboard Server

Provides a web interface for monitoring live trading during MNF game.

Features:
- Real-time game state updates (ESPN)
- Live market prices (Kalshi)
- Model win probability predictions
- Arbitrage signals
- Trade execution controls
- Circuit breaker status
- Performance metrics

Usage:
    PYTHONPATH=. ./venv/bin/python3 src/dashboard/dashboard_server.py

    Then open: http://localhost:8000
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.execution.circuit_breaker import registry
from src.execution.audit_log import audit_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kalshi Trading Dashboard")

# Global state (updated by trading engine)
dashboard_state = {
    "game_state": {},
    "market_state": {},
    "model_prediction": {},
    "last_signal": {},
    "trades": [],
    "performance": {
        "total_trades": 0,
        "winning_trades": 0,
        "total_pnl": 0.0,
        "win_rate": 0.0
    },
    "circuit_breaker": {},
    "config": {
        "paper_trading": True,
        "min_edge": 0.10,
        "max_position_size": 100,
        "auto_trading": False
    },
    "status": "disconnected",
    "last_update": 0
}

# WebSocket connections
active_connections: List[WebSocket] = []


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")


manager = ConnectionManager()


@app.get("/")
async def get_dashboard():
    """Serve the dashboard HTML."""
    html_path = Path(__file__).parent / "dashboard.html"

    if html_path.exists():
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content=get_fallback_html())


@app.get("/api/state")
async def get_state():
    """Get current dashboard state."""
    return dashboard_state


@app.get("/api/circuit-breaker")
async def get_circuit_breaker_stats():
    """Get circuit breaker statistics."""
    return registry.get_all_stats()


@app.get("/api/audit-log")
async def get_audit_log(limit: int = 50):
    """Get recent audit log entries."""
    try:
        entries = audit_logger.wal.query(max_results=limit)
        return {
            "entries": [entry.to_dict() for entry in entries],
            "total": len(entries)
        }
    except Exception as e:
        logger.error(f"Error fetching audit log: {e}")
        return {"entries": [], "total": 0, "error": str(e)}


@app.post("/api/config")
async def update_config(config: dict):
    """Update dashboard configuration."""
    dashboard_state["config"].update(config)
    await manager.broadcast({
        "type": "config_update",
        "config": dashboard_state["config"]
    })
    return {"status": "success", "config": dashboard_state["config"]}


@app.post("/api/execute-trade")
async def execute_trade(trade: dict):
    """Manually execute a trade (for manual mode)."""
    # This would trigger the trading engine
    logger.info(f"Manual trade execution requested: {trade}")

    # Add to trades list
    trade_record = {
        "timestamp": time.time(),
        "type": "manual",
        **trade
    }
    dashboard_state["trades"].insert(0, trade_record)

    # Broadcast update
    await manager.broadcast({
        "type": "trade_executed",
        "trade": trade_record
    })

    return {"status": "success", "trade": trade_record}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)

    try:
        # Send initial state
        await websocket.send_json({
            "type": "initial_state",
            "state": dashboard_state
        })

        # Keep connection alive and listen for client messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle client messages
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "subscribe":
                # Client subscribed to updates
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def update_dashboard(update_type: str, data: dict):
    """Update dashboard state and broadcast to clients."""
    dashboard_state["last_update"] = time.time()

    if update_type == "game_state":
        dashboard_state["game_state"] = data
    elif update_type == "market_state":
        dashboard_state["market_state"] = data
    elif update_type == "model_prediction":
        dashboard_state["model_prediction"] = data
    elif update_type == "signal":
        dashboard_state["last_signal"] = data
    elif update_type == "trade":
        dashboard_state["trades"].insert(0, data)
        # Keep only last 20 trades
        dashboard_state["trades"] = dashboard_state["trades"][:20]
        # Update performance
        update_performance(data)
    elif update_type == "status":
        dashboard_state["status"] = data.get("status", "unknown")

    # Broadcast to all connected clients
    await manager.broadcast({
        "type": update_type,
        "data": data,
        "timestamp": time.time()
    })


def update_performance(trade: dict):
    """Update performance metrics."""
    perf = dashboard_state["performance"]
    perf["total_trades"] += 1

    if trade.get("pnl", 0) > 0:
        perf["winning_trades"] += 1

    perf["total_pnl"] += trade.get("pnl", 0)
    perf["win_rate"] = (perf["winning_trades"] / perf["total_trades"]) if perf["total_trades"] > 0 else 0.0


def get_fallback_html():
    """Fallback HTML if dashboard.html not found."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kalshi Trading Dashboard</title>
        <style>
            body { font-family: Arial; text-align: center; padding: 50px; }
            .error { color: red; }
        </style>
    </head>
    <body>
        <h1>Dashboard Error</h1>
        <p class="error">dashboard.html not found</p>
        <p>Expected location: src/dashboard/dashboard.html</p>
    </body>
    </html>
    """


# Export for use by trading engine
__all__ = ["app", "update_dashboard", "dashboard_state"]


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  KALSHI TRADING DASHBOARD")
    print("=" * 60)
    print()
    print("  Dashboard URL: http://localhost:8000")
    print("  API Docs:      http://localhost:8000/docs")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
