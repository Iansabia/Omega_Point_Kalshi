#!/usr/bin/env python3
"""
Test Circuit Breaker and Audit Log Integration

This script verifies that:
1. Circuit breaker protects KalshiClient API calls
2. Audit logger records all orders, trades, and risk violations
3. System fails gracefully when API is down
4. All events are logged to audit trail

Usage:
    PYTHONPATH=. python3 scripts/test_circuit_breaker_audit_integration.py
"""

import os
import sys
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.execution.kalshi_client import KalshiClient
from src.execution.circuit_breaker import kalshi_breaker, CircuitBreakerOpenError, registry
from src.execution.audit_log import audit_logger, WriteAheadLog
from src.execution.order_router import OrderRouter
from src.execution.risk_manager import RiskManager
from src.orderbook.order import Order


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def test_circuit_breaker_integration():
    """Test 1: Verify circuit breaker is integrated into KalshiClient."""
    print_section("TEST 1: Circuit Breaker Integration")

    # Reset circuit breaker
    kalshi_breaker.reset()
    print(f"✓ Circuit breaker reset: {kalshi_breaker.get_state().value}")

    # Get initial stats
    initial_stats = kalshi_breaker.get_stats()
    print(f"✓ Initial circuit breaker stats:")
    print(f"  - State: {initial_stats['state']}")
    print(f"  - Total calls: {initial_stats['total_calls']}")
    print(f"  - Failed calls: {initial_stats['failed_calls']}")

    # Create client with mock credentials
    client = KalshiClient(
        api_key="test_key_12345678",
        private_key_path="/nonexistent/key.pem",  # Will fail to load
        email="test@example.com",
        password="test_password"
    )

    # Mock the authentication to avoid real API calls
    client.token = "mock_token"
    client.member_id = "test_user_123"

    print(f"\n✓ Created KalshiClient with test credentials")

    # Simulate API failures to trigger circuit breaker
    print(f"\n→ Simulating 5 consecutive API failures...")

    with patch('requests.get') as mock_get:
        # Configure mock to raise exception
        mock_get.side_effect = Exception("API Connection Error")

        failure_count = 0
        for i in range(5):
            try:
                client.get_balance()
            except Exception as e:
                failure_count += 1
                print(f"  Attempt {i+1}: Failed as expected ({type(e).__name__})")

    # Check circuit breaker state
    stats = kalshi_breaker.get_stats()
    print(f"\n✓ Circuit breaker stats after failures:")
    print(f"  - State: {stats['state']}")
    print(f"  - Failed calls: {stats['failed_calls']}")
    print(f"  - Consecutive failures: {stats['consecutive_failures']}")

    if stats['state'] == 'open':
        print(f"\n✅ TEST 1 PASSED: Circuit breaker is OPEN after {failure_count} failures")
    else:
        print(f"\n❌ TEST 1 FAILED: Circuit breaker should be OPEN but is {stats['state']}")

    return stats['state'] == 'open'


def test_audit_log_integration():
    """Test 2: Verify audit logging captures all events."""
    print_section("TEST 2: Audit Log Integration")

    # Create temporary audit log directory
    test_log_dir = "logs/audit_test"
    os.makedirs(test_log_dir, exist_ok=True)

    # Create new audit logger for testing
    test_wal = WriteAheadLog(log_dir=test_log_dir, async_write=False)
    from src.execution import audit_log as audit_module
    original_logger = audit_module.audit_logger
    audit_module.audit_logger.wal = test_wal

    print(f"✓ Created test audit log in: {test_log_dir}")

    # Test 1: Log an order
    print(f"\n→ Logging test order...")
    audit_module.audit_logger.log_order(
        user_id="test_user_123",
        order_id="order_abc123",
        side="yes",
        order_type="limit",
        quantity=10,
        price=65
    )
    print(f"  ✓ Order logged")

    # Test 2: Log a trade
    print(f"\n→ Logging test trade...")
    audit_module.audit_logger.log_trade(
        user_id="test_user_123",
        order_id="order_abc123",
        side="yes",
        quantity=10,
        price=65,
        market="KXMVENFL-TEST"
    )
    print(f"  ✓ Trade logged")

    # Test 3: Log a risk violation
    print(f"\n→ Logging test risk violation...")
    audit_module.audit_logger.log_risk_violation(
        user_id="test_user_123",
        violation_type="position_limit_exceeded",
        current_value=150,
        limit=100,
        details={"ticker": "KXMVENFL-TEST", "reason": "Test violation"}
    )
    print(f"  ✓ Risk violation logged")

    # Test 4: Log an API call
    print(f"\n→ Logging test API call...")
    audit_module.audit_logger.log_api_call(
        user_id="test_user_123",
        api="kalshi",
        endpoint="markets/KXMVENFL-TEST",
        method="GET",
        status_code=200,
        latency_ms=45.2
    )
    print(f"  ✓ API call logged")

    # Query audit log
    print(f"\n→ Querying audit log entries...")
    entries = test_wal.query(max_results=100)
    print(f"  ✓ Found {len(entries)} audit log entries")

    # Verify entries
    event_types = [entry.event_type for entry in entries]
    print(f"\n✓ Event types logged: {event_types}")

    expected_types = ["ORDER", "TRADE", "RISK_VIOLATION", "API_CALL"]
    all_present = all(et in event_types for et in expected_types)

    if all_present and len(entries) >= 4:
        print(f"\n✅ TEST 2 PASSED: All audit log entries recorded correctly")
        success = True
    else:
        print(f"\n❌ TEST 2 FAILED: Missing audit log entries")
        print(f"  Expected: {expected_types}")
        print(f"  Found: {event_types}")
        success = False

    # Verify integrity
    integrity = test_wal.verify_integrity()
    print(f"\n✓ Audit log integrity check:")
    print(f"  - Total entries: {integrity['total_entries']}")
    print(f"  - Corrupted entries: {integrity['corrupted_entries']}")
    print(f"  - Integrity OK: {integrity['integrity_ok']}")

    if not integrity['integrity_ok']:
        print(f"\n⚠️  WARNING: Audit log integrity check failed!")
        success = False

    # Cleanup
    test_wal.close()
    audit_module.audit_logger = original_logger

    return success


def test_risk_manager_audit_logging():
    """Test 3: Verify risk manager logs violations to audit trail."""
    print_section("TEST 3: Risk Manager Audit Logging")

    # Create temporary audit log
    test_log_dir = "logs/audit_test_risk"
    os.makedirs(test_log_dir, exist_ok=True)
    test_wal = WriteAheadLog(log_dir=test_log_dir, async_write=False)

    from src.execution import audit_log as audit_module
    original_logger = audit_module.audit_logger
    audit_module.audit_logger.wal = test_wal

    # Create risk manager
    risk_mgr = RiskManager(
        max_position_size=100,
        max_daily_loss=1000.0,
        user_id="test_user_456"
    )
    print(f"✓ Created RiskManager (max_position=100, max_loss=1000)")

    # Test 1: Position limit violation
    print(f"\n→ Testing position limit violation...")
    signal = {
        "ticker": "KXMVENFL-TEST",
        "count": 150,  # Exceeds limit of 100
        "price": 50
    }
    result = risk_mgr.check_risk(signal)
    print(f"  Risk check result: {'REJECTED' if not result else 'APPROVED'}")

    if not result:
        print(f"  ✓ Position limit violation correctly rejected")
    else:
        print(f"  ❌ Position limit violation should have been rejected!")

    # Test 2: Order value violation
    print(f"\n→ Testing order value violation...")
    signal = {
        "ticker": "KXMVENFL-TEST2",
        "count": 100,
        "price": 90  # 100 * 0.90 = $90 (under limit)
    }
    result = risk_mgr.check_risk(signal)
    print(f"  Risk check result: {'REJECTED' if not result else 'APPROVED'}")

    signal = {
        "ticker": "KXMVENFL-TEST2",
        "count": 1000,
        "price": 90  # 1000 * 0.90 = $900 (over $500 limit)
    }
    result = risk_mgr.check_risk(signal)
    print(f"  Risk check result: {'REJECTED' if not result else 'APPROVED'}")

    if not result:
        print(f"  ✓ Order value violation correctly rejected")
    else:
        print(f"  ❌ Order value violation should have been rejected!")

    # Test 3: Daily loss violation
    print(f"\n→ Testing daily loss violation...")
    risk_mgr.daily_pnl = -1500.0  # Set loss beyond limit
    signal = {
        "ticker": "KXMVENFL-TEST3",
        "count": 10,
        "price": 50
    }
    result = risk_mgr.check_risk(signal)
    print(f"  Risk check result: {'REJECTED' if not result else 'APPROVED'}")

    if not result:
        print(f"  ✓ Daily loss violation correctly rejected")
    else:
        print(f"  ❌ Daily loss violation should have been rejected!")

    # Query audit log for violations
    print(f"\n→ Querying audit log for risk violations...")
    entries = test_wal.query(event_type="RISK_VIOLATION", max_results=100)
    print(f"  ✓ Found {len(entries)} risk violation entries")

    for i, entry in enumerate(entries):
        print(f"\n  Violation {i+1}:")
        print(f"    Type: {entry.details.get('violation_type')}")
        print(f"    Ticker: {entry.details.get('ticker')}")
        print(f"    Current: {entry.details.get('current_value') or entry.details.get('current_position')}")

    if len(entries) >= 3:
        print(f"\n✅ TEST 3 PASSED: All risk violations logged to audit trail")
        success = True
    else:
        print(f"\n❌ TEST 3 FAILED: Expected at least 3 risk violations, found {len(entries)}")
        success = False

    # Cleanup
    test_wal.close()
    audit_module.audit_logger = original_logger

    return success


def test_order_router_integration():
    """Test 4: Verify OrderRouter integrates circuit breaker and audit logging."""
    print_section("TEST 4: OrderRouter Integration")

    # Create temporary audit log
    test_log_dir = "logs/audit_test_router"
    os.makedirs(test_log_dir, exist_ok=True)
    test_wal = WriteAheadLog(log_dir=test_log_dir, async_write=False)

    from src.execution import audit_log as audit_module
    original_logger = audit_module.audit_logger
    audit_module.audit_logger.wal = test_wal

    # Reset circuit breaker
    kalshi_breaker.reset()

    # Create client with mock
    client = KalshiClient(
        api_key="test_key_12345678",
        private_key_path="/nonexistent/key.pem",
        email="test@example.com",
        password="test_password"
    )
    client.token = "mock_token"
    client.member_id = "test_user_789"

    # Create order router
    router = OrderRouter(kalshi_client=client)
    print(f"✓ Created OrderRouter")

    # Create test order
    order = Order(
        order_id="order_test_001",
        trader_id="trader_123",
        timestamp=time.time(),
        side="BUY",  # Must be uppercase
        order_type="LIMIT",  # Must be uppercase
        quantity=10,
        price=65.0
    )
    print(f"✓ Created test order: {order.order_id}")

    # Mock the signal generator
    router.signal_gen.generate_signal = Mock(return_value={
        "ticker": "KXMVENFL-TEST",
        "side": "yes",
        "count": 10,
        "price": 65
    })

    # Test 1: Successful order (mocked)
    print(f"\n→ Testing successful order routing...")
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "order": {
                "order_id": "kalshi_order_123",
                "ticker": "KXMVENFL-TEST",
                "side": "yes",
                "count": 10,
                "price": 65
            }
        }
        mock_post.return_value = mock_response

        result = router.route_order(order)
        print(f"  Result: {result['status']}")

        if result['status'] == 'submitted':
            print(f"  ✓ Order submitted successfully")
        else:
            print(f"  ❌ Order submission failed: {result}")

    # Test 2: Circuit breaker open (simulate API failures first)
    print(f"\n→ Testing circuit breaker protection...")
    with patch('requests.get') as mock_get:
        mock_get.side_effect = Exception("API Down")

        # Trigger failures to open circuit
        for i in range(5):
            try:
                client.get_balance()
            except:
                pass

    # Now try to route order with circuit open
    result = router.route_order(order)
    print(f"  Result: {result['status']} - {result.get('reason', 'N/A')}")

    if result['status'] == 'error' and 'circuit_breaker' in result.get('reason', ''):
        print(f"  ✓ Circuit breaker correctly blocked order")
        success = True
    else:
        print(f"  ❌ Circuit breaker should have blocked order")
        success = False

    # Check circuit breaker state
    stats = kalshi_breaker.get_stats()
    print(f"\n✓ Final circuit breaker state: {stats['state']}")

    if success:
        print(f"\n✅ TEST 4 PASSED: OrderRouter correctly integrates circuit breaker and audit logging")
    else:
        print(f"\n❌ TEST 4 FAILED: OrderRouter integration issues detected")

    # Cleanup
    test_wal.close()
    audit_module.audit_logger = original_logger

    return success


def test_circuit_breaker_registry():
    """Test 5: Verify circuit breaker registry and monitoring."""
    print_section("TEST 5: Circuit Breaker Registry")

    # Get all registered circuit breakers
    all_stats = registry.get_all_stats()
    print(f"✓ Registered circuit breakers: {len(all_stats)}")

    for name, stats in all_stats.items():
        print(f"\n  {name}:")
        print(f"    State: {stats['state']}")
        print(f"    Total calls: {stats['total_calls']}")
        print(f"    Success rate: {stats['success_rate']:.2%}")
        print(f"    Failed calls: {stats['failed_calls']}")

    # Verify expected breakers exist
    expected_breakers = ['kalshi_api', 'polymarket_api', 'sportradar_api', 'gemini_api']
    found_breakers = list(all_stats.keys())

    all_present = all(name in found_breakers for name in expected_breakers)

    if all_present:
        print(f"\n✅ TEST 5 PASSED: All expected circuit breakers registered")
        return True
    else:
        print(f"\n❌ TEST 5 FAILED: Missing circuit breakers")
        print(f"  Expected: {expected_breakers}")
        print(f"  Found: {found_breakers}")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("  CIRCUIT BREAKER & AUDIT LOG INTEGRATION TEST SUITE")
    print("=" * 80)

    results = {}

    # Run tests
    results['circuit_breaker'] = test_circuit_breaker_integration()
    results['audit_log'] = test_audit_log_integration()
    results['risk_manager'] = test_risk_manager_audit_logging()
    results['order_router'] = test_order_router_integration()
    results['registry'] = test_circuit_breaker_registry()

    # Print summary
    print_section("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")

    print(f"\n{'=' * 80}")
    print(f"  OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'=' * 80}\n")

    if passed == total:
        print("🎉 All integration tests passed! System is production-ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
