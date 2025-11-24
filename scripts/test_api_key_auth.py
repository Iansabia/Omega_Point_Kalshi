#!/usr/bin/env python3
"""
Test script for Kalshi API key authentication.

Before running:
1. Get your API key from: https://kalshi.com/profile/api-keys
2. Download your private key file (.pem)
3. Update .env with:
   KALSHI_API_KEY_ID=your_api_key_id
   KALSHI_PRIVATE_KEY_PATH=/path/to/your/private_key.pem
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from execution.kalshi_client import KalshiClient


def test_api_key_auth():
    """Test API key authentication and basic API calls."""

    print("=" * 60)
    print("Kalshi API Key Authentication Test")
    print("=" * 60)

    # Initialize client
    print("\n1. Initializing Kalshi client...")
    client = KalshiClient()

    # Check authentication method
    if client.private_key and client.api_key_id:
        print("   ✓ Using API key authentication")
        print(f"   Key ID: {client.api_key_id[:12]}...")
    elif client.token:
        print("   ⚠ Using email/password authentication (legacy)")
    else:
        print("   ✗ No authentication configured!")
        print("\nPlease update your .env file with:")
        print("   KALSHI_API_KEY_ID=your_api_key_id")
        print("   KALSHI_PRIVATE_KEY_PATH=/path/to/your/private_key.pem")
        return False

    # Test balance endpoint
    print("\n2. Testing balance endpoint...")
    try:
        balance = client.get_balance()
        if "balance" in balance or "error" not in balance:
            print("   ✓ Balance request successful")
            if "balance" in balance:
                print(f"   Balance: ${balance.get('balance', 0) / 100:.2f}")
        else:
            print(f"   ✗ Balance request failed: {balance}")
            return False
    except Exception as e:
        print(f"   ✗ Balance request error: {e}")
        return False

    # Test markets endpoint
    print("\n3. Testing markets endpoint...")
    try:
        markets = client.get_markets(limit=5)
        market_list = markets.get("markets", [])
        if market_list:
            print(f"   ✓ Markets request successful")
            print(f"   Found {len(market_list)} markets")
            if market_list:
                print(f"   Sample market: {market_list[0].get('ticker', 'N/A')}")
        else:
            print(f"   ⚠ No markets returned (might be expected)")
    except Exception as e:
        print(f"   ✗ Markets request error: {e}")
        return False

    # Test events endpoint
    print("\n4. Testing events endpoint...")
    try:
        events = client.get_events(limit=5)
        event_list = events.get("events", [])
        if event_list:
            print(f"   ✓ Events request successful")
            print(f"   Found {len(event_list)} events")
            if event_list:
                print(f"   Sample event: {event_list[0].get('title', 'N/A')}")
        else:
            print(f"   ⚠ No events returned (might be expected)")
    except Exception as e:
        print(f"   ✗ Events request error: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed! API key authentication is working.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_api_key_auth()
    sys.exit(0 if success else 1)
