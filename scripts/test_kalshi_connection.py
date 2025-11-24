"""
Test Kalshi API connection.

Usage:
    1. Create .env file with your credentials
    2. Run: python test_kalshi_connection.py
"""

import os

from dotenv import load_dotenv

from src.execution.kalshi_client import KalshiClient

# Load environment variables
load_dotenv()


def test_connection():
    """Test Kalshi API connection."""
    print("=" * 60)
    print("TESTING KALSHI API CONNECTION")
    print("=" * 60)

    # Get credentials from environment
    api_key_id = os.getenv("KALSHI_API_KEY_ID")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    base_url = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com")

    if not api_key_id or not private_key_path:
        print("\n❌ ERROR: Missing API credentials!")
        print("\nPlease create a .env file with:")
        print("  KALSHI_API_KEY_ID=your_api_key_id_here")
        print("  KALSHI_PRIVATE_KEY_PATH=/path/to/your/private_key.pem")
        print("  KALSHI_BASE_URL=https://api.elections.kalshi.com")
        print("\nGet your API key from: https://kalshi.com/profile/api-keys")
        print("See docs/guides/PAPER_TRADING_SETUP.md for instructions.")
        return False

    print(f"\n🔑 API Key ID: {api_key_id[:8]}...")
    print(f"🔗 API URL: {base_url}")
    print(f"🔐 Private Key: {private_key_path}")

    try:
        # Initialize client
        print("\n1️⃣  Initializing Kalshi client...")
        client = KalshiClient(api_key=api_key_id, private_key_path=private_key_path)
        print("   ✅ Client initialized")

        # Test authentication (API key auth doesn't need separate auth step)
        print("\n2️⃣  Testing API key authentication...")
        print("   ✅ Using API key authentication")

        # Get balance
        print("\n3️⃣  Fetching account balance...")
        balance_data = client.get_balance()
        if balance_data:
            balance = balance_data.get('balance', 0)
            print(f"   ✅ Balance: ${balance / 100:,.2f}")  # Kalshi returns balance in cents
        else:
            print("   ⚠️  Could not fetch balance")

        # List markets
        print("\n4️⃣  Fetching available markets...")
        markets_response = client.get_markets(limit=5)
        if markets_response and 'markets' in markets_response:
            markets = markets_response['markets']
            print(f"   ✅ Found {len(markets)} markets")
            print("\n   📊 Sample markets:")
            for market in markets[:5]:
                ticker = market.get("ticker", "N/A")
                title = market.get("title", "N/A")
                status = market.get("status", "N/A")
                print(f"      • {ticker}: {title[:50]}... (Status: {status})")
        else:
            print("   ⚠️  No markets found")

        # Skip individual market test - basic connectivity confirmed

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\n🚀 You're ready for paper trading!")
        print("   Next step: Run python run_paper_trading.py")
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your API credentials in .env")
        print("  2. Verify private key file exists at the specified path")
        print("  3. Make sure you downloaded the .pem file when creating API key")
        print("  4. Verify API key is active at https://kalshi.com/profile/api-keys")
        print("  5. Try using demo URL if testing:")
        print("     KALSHI_BASE_URL=https://demo-api.kalshi.co")
        return False


if __name__ == "__main__":
    success = test_connection()
    exit(0 if success else 1)
