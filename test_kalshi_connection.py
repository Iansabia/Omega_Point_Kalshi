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
    print("="*60)
    print("TESTING KALSHI API CONNECTION")
    print("="*60)

    # Get credentials from environment
    email = os.getenv('KALSHI_EMAIL')
    password = os.getenv('KALSHI_PASSWORD')
    base_url = os.getenv('KALSHI_BASE_URL', 'https://demo-api.kalshi.co')

    if not email or not password:
        print("\n❌ ERROR: Missing credentials!")
        print("\nPlease create a .env file with:")
        print("  KALSHI_EMAIL=your_email@example.com")
        print("  KALSHI_PASSWORD=your_password")
        print("  KALSHI_BASE_URL=https://demo-api.kalshi.co")
        print("\nSee PAPER_TRADING_SETUP.md for instructions.")
        return False

    print(f"\n📧 Email: {email}")
    print(f"🔗 API URL: {base_url}")
    print(f"🔐 Password: {'*' * len(password)}")

    try:
        # Initialize client
        print("\n1️⃣  Initializing Kalshi client...")
        client = KalshiClient(
            email=email,
            password=password
        )
        print("   ✅ Client initialized")

        # Test authentication
        print("\n2️⃣  Testing authentication...")
        if client.authenticate():
            print("   ✅ Authentication successful!")
        else:
            print("   ❌ Authentication failed")
            return False

        # Get balance
        print("\n3️⃣  Fetching account balance...")
        balance = client.get_balance()
        if balance is not None:
            print(f"   ✅ Balance: ${balance:,.2f}")
        else:
            print("   ⚠️  Could not fetch balance")

        # List markets
        print("\n4️⃣  Fetching available markets...")
        markets = client.get_markets(limit=5)
        if markets:
            print(f"   ✅ Found {len(markets)} markets")
            print("\n   📊 Sample markets:")
            for market in markets[:5]:
                ticker = market.get('ticker', 'N/A')
                title = market.get('title', 'N/A')
                status = market.get('status', 'N/A')
                print(f"      • {ticker}: {title[:50]}... (Status: {status})")
        else:
            print("   ⚠️  No markets found")

        # Test getting market data
        if markets:
            print("\n5️⃣  Testing market data fetch...")
            sample_ticker = markets[0].get('ticker')
            market_data = client.get_market(sample_ticker)
            if market_data:
                print(f"   ✅ Successfully fetched data for {sample_ticker}")
                print(f"      Yes Price: {market_data.get('yes_bid', 'N/A')} - {market_data.get('yes_ask', 'N/A')}")
                print(f"      No Price: {market_data.get('no_bid', 'N/A')} - {market_data.get('no_ask', 'N/A')}")
            else:
                print(f"   ⚠️  Could not fetch market data for {sample_ticker}")

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\n🚀 You're ready for paper trading!")
        print("   Next step: Run python run_paper_trading.py")
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your credentials in .env")
        print("  2. Verify you can log in at https://demo.kalshi.com")
        print("  3. Make sure API access is enabled in your account")
        print("  4. Try using production URL if demo doesn't work:")
        print("     KALSHI_BASE_URL=https://trading-api.kalshi.com")
        return False


if __name__ == "__main__":
    success = test_connection()
    exit(0 if success else 1)
