"""Test if Stripe is configured"""
from dotenv import load_dotenv
import os

load_dotenv()

stripe_secret = os.environ.get('STRIPE_SECRET_KEY')
stripe_pub = os.environ.get('STRIPE_PUBLISHABLE_KEY')

print("=" * 60)
print("STRIPE CONFIGURATION CHECK")
print("=" * 60)

if stripe_secret:
    print(f"✅ STRIPE_SECRET_KEY: {stripe_secret[:20]}...")
else:
    print("❌ STRIPE_SECRET_KEY: NOT SET")

if stripe_pub:
    print(f"✅ STRIPE_PUBLISHABLE_KEY: {stripe_pub[:20]}...")
else:
    print("❌ STRIPE_PUBLISHABLE_KEY: NOT SET")

print("=" * 60)

if stripe_secret and stripe_pub:
    print("✅ Stripe is FULLY CONFIGURED")
else:
    print("❌ Stripe is NOT configured")
