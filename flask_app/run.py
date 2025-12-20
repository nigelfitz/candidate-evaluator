"""
Initialize the database and run the Flask app
"""
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from app import create_app
from database import db

if __name__ == '__main__':
    app = create_app()
    
    with app.app_context():
        # Create all database tables
        db.create_all()
        print("✅ Database tables created successfully!")
        print("🚀 Starting Flask development server...")
        print("📱 Open http://127.0.0.1:5000 in your browser")
        print("💳 Stripe is in TEST MODE - use test cards")
        print("\n   Test Card: 4242 4242 4242 4242")
        print("   Expiry: Any future date")
        print("   CVC: Any 3 digits\n")
    
    # Use use_reloader=False to prevent crashes when loading transformers/sentence-transformers
    # (The reloader detects library imports as file changes and restarts mid-request)
    app.run(debug=True, port=5000, use_reloader=False)
