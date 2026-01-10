"""
Migration script to add stripe_customer_id column to users table
Run this once to update your database schema
"""
from dotenv import load_dotenv
import os

# Load environment variables first
load_dotenv()

from app import create_app
from database import db
from sqlalchemy import text

def migrate():
    app = create_app()
    with app.app_context():
        try:
            # Check if column already exists
            result = db.session.execute(text(
                "SELECT COUNT(*) FROM pragma_table_info('users') WHERE name='stripe_customer_id'"
            ))
            exists = result.scalar() > 0
            
            if exists:
                print("✅ Column 'stripe_customer_id' already exists!")
                return
            
            # Add the column
            print("Adding stripe_customer_id column to users table...")
            db.session.execute(text(
                "ALTER TABLE users ADD COLUMN stripe_customer_id VARCHAR(255)"
            ))
            db.session.commit()
            print("✅ Successfully added stripe_customer_id column!")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            db.session.rollback()
            raise

if __name__ == '__main__':
    migrate()
