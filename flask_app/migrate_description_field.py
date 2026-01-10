"""
Migration: Extend Transaction.description field from VARCHAR(255) to TEXT
Run this script to update the production database on Railway
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from database import db
from sqlalchemy import text

def migrate():
    app = create_app()
    
    with app.app_context():
        try:
            # Alter the column type from VARCHAR(255) to TEXT
            db.session.execute(text(
                "ALTER TABLE transactions ALTER COLUMN description TYPE TEXT;"
            ))
            db.session.commit()
            print("✅ Successfully migrated description field to TEXT")
            return True
        except Exception as e:
            print(f"❌ Migration failed: {str(e)}")
            db.session.rollback()
            return False

if __name__ == '__main__':
    success = migrate()
    sys.exit(0 if success else 1)
