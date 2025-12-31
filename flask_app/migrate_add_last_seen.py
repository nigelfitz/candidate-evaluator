"""
Migration script to add last_seen column to users table
Run this once to update the database schema
"""

from app import create_app
from database import db
from sqlalchemy import inspect, text

def migrate():
    app = create_app()
    
    with app.app_context():
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('users')]
        
        print("🔍 Checking users table schema...")
        
        # Check and add last_seen column
        if 'last_seen' not in columns:
            print("  ➕ Adding last_seen column...")
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users ADD COLUMN last_seen TIMESTAMP'))
                conn.commit()
            print("  ✅ Added last_seen column")
        else:
            print("  ℹ️  last_seen column already exists")
        
        print("\n✅ Last seen tracking migration completed!")
        print("   Users' online status will now be tracked based on activity within the last 5 minutes.")

if __name__ == '__main__':
    migrate()
