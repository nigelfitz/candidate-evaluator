"""
Migration: Add is_suspended and suspension_reason fields to User table

This migration adds account suspension functionality to allow admins to
block problematic users from accessing the service.

To run this migration on Railway:
1. It will run automatically on next deployment via start.sh
2. Or run manually via Railway CLI: railway run python migrate_add_suspension.py
"""

from database import db, User
from sqlalchemy import text

def migrate():
    """Add suspension fields to users table"""
    print("🔄 Starting suspension fields migration...")
    
    try:
        # Check if columns already exist
        result = db.session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='users' AND column_name IN ('is_suspended', 'suspension_reason');
        """))
        
        existing_columns = [row[0] for row in result.fetchall()]
        
        if 'is_suspended' in existing_columns and 'suspension_reason' in existing_columns:
            print("✅ Suspension columns already exist. Skipping.")
            return True
        
        # Add is_suspended column
        if 'is_suspended' not in existing_columns:
            print("📝 Adding 'is_suspended' column to users table...")
            db.session.execute(text("""
                ALTER TABLE users 
                ADD COLUMN is_suspended BOOLEAN DEFAULT FALSE NOT NULL;
            """))
        
        # Add suspension_reason column
        if 'suspension_reason' not in existing_columns:
            print("📝 Adding 'suspension_reason' column to users table...")
            db.session.execute(text("""
                ALTER TABLE users 
                ADD COLUMN suspension_reason VARCHAR(500);
            """))
        
        db.session.commit()
        print("✅ Suspension fields migration completed successfully!")
        return True
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == '__main__':
    from app import create_app
    app = create_app()
    with app.app_context():
        migrate()
