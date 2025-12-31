"""
Migration: Add error tracking fields to Analysis model
Run this once to update the database schema
"""

from database import db, Analysis
from sqlalchemy import text, inspect

def migrate():
    """Add error_message and failed_at columns to analyses table"""
    print("🔄 Starting error tracking migration...")
    
    try:
        # Check if columns already exist
        inspector = inspect(db.engine)
        existing_columns = [col['name'] for col in inspector.get_columns('analyses')]
        
        columns_added = []
        
        if 'error_message' not in existing_columns:
            print("📝 Adding 'error_message' column to analyses table...")
            db.session.execute(text("""
                ALTER TABLE analyses 
                ADD COLUMN error_message TEXT;
            """))
            columns_added.append('error_message')
        else:
            print("✅ Column 'error_message' already exists.")
        
        if 'failed_at' not in existing_columns:
            print("📝 Adding 'failed_at' column to analyses table...")
            db.session.execute(text("""
                ALTER TABLE analyses 
                ADD COLUMN failed_at DATETIME;
            """))
            columns_added.append('failed_at')
        else:
            print("✅ Column 'failed_at' already exists.")
        
        if columns_added:
            db.session.commit()
            print(f"✅ Error tracking migration completed! Added: {', '.join(columns_added)}")
        else:
            print("✅ All columns already exist. No migration needed.")
        
        return True
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    from app import create_app
    app = create_app()
    with app.app_context():
        migrate()
