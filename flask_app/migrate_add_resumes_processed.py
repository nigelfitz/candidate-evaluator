"""
Migration: Add resumes_processed field to Analysis model
Run this once to update the database schema
"""

from database import db, Analysis
from sqlalchemy import text

def migrate():
    """Add resumes_processed column to analyses table"""
    print("🔄 Starting resumes_processed migration...")
    
    try:
        # Check if column already exists
        result = db.session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='analyses' AND column_name='resumes_processed';
        """))
        
        existing_columns = [row[0] for row in result.fetchall()]
        
        if 'resumes_processed' in existing_columns:
            print("✅ Column 'resumes_processed' already exists. Skipping.")
            return True
        
        # Add column
        print("📝 Adding 'resumes_processed' column to analyses table...")
        db.session.execute(text("""
            ALTER TABLE analyses 
            ADD COLUMN resumes_processed INTEGER DEFAULT 0;
        """))
        
        db.session.commit()
        print("✅ resumes_processed migration completed successfully!")
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
