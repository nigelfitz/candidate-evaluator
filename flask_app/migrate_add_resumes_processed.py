"""
Migration: Add resumes_processed field to Analysis model
Run this once to update the database schema
"""

from database import db, Analysis

def migrate_add_resumes_processed():
    """Add resumes_processed column to analyses table"""
    try:
        # Check if column already exists
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('analyses')]
        
        if 'resumes_processed' in columns:
            print("✅ Column 'resumes_processed' already exists")
            return
        
        # Add column
        with db.engine.connect() as conn:
            conn.execute(db.text("""
                ALTER TABLE analyses 
                ADD COLUMN resumes_processed INTEGER DEFAULT 0
            """))
            conn.commit()
        
        print("✅ Successfully added 'resumes_processed' column to analyses table")
        
    except Exception as e:
        print(f"ERROR: Failed to add resumes_processed column: {str(e)}")
        raise

if __name__ == "__main__":
    migrate_add_resumes_processed()
