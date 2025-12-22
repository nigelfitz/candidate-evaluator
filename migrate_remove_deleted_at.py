"""
Database Migration Script: Remove deleted_at Column
Run this AFTER deploying the code changes to ensure database schema matches the model.
"""

import os
from sqlalchemy import create_engine, inspect, text

def migrate_database():
    """Remove deleted_at column from analyses table"""
    
    # Get database URL from environment or use local SQLite
    database_url = os.environ.get('DATABASE_URL', 'sqlite:///candidate_evaluator.db')
    
    # Fix Railway PostgreSQL URL format
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    print(f"Connecting to database...")
    engine = create_engine(database_url)
    
    # Check if column exists
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns('analyses')]
    
    if 'deleted_at' not in columns:
        print("✅ Migration already applied - deleted_at column does not exist")
        return
    
    print(f"Found deleted_at column in analyses table")
    
    # Perform migration based on database type
    if database_url.startswith('postgresql://'):
        print("Migrating PostgreSQL database...")
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE analyses DROP COLUMN IF EXISTS deleted_at;"))
            conn.commit()
        print("✅ Migration complete - deleted_at column removed from PostgreSQL")
    
    elif database_url.startswith('sqlite://'):
        print("⚠️  SQLite detected - cannot easily drop columns")
        print("Recommended: Delete candidate_evaluator.db and restart Flask to recreate schema")
        print("Alternative: Run this manually:")
        print("  1. Stop Flask")
        print("  2. Delete candidate_evaluator.db")
        print("  3. Restart Flask (schema will auto-create)")
        return
    
    print("\n✅ Migration successful!")
    print("Verify by checking the analyses table structure")

if __name__ == '__main__':
    migrate_database()
