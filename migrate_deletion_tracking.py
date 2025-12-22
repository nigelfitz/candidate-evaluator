"""
Add analysis_deleted_at column to transactions table
This tracks when an associated analysis was deleted (for audit trail and user accountability)
"""

import os
from sqlalchemy import create_engine, text

def migrate_database():
    """Add analysis_deleted_at column to transactions table"""
    
    # Get database URL from environment or use local SQLite
    database_url = os.environ.get('DATABASE_URL', 'sqlite:///candidate_evaluator.db')
    
    # Fix Railway PostgreSQL URL format
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    print(f"Connecting to database...")
    engine = create_engine(database_url)
    
    # Perform migration based on database type
    if database_url.startswith('postgresql://'):
        print("Migrating PostgreSQL database...")
        with engine.connect() as conn:
            # Add column
            conn.execute(text("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS analysis_deleted_at TIMESTAMP;"))
            # Remove deleted_at from analyses if it exists
            conn.execute(text("ALTER TABLE analyses DROP COLUMN IF EXISTS deleted_at;"))
            conn.commit()
        print("✅ Migration complete")
        print("   - Added analysis_deleted_at to transactions")
        print("   - Removed deleted_at from analyses")
    
    elif database_url.startswith('sqlite://'):
        print("⚠️  SQLite detected")
        print("Recommended: Delete candidate_evaluator.db and restart Flask to recreate schema")
        print("\nAlternative manual steps:")
        print("  1. Stop Flask")
        print("  2. Delete candidate_evaluator.db")
        print("  3. Restart Flask (schema will auto-create)")
        return
    
    print("\n✅ Migration successful!")

if __name__ == '__main__':
    migrate_database()
