"""
Migration: Add performance tracking fields to Analysis table

Adds fields for monitoring system health:
- retry_count: Number of API retries during analysis
- json_fallback_count: Times GPT-4o-mini was used instead of GPT-4o (JSON parsing failures)
- cost_multiplier: Actual cost / expected cost ratio
- api_calls_made: Total OpenAI API calls for this analysis
- avg_api_response_ms: Average API response time in milliseconds

Run this script once to add the new columns.
"""

import sys
from app import create_app
from database import db

def run_migration():
    """Add performance tracking columns to analyses table"""
    app = create_app()
    
    with app.app_context():
        try:
            print("🔄 Starting performance tracking migration...")
            
            # Check if columns already exist
            inspector = db.inspect(db.engine)
            existing_columns = [col['name'] for col in inspector.get_columns('analyses')]
            
            migrations_needed = []
            
            # Define new columns with their SQL definitions
            new_columns = {
                'retry_count': 'INTEGER DEFAULT 0',
                'json_fallback_count': 'INTEGER DEFAULT 0',
                'cost_multiplier': 'FLOAT',
                'api_calls_made': 'INTEGER',
                'avg_api_response_ms': 'INTEGER'
            }
            
            # Check which columns need to be added
            for col_name, col_def in new_columns.items():
                if col_name not in existing_columns:
                    migrations_needed.append((col_name, col_def))
            
            if not migrations_needed:
                print("✅ All performance tracking columns already exist!")
                return True
            
            # Add missing columns
            for col_name, col_def in migrations_needed:
                sql = f"ALTER TABLE analyses ADD COLUMN {col_name} {col_def}"
                print(f"   Adding column: {col_name}...")
                db.session.execute(db.text(sql))
            
            db.session.commit()
            
            print(f"✅ Migration complete! Added {len(migrations_needed)} columns:")
            for col_name, _ in migrations_needed:
                print(f"   - {col_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            db.session.rollback()
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    success = run_migration()
    sys.exit(0 if success else 1)
