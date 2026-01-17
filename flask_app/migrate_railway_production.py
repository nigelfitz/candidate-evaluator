"""
EMERGENCY MIGRATION FOR RAILWAY PRODUCTION

This script adds all missing database columns needed for the new cost breakdown
and performance tracking features.

SAFE TO RUN MULTIPLE TIMES - checks for existing columns first.

Columns added:
- openai_cost_usd: Actual OpenAI API costs
- ranker_cost_usd: Cost for ranking/scoring calls  
- insight_cost_usd: Cost for insight generation
- retry_count: Number of API retries
- json_fallback_count: JSON parsing fallback count
- api_calls_made: Total API calls
- avg_api_response_ms: Average response time

Usage:
  python migrate_railway_production.py
"""

import sys
import os
from app import create_app
from database import db

def run_production_migration():
    """Add all missing columns to analyses table"""
    app = create_app()
    
    with app.app_context():
        try:
            print("=" * 60)
            print("🚀 RAILWAY PRODUCTION MIGRATION")
            print("=" * 60)
            print()
            
            # Check existing columns
            inspector = db.inspect(db.engine)
            existing_columns = [col['name'] for col in inspector.get_columns('analyses')]
            
            print(f"📋 Current analyses table has {len(existing_columns)} columns")
            print()
            
            # Define all new columns
            new_columns = {
                'openai_cost_usd': {
                    'sql': 'NUMERIC(10, 4)',
                    'description': 'Actual OpenAI API costs from token usage'
                },
                'ranker_cost_usd': {
                    'sql': 'NUMERIC(10, 4)',
                    'description': 'Cost for ranking/scoring calls (3 per resume)'
                },
                'insight_cost_usd': {
                    'sql': 'NUMERIC(10, 4)',
                    'description': 'Cost for deep insight generation (top 5)'
                },
                'retry_count': {
                    'sql': 'INTEGER DEFAULT 0',
                    'description': 'Number of API retries during analysis'
                },
                'json_fallback_count': {
                    'sql': 'INTEGER DEFAULT 0',
                    'description': 'Times fallback model was used'
                },
                'api_calls_made': {
                    'sql': 'INTEGER',
                    'description': 'Total OpenAI API calls made'
                },
                'avg_api_response_ms': {
                    'sql': 'INTEGER',
                    'description': 'Average API response time in milliseconds'
                }
            }
            
            # Check which need to be added
            to_add = []
            already_exists = []
            
            for col_name, col_info in new_columns.items():
                if col_name in existing_columns:
                    already_exists.append(col_name)
                else:
                    to_add.append((col_name, col_info))
            
            # Report status
            if already_exists:
                print(f"✅ Already have {len(already_exists)} columns:")
                for col in already_exists:
                    print(f"   • {col}")
                print()
            
            if not to_add:
                print("✨ ALL COLUMNS ALREADY EXIST - Migration not needed!")
                print()
                return True
            
            # Add missing columns
            print(f"🔧 Adding {len(to_add)} missing columns:")
            print()
            
            for col_name, col_info in to_add:
                sql = f"ALTER TABLE analyses ADD COLUMN {col_name} {col_info['sql']}"
                print(f"   Adding: {col_name}")
                print(f"   Type: {col_info['sql']}")
                print(f"   Purpose: {col_info['description']}")
                print(f"   SQL: {sql}")
                print()
                
                db.session.execute(db.text(sql))
            
            # Commit all changes
            db.session.commit()
            
            print("=" * 60)
            print(f"✅ SUCCESS! Added {len(to_add)} columns to analyses table")
            print("=" * 60)
            print()
            print("📊 Migration Summary:")
            print(f"   • Existing columns: {len(existing_columns)}")
            print(f"   • Already had: {len(already_exists)}")
            print(f"   • Newly added: {len(to_add)}")
            print(f"   • Total columns now: {len(existing_columns) + len(to_add)}")
            print()
            print("🎉 Database is now up to date!")
            print()
            
            return True
            
        except Exception as e:
            print("=" * 60)
            print("❌ MIGRATION FAILED")
            print("=" * 60)
            print()
            print(f"Error: {e}")
            print()
            db.session.rollback()
            
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            print()
            
            return False

if __name__ == '__main__':
    print()
    print("Starting migration...")
    print()
    
    success = run_production_migration()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
