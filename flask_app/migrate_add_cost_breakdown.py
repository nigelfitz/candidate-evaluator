"""
Migration: Add ranker_cost_usd and insight_cost_usd fields to Analysis table

Adds fields for tracking cost breakdown between:
- ranker_cost_usd: Cost for ranking/scoring calls (3 per resume)
- insight_cost_usd: Cost for deep insight generation (top 5 candidates)

This allows proper analysis of cost drivers in the system.

Run this script once to add the new columns.
"""

import sys
from app import create_app
from database import db

def run_migration():
    """Add cost breakdown columns to analyses table"""
    app = create_app()
    
    with app.app_context():
        try:
            print("🔄 Starting cost breakdown migration...")
            
            # Check if columns already exist
            inspector = db.inspect(db.engine)
            existing_columns = [col['name'] for col in inspector.get_columns('analyses')]
            
            migrations_needed = []
            
            if 'ranker_cost_usd' not in existing_columns:
                migrations_needed.append('ranker_cost_usd')
            
            if 'insight_cost_usd' not in existing_columns:
                migrations_needed.append('insight_cost_usd')
            
            if not migrations_needed:
                print("✅ Cost breakdown columns already exist!")
                return True
            
            # Add missing columns
            for col_name in migrations_needed:
                sql = f"ALTER TABLE analyses ADD COLUMN {col_name} NUMERIC(10, 4)"
                print(f"   Adding column: {col_name}...")
                db.session.execute(db.text(sql))
            
            db.session.commit()
            
            print(f"✅ Migration complete! Added {len(migrations_needed)} columns")
            print(f"   ranker_cost_usd: Cost for ranking (3 calls per resume)")
            print(f"   insight_cost_usd: Cost for insights (top 5 candidates)")
            
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
