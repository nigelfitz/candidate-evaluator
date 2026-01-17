"""
Migration: Add openai_cost_usd field to Analysis table

Adds field for tracking actual OpenAI API costs based on token usage.
This is separate from cost_usd which represents what the customer paid.

Run this script once to add the new column.
"""

import sys
from app import create_app
from database import db

def run_migration():
    """Add openai_cost_usd column to analyses table"""
    app = create_app()
    
    with app.app_context():
        try:
            print("🔄 Starting OpenAI cost tracking migration...")
            
            # Check if column already exists
            inspector = db.inspect(db.engine)
            existing_columns = [col['name'] for col in inspector.get_columns('analyses')]
            
            if 'openai_cost_usd' in existing_columns:
                print("✅ openai_cost_usd column already exists!")
                return True
            
            # Add the column
            sql = "ALTER TABLE analyses ADD COLUMN openai_cost_usd NUMERIC(10, 4)"
            print(f"   Adding column: openai_cost_usd...")
            db.session.execute(db.text(sql))
            db.session.commit()
            
            print(f"✅ Migration complete! Added openai_cost_usd column")
            print(f"   This tracks actual OpenAI API costs from token usage")
            print(f"   Format: NUMERIC(10, 4) - up to $999,999.9999")
            
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
