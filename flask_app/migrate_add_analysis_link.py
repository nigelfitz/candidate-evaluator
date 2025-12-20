"""
Migration script to add analysis_id column to transactions table
Run this with: python migrate_add_analysis_link.py
"""
from app import create_app
from database import db
from sqlalchemy import text

app = create_app()

with app.app_context():
    try:
        # Check if column already exists
        result = db.session.execute(text("PRAGMA table_info(transactions)"))
        columns = [row[1] for row in result]
        
        if 'analysis_id' in columns:
            print("✅ analysis_id column already exists in transactions table")
        else:
            # Add analysis_id column
            print("Adding analysis_id column to transactions table...")
            db.session.execute(text("""
                ALTER TABLE transactions 
                ADD COLUMN analysis_id INTEGER 
                REFERENCES analyses(id) ON DELETE SET NULL
            """))
            
            # Create index for performance
            print("Creating index on analysis_id...")
            db.session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_transactions_analysis_id 
                ON transactions(analysis_id)
            """))
            
            db.session.commit()
            print("✅ Successfully added analysis_id column with index")
            print("✅ Foreign key constraint: ON DELETE SET NULL")
            print("✅ Column is nullable (credit purchases don't need analysis_id)")
            
    except Exception as e:
        db.session.rollback()
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
