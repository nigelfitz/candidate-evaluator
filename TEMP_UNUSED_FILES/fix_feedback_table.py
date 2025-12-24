"""
Fix Feedback table creation issue - drop and recreate if sequence conflict exists
Run this once to fix the duplicate sequence error
"""
import os
import sys
from app import create_app
from database import db

def fix_feedback_table():
    """Drop and recreate feedback table to fix sequence conflict"""
    app = create_app()
    
    with app.app_context():
        try:
            # Check if we're using PostgreSQL
            if 'postgresql' in str(db.engine.url):
                print("📊 Fixing Feedback table in PostgreSQL...")
                
                # Drop the table if it exists (this will also drop the sequence)
                db.session.execute(db.text("DROP TABLE IF EXISTS feedback CASCADE;"))
                db.session.commit()
                print("✅ Dropped existing feedback table (if any)")
                
                # Drop the sequence if it exists
                db.session.execute(db.text("DROP SEQUENCE IF EXISTS feedback_id_seq CASCADE;"))
                db.session.commit()
                print("✅ Dropped existing sequence (if any)")
                
                # Now create all tables (this will create feedback table fresh)
                db.create_all()
                print("✅ Created feedback table successfully")
                
                # Verify the table exists
                result = db.session.execute(db.text(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'feedback'"
                ))
                count = result.scalar()
                
                if count > 0:
                    print("✅ Verification successful - feedback table exists")
                    print("\n🎉 Database fix completed successfully!")
                else:
                    print("❌ ERROR: Feedback table was not created")
                    return False
                    
            else:
                print("📊 Using SQLite - no sequence conflict possible")
                db.create_all()
                print("✅ Tables created successfully")
                
            return True
            
        except Exception as e:
            print(f"❌ ERROR during database fix: {str(e)}")
            db.session.rollback()
            return False

if __name__ == '__main__':
    print("🔧 Starting Feedback table fix...")
    print("=" * 60)
    
    success = fix_feedback_table()
    
    print("=" * 60)
    if success:
        print("✅ Fix completed successfully!")
        sys.exit(0)
    else:
        print("❌ Fix failed - check errors above")
        sys.exit(1)
