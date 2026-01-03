#!/usr/bin/env python3
"""
Migration: Add jd_bytes column to analyses table
This allows storing PDF bytes in job history for preview
"""

import sys
from app import create_app
from database import db
from sqlalchemy import text

app = create_app()

def run_migration():
    """Add jd_bytes column to analyses table"""
    with app.app_context():
        print("\n" + "="*70)
        print("📄 Adding jd_bytes Column to Analysis Table")
        print("="*70 + "\n")
        
        try:
            print("📊 Adding jd_bytes (LargeBinary) column to analyses table...")
            
            try:
                db.session.execute(text(
                    "ALTER TABLE analyses ADD COLUMN jd_bytes BYTEA;"
                ))
                db.session.commit()
                print("  ✅ Added: jd_bytes column")
            except Exception as e:
                if "already exists" in str(e) or "duplicate column" in str(e).lower():
                    print("  ⏭️  Skip: jd_bytes (already exists)")
                else:
                    raise e
            
            print("\n" + "="*70)
            print("✅ Migration Complete!")
            print("="*70)
            print("\n📋 Summary:")
            print("  - Added jd_bytes column to store PDF bytes in job history")
            print("  - Future analyses will preserve PDF files when loaded from history")
            print("  - Existing analyses without PDF bytes will continue to show text only")
            print("\n")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Migration failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == '__main__':
    print("\n⚠️  This will add the jd_bytes column to your analyses table.")
    print("This is safe and reversible.\n")
    
    response = input("Continue? (yes/no): ").strip().lower()
    if response == 'yes':
        run_migration()
    else:
        print("\n❌ Cancelled")
        sys.exit(0)
