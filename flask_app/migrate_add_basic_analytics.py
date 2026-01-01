"""
Quick Analytics Migration - Add Essential Tracking Fields
==========================================================
Adds 4 strategic fields to capture the most valuable analytics data
without requiring extensive code changes.

Fields added to Analysis table:
1. completed_at - When job finished (enables time-based analysis)
2. processing_duration_seconds - How long it took (performance tracking)
3. exceeded_resume_limit - Did they hit the 200+ warning? (behavior pattern)
4. user_chose_override - Did they click "Try All Anyway"? (risk indicator)

Run this after backing up your database.
"""

import sys
import os

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from database import db
from sqlalchemy import text

# Initialize minimal Flask app for migration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///candidate_evaluator.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def run_migration():
    """Execute migration"""
    with app.app_context():
        print("\n" + "="*70)
        print("📊 QUICK ANALYTICS - Adding Essential Tracking Fields")
        print("="*70 + "\n")
        
        try:
            print("📈 Adding 4 essential fields to Analysis table...")
            
            fields = [
                ("completed_at", "TIMESTAMP"),
                ("processing_duration_seconds", "INTEGER"),
                ("exceeded_resume_limit", "BOOLEAN DEFAULT FALSE"),
                ("user_chose_override", "BOOLEAN DEFAULT FALSE"),
            ]
            
            for field_name, field_type in fields:
                try:
                    db.session.execute(text(
                        f"ALTER TABLE analyses ADD COLUMN {field_name} {field_type};"
                    ))
                    print(f"  ✅ Added: {field_name}")
                except Exception as e:
                    if "already exists" in str(e) or "duplicate column" in str(e):
                        print(f"  ⏭️  Skip: {field_name} (already exists)")
                    else:
                        print(f"  ❌ Error: {field_name}: {e}")
            
            db.session.commit()
            
            print("\n" + "="*70)
            print("✅ MIGRATION COMPLETE!")
            print("="*70)
            print("\n📊 Basic analytics tracking is now active!")
            print("📈 Visit /admin/analytics to view insights\n")
            
        except Exception as e:
            print(f"\n❌ MIGRATION FAILED: {e}")
            print("Rolling back changes...")
            db.session.rollback()
            sys.exit(1)


if __name__ == '__main__':
    print("\n⚠️  This will add 4 fields to your analyses table.")
    print("Make sure you have a backup.\n")
    
    response = input("Continue? (yes/no): ").strip().lower()
    if response == 'yes':
        run_migration()
    else:
        print("\n❌ Cancelled")
        sys.exit(0)
