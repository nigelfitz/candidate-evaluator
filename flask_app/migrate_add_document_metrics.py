"""
Document Size Metrics Migration
================================
Adds character count tracking to enable data volume analysis.

Fields added to Analysis table:
1. jd_character_count - Length of job description
2. avg_resume_character_count - Average resume length
3. min_resume_character_count - Shortest resume
4. max_resume_character_count - Longest resume

These enable insights like:
- Processing time correlation with document size
- Bell curve distribution of resume lengths
- Identifying outlier documents causing issues
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
        print("📄 DOCUMENT METRICS - Adding Character Count Tracking")
        print("="*70 + "\n")
        
        try:
            print("📊 Adding 4 document size fields to Analysis table...")
            
            fields = [
                ("jd_character_count", "INTEGER"),
                ("avg_resume_character_count", "INTEGER"),
                ("min_resume_character_count", "INTEGER"),
                ("max_resume_character_count", "INTEGER"),
            ]
            
            for field_name, field_type in fields:
                try:
                    sql = f"ALTER TABLE analyses ADD COLUMN {field_name} {field_type}"
                    db.session.execute(text(sql))
                    db.session.commit()
                    print(f"  ✅ Added: {field_name}")
                except Exception as e:
                    if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                        print(f"  ⏭️  Skipped: {field_name} (already exists)")
                    else:
                        raise
            
            print("\n" + "="*70)
            print("✅ MIGRATION COMPLETE!")
            print("="*70 + "\n")
            print("📊 Document size tracking is now active!")
            print("📈 Bell curve analysis available in analytics dashboard")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Migration failed: {e}")
            sys.exit(1)

if __name__ == '__main__':
    print("\n⚠️  This will add 4 fields to your analyses table.")
    print("Make sure you have a backup.")
    response = input("\nContinue? (yes/no): ")
    
    if response.lower() == 'yes':
        run_migration()
    else:
        print("❌ Migration cancelled.")
