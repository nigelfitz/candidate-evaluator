"""
Emergency Migration Runner
Run this via: railway run python flask_app/run_migrations.py
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from database import db
from sqlalchemy import text

# Initialize Flask app
app = Flask(__name__)

# Use Railway's DATABASE_URL environment variable
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def run_migrations():
    """Run both analytics migrations"""
    with app.app_context():
        print("\n" + "="*70)
        print("🚀 RUNNING ANALYTICS MIGRATIONS ON PRODUCTION")
        print("="*70 + "\n")
        
        # Migration 1: Basic Analytics
        print("📊 Migration 1/2: Basic Analytics Fields...")
        basic_fields = [
            ("completed_at", "TIMESTAMP"),
            ("processing_duration_seconds", "INTEGER"),
            ("exceeded_resume_limit", "BOOLEAN DEFAULT FALSE"),
            ("user_chose_override", "BOOLEAN DEFAULT FALSE"),
        ]
        
        for field_name, field_type in basic_fields:
            try:
                sql = f"ALTER TABLE analyses ADD COLUMN {field_name} {field_type}"
                db.session.execute(text(sql))
                db.session.commit()
                print(f"  ✅ Added: {field_name}")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                    print(f"  ⏭️  Skipped: {field_name} (already exists)")
                else:
                    print(f"  ❌ Error: {field_name}: {e}")
        
        # Migration 2: Document Metrics
        print("\n📄 Migration 2/2: Document Metrics Fields...")
        doc_fields = [
            ("jd_character_count", "INTEGER"),
            ("avg_resume_character_count", "INTEGER"),
            ("min_resume_character_count", "INTEGER"),
            ("max_resume_character_count", "INTEGER"),
        ]
        
        for field_name, field_type in doc_fields:
            try:
                sql = f"ALTER TABLE analyses ADD COLUMN {field_name} {field_type}"
                db.session.execute(text(sql))
                db.session.commit()
                print(f"  ✅ Added: {field_name}")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                    print(f"  ⏭️  Skipped: {field_name} (already exists)")
                else:
                    print(f"  ❌ Error: {field_name}: {e}")
        
        print("\n" + "="*70)
        print("✅ MIGRATIONS COMPLETE!")
        print("="*70 + "\n")

if __name__ == '__main__':
    run_migrations()
