"""Add JobQueue table for background processing

Revision ID: add_job_queue
Created: 2026-01-16
"""

from flask import Flask
from database import db, JobQueue
import os
import sys

def run_migration():
    """Add JobQueue table to database"""
    # Create Flask app
    app = Flask(__name__)
    
    # Configure database
    database_url = os.environ.get('DATABASE_URL')
    if database_url and database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///instance/candidate_evaluator.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize database
    db.init_app(app)
    
    with app.app_context():
        try:
            # Check if table already exists
            inspector = db.inspect(db.engine)
            if 'job_queue' in inspector.get_table_names():
                print("✅ JobQueue table already exists, skipping migration")
                return True
            
            # Create the table
            JobQueue.__table__.create(db.engine)
            print("✅ Successfully created JobQueue table")
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    success = run_migration()
    sys.exit(0 if success else 1)
