"""Verify and fix database schema"""
from app import create_app
from database import db, Draft

app = create_app()

with app.app_context():
    # Drop all tables and recreate
    db.drop_all()
    db.create_all()
    print("✅ Database recreated with correct schema")
    
    # Verify drafts table has jd_bytes column
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    columns = [col['name'] for col in inspector.get_columns('drafts')]
    print(f"📋 Drafts table columns: {columns}")
    
    if 'jd_bytes' in columns:
        print("✅ jd_bytes column exists!")
    else:
        print("❌ jd_bytes column MISSING!")
