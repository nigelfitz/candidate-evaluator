"""
Add deleted_at column to analyses table for soft deletes
"""
from app import create_app
from database import db
from sqlalchemy import text

app = create_app()

with app.app_context():
    print("Adding deleted_at column to analyses table...")
    
    # Add the column
    db.session.execute(text('ALTER TABLE analyses ADD COLUMN deleted_at DATETIME'))
    
    # Create index for performance
    print("Creating index on deleted_at...")
    db.session.execute(text('CREATE INDEX idx_analyses_deleted_at ON analyses(deleted_at)'))
    
    db.session.commit()
    
    print("✅ Successfully added deleted_at column with index")
    print("✅ Soft deletes enabled - analyses will be marked as deleted instead of removed")
