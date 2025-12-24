"""
Add candidate_files table for storing resume files
"""
from app import create_app
from database import db
from sqlalchemy import text

app = create_app()

with app.app_context():
    print("Creating candidate_files table...")
    
    # Create the table
    db.session.execute(text('''
        CREATE TABLE IF NOT EXISTS candidate_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            candidate_name VARCHAR(255) NOT NULL,
            file_name VARCHAR(255) NOT NULL,
            file_bytes BLOB NOT NULL,
            extracted_text TEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
        )
    '''))
    
    # Create index
    print("Creating index on analysis_id...")
    db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_candidate_files_analysis_id ON candidate_files(analysis_id)'))
    
    db.session.commit()
    
    print("✅ Successfully created candidate_files table with CASCADE delete")
    print("✅ Candidate resumes will now be stored for viewing")
