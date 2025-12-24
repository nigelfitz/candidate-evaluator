"""
Migration script to add job_title column to drafts table
"""
import sqlite3
import os

# Path to the database
db_path = os.path.join(os.path.dirname(__file__), 'instance', 'candidate_evaluator.db')

print(f"Connecting to database: {db_path}")

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Check if column already exists
    cursor.execute("PRAGMA table_info(drafts)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'job_title' in columns:
        print("✓ job_title column already exists in drafts table")
    else:
        # Add the job_title column
        print("Adding job_title column to drafts table...")
        cursor.execute("ALTER TABLE drafts ADD COLUMN job_title VARCHAR(255)")
        conn.commit()
        print("✓ Successfully added job_title column to drafts table")
    
except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()
finally:
    conn.close()
    print("Migration complete!")
