"""
Migration script to add jd_full_text and jd_filename fields to analyses table
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
    # Check if columns already exist
    cursor.execute("PRAGMA table_info(analyses)")
    columns = [col[1] for col in cursor.fetchall()]
    
    needs_jd_full_text = 'jd_full_text' not in columns
    needs_jd_filename = 'jd_filename' not in columns
    
    if needs_jd_full_text:
        print("Adding jd_full_text column to analyses table...")
        cursor.execute("ALTER TABLE analyses ADD COLUMN jd_full_text TEXT")
        print("✓ Successfully added jd_full_text column")
    else:
        print("✓ jd_full_text column already exists in analyses table")
    
    if needs_jd_filename:
        print("Adding jd_filename column to analyses table...")
        cursor.execute("ALTER TABLE analyses ADD COLUMN jd_filename VARCHAR(255)")
        print("✓ Successfully added jd_filename column")
    else:
        print("✓ jd_filename column already exists in analyses table")
    
    conn.commit()
    
except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()
finally:
    conn.close()
    print("Migration complete!")
