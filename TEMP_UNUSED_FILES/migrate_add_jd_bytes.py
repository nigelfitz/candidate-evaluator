"""Migration script to add jd_bytes column to drafts table"""
import sqlite3
import os

# Path to database
db_path = os.path.join(os.path.dirname(__file__), 'candidate_evaluator.db')

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Add the new column
    cursor.execute('ALTER TABLE drafts ADD COLUMN jd_bytes BLOB')
    conn.commit()
    print("✅ Successfully added jd_bytes column to drafts table")
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        print("⚠️ Column jd_bytes already exists")
    else:
        print(f"❌ Error: {e}")
finally:
    conn.close()

print("\nMigration complete!")
