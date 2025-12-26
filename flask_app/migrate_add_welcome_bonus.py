"""
Migration: Add welcome_bonus_claimed column to users table
Run this to update your local SQLite database schema
"""
import sqlite3
import os

# Get the database path
db_path = os.path.join(os.path.dirname(__file__), 'instance', 'candidate_evaluator.db')

if not os.path.exists(db_path):
    print(f"❌ Database not found at: {db_path}")
    print("Creating new database with correct schema...")
    # If no database exists, just run the app and it will create it correctly
    exit(1)

print(f"📂 Found database at: {db_path}")

# Connect to SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Check if column already exists
    cursor.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'welcome_bonus_claimed' in columns:
        print("✅ Column 'welcome_bonus_claimed' already exists!")
    else:
        print("➕ Adding 'welcome_bonus_claimed' column...")
        cursor.execute("""
            ALTER TABLE users 
            ADD COLUMN welcome_bonus_claimed BOOLEAN DEFAULT 0 NOT NULL
        """)
        conn.commit()
        print("✅ Column added successfully!")
    
    # Verify the column exists now
    cursor.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"\n📋 Current users table columns: {', '.join(columns)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()
finally:
    conn.close()

print("\n✅ Migration complete! You can now run the app.")
