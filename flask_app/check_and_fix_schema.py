"""
Check and add any missing columns to users table
"""
import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'instance', 'candidate_evaluator.db')

print(f"📂 Database: {db_path}\n")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Check current columns
    cursor.execute("PRAGMA table_info(users)")
    existing_columns = [row[1] for row in cursor.fetchall()]
    print(f"📋 Existing columns: {', '.join(existing_columns)}\n")
    
    # Define all expected columns (from database.py User model)
    required_columns = {
        'id': 'INTEGER PRIMARY KEY',
        'email': 'VARCHAR(120)',
        'password_hash': 'VARCHAR(255)',
        'name': 'VARCHAR(100)',
        'balance_usd': 'NUMERIC(10, 2) DEFAULT 0.00 NOT NULL',
        'welcome_bonus_claimed': 'BOOLEAN DEFAULT 0 NOT NULL',
        'is_suspended': 'BOOLEAN DEFAULT 0 NOT NULL',
        'suspension_reason': 'VARCHAR(500)',
        'created_at': 'DATETIME',
        'last_login': 'DATETIME',
        'signup_source': 'VARCHAR(100)',
        'total_analyses_count': 'INTEGER DEFAULT 0 NOT NULL',
        'total_revenue_usd': 'NUMERIC(10, 2) DEFAULT 0.00 NOT NULL'
    }
    
    # Find missing columns
    missing_columns = [col for col in required_columns.keys() if col not in existing_columns]
    
    if not missing_columns:
        print("✅ All columns present!")
    else:
        print(f"➕ Missing columns: {', '.join(missing_columns)}\n")
        
        for col in missing_columns:
            col_def = required_columns[col]
            print(f"Adding {col}...")
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col} {col_def}")
        
        conn.commit()
        print("\n✅ All missing columns added!")
    
    # Show final schema
    cursor.execute("PRAGMA table_info(users)")
    final_columns = [row[1] for row in cursor.fetchall()]
    print(f"\n📋 Final columns: {', '.join(final_columns)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()
finally:
    conn.close()
