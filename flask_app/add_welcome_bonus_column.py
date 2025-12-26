"""
Simple migration script to add welcome_bonus_claimed column
Run via: railway run python add_welcome_bonus_column.py
"""
import os
import psycopg2

DATABASE_URL = os.environ.get('DATABASE_URL')

if not DATABASE_URL:
    print("❌ DATABASE_URL not found")
    exit(1)

print("🔄 Connecting to database...")
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

try:
    # Check if column exists
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='users' AND column_name='welcome_bonus_claimed';
    """)
    
    if cur.fetchone():
        print("✅ Column 'welcome_bonus_claimed' already exists")
    else:
        print("📝 Adding welcome_bonus_claimed column...")
        cur.execute("""
            ALTER TABLE users 
            ADD COLUMN welcome_bonus_claimed BOOLEAN DEFAULT FALSE NOT NULL;
        """)
        
        print("📝 Marking existing users as already having received bonus...")
        cur.execute("""
            UPDATE users 
            SET welcome_bonus_claimed = TRUE 
            WHERE created_at < NOW();
        """)
        
        conn.commit()
        print("✅ Migration completed successfully!")
        
except Exception as e:
    conn.rollback()
    print(f"❌ Error: {e}")
    exit(1)
finally:
    cur.close()
    conn.close()
