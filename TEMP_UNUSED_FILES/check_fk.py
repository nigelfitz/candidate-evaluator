"""
Check foreign key configuration
"""
from app import create_app
from database import db
from sqlalchemy import text

app = create_app()

with app.app_context():
    # Check if foreign keys are enabled
    result = db.session.execute(text('PRAGMA foreign_keys'))
    fk_enabled = result.fetchone()[0]
    print(f"Foreign keys enabled: {fk_enabled}")
    
    if fk_enabled == 0:
        print("\n⚠️  WARNING: Foreign keys are NOT enabled in SQLite!")
        print("This means ON DELETE SET NULL won't work.")
        print("\nTo fix: Need to enable foreign keys in database connection")
    
    # Check foreign key constraints on transactions table
    result = db.session.execute(text('PRAGMA foreign_key_list(transactions)'))
    print("\nForeign key constraints on transactions table:")
    print("-" * 80)
    for row in result:
        print(f"Column: {row[3]} → {row[2]}.{row[4]} (ON DELETE {row[5]})")
