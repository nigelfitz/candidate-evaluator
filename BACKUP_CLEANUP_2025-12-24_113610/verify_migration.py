"""
Verify database schema after migration
"""
from app import create_app
from database import db
from sqlalchemy import text

app = create_app()

with app.app_context():
    result = db.session.execute(text("PRAGMA table_info(transactions)"))
    
    print("\nTransactions table structure:")
    print("-" * 60)
    for row in result:
        col_id, name, type_, notnull, default, pk = row
        nullable = "nullable" if notnull == 0 else "NOT NULL"
        pk_marker = " (PRIMARY KEY)" if pk else ""
        print(f"{name:20} {type_:15} {nullable:10}{pk_marker}")
    
    # Check if any transactions have analysis_id
    result2 = db.session.execute(text("SELECT COUNT(*) as total, COUNT(analysis_id) as with_analysis FROM transactions"))
    row = result2.fetchone()
    print("\n" + "-" * 60)
    print(f"Total transactions: {row[0]}")
    print(f"Transactions with analysis_id: {row[1]}")
    print(f"Transactions without analysis_id: {row[0] - row[1]}")
