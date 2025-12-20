"""
Debug the existing_analyses query
"""
from app import create_app
from database import db, Transaction, Analysis
from flask_login import current_user

app = create_app()

with app.app_context():
    # Get a user (user ID 1)
    from database import User
    user = User.query.first()
    
    if not user:
        print("No users found")
    else:
        print(f"User: {user.email}\n")
        
        # Get transactions like the route does
        transactions = user.transactions.order_by(db.desc('created_at')).limit(50).all()
        
        print(f"Total transactions: {len(transactions)}\n")
        
        # Check analysis IDs
        analysis_ids = [t.analysis_id for t in transactions if t.analysis_id]
        print(f"Analysis IDs from transactions: {analysis_ids}\n")
        
        # Query for existing analyses
        if analysis_ids:
            existing = Analysis.query.filter(Analysis.id.in_(analysis_ids)).all()
            print(f"Found {len(existing)} existing analyses")
            existing_analyses = {a.id for a in existing}
            print(f"Existing analysis IDs: {existing_analyses}\n")
        else:
            print("No analysis IDs found in transactions\n")
            existing_analyses = set()
        
        # Show each transaction
        print("Transactions breakdown:")
        print("-" * 80)
        for t in transactions[:10]:  # Show first 10
            status = "EXISTS" if t.analysis_id in existing_analyses else "DELETED" if t.analysis_id else "NO ANALYSIS"
            print(f"ID {t.id}: {t.description[:50]:50} | analysis_id={t.analysis_id} | {status}")
