"""
Check existing analyses and their costs to potentially backfill transaction links
"""
from app import create_app
from database import db, Analysis, Transaction
from sqlalchemy import text

app = create_app()

with app.app_context():
    # Get all analyses
    analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(10).all()
    
    print("\nRecent Analyses:")
    print("-" * 80)
    for a in analyses:
        print(f"ID: {a.id:3} | Date: {a.created_at} | Cost: ${a.cost_usd:6.2f} | Job: {a.job_title[:40]}")
    
    # Get all debit transactions
    debits = Transaction.query.filter_by(transaction_type='debit').order_by(Transaction.created_at.desc()).limit(10).all()
    
    print("\n\nRecent Debit Transactions:")
    print("-" * 80)
    for t in debits:
        print(f"ID: {t.id:3} | Date: {t.created_at} | Amount: ${abs(t.amount_usd):6.2f} | Desc: {t.description[:40]}")
        print(f"      analysis_id: {t.analysis_id}")
