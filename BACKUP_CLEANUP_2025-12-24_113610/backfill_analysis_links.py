"""
Backfill analysis_id links for existing transactions
"""
from app import create_app
from database import db, Analysis, Transaction
from datetime import timedelta

app = create_app()

with app.app_context():
    # Get all analyses
    analyses = Analysis.query.all()
    
    updated_count = 0
    
    for analysis in analyses:
        # Find transaction within 1 second of analysis creation
        transaction = Transaction.query.filter(
            Transaction.transaction_type == 'debit',
            Transaction.created_at >= analysis.created_at - timedelta(seconds=1),
            Transaction.created_at <= analysis.created_at + timedelta(seconds=1),
            Transaction.analysis_id == None
        ).first()
        
        if transaction:
            transaction.analysis_id = analysis.id
            updated_count += 1
            print(f"✓ Linked transaction {transaction.id} to analysis {analysis.id} - {analysis.job_title[:50]}")
    
    db.session.commit()
    
    print(f"\n✅ Updated {updated_count} transactions with analysis_id links")
    
    # Verify
    linked = Transaction.query.filter(Transaction.analysis_id != None).count()
    total = Transaction.query.count()
    print(f"📊 {linked} of {total} transactions now have analysis links")
