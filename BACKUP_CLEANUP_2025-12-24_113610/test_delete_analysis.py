"""
Delete one test analysis to demonstrate the deleted analysis display
"""
from app import create_app
from database import db, Analysis, Transaction

app = create_app()

with app.app_context():
    # Get the oldest analysis to delete (ID 1)
    analysis = Analysis.query.get(1)
    
    if analysis:
        print(f"Deleting analysis ID {analysis.id}: {analysis.job_title}")
        print(f"Created: {analysis.created_at}")
        
        # Check transaction before deletion
        transaction = Transaction.query.filter_by(analysis_id=analysis.id).first()
        if transaction:
            print(f"\nLinked transaction ID {transaction.id}:")
            print(f"  Description: {transaction.description}")
            print(f"  analysis_id BEFORE deletion: {transaction.analysis_id}")
        
        # Delete the analysis
        db.session.delete(analysis)
        db.session.commit()
        
        # Check transaction after deletion (refresh from DB)
        db.session.expire_all()
        transaction = Transaction.query.get(transaction.id)
        print(f"\n  analysis_id AFTER deletion: {transaction.analysis_id}")
        print(f"\n✅ Analysis deleted - transaction now shows analysis_id={transaction.analysis_id}")
        print(f"🎯 Refresh your account page to see '(deleted)' indicator")
    else:
        print("Analysis ID 1 not found")
