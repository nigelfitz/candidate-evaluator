"""
Mark analysis ID 1 as deleted (it was hard-deleted earlier, so it doesn't exist)
Let's test with a different one instead
"""
from app import create_app
from database import db, Analysis
from datetime import datetime

app = create_app()

with app.app_context():
    # Get analysis ID 2 to test soft delete
    analysis = Analysis.query.get(2)
    
    if analysis:
        print(f"Soft-deleting analysis ID {analysis.id}: {analysis.job_title}")
        print(f"Created: {analysis.created_at}")
        
        # Mark as deleted (soft delete)
        analysis.deleted_at = datetime.utcnow()
        db.session.commit()
        
        print(f"Deleted at: {analysis.deleted_at}")
        print(f"\n✅ Analysis soft-deleted")
        print(f"🎯 Restart Flask server and refresh account page")
        print(f"   You should see '(deleted {analysis.deleted_at.strftime('%b %d, %Y')})' for this transaction")
    else:
        print("Analysis ID 2 not found")
