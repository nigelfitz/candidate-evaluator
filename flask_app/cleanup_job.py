from app import create_app
from database import db, JobQueue, Analysis

app = create_app()
with app.app_context():
    job = JobQueue.query.get(1)
    if job:
        db.session.delete(job)
        print('✅ Deleted Job #1')
    
    analysis = Analysis.query.get(106)
    if analysis:
        db.session.delete(analysis)
        print('✅ Deleted Analysis #106')
    
    db.session.commit()
    print('✅ Database cleaned up - ready for new test')
