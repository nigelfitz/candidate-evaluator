"""Delete a user from production database"""
from flask_app.database import db, User
from flask_app.app import create_app

app = create_app('production')
app.app_context().push()

email = 'nigelfitz@gmail.com'
user = User.query.filter_by(email=email).first()

if user:
    db.session.delete(user)
    db.session.commit()
    print(f'✅ User {email} deleted from production')
else:
    print(f'❌ User {email} not found in production')
