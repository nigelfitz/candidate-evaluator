from app import create_app
from database import db, User

app = create_app()
app.app_context().push()

# Find and delete the user
user = User.query.filter_by(email='testlocal3@test.com').first()

if user:
    print(f'Found user: {user.email}')
    print(f'  Created: {user.created_at}')
    print(f'  Balance: ${user.balance_usd}')
    print(f'  Total analyses: {user.total_analyses_count}')
    
    # Delete user (cascade will delete related transactions, analyses, etc.)
    db.session.delete(user)
    db.session.commit()
    
    print('\n✅ User deleted successfully!')
else:
    print('❌ User not found')
