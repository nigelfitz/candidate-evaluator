from app import create_app
from database import db, User, Transaction
from datetime import datetime

app = create_app()
app.app_context().push()

user = User.query.filter_by(email='testlocal3@test.com').first()
print('=== USER ACCOUNT INFO ===')
print(f'Created at: {user.created_at}')
print(f'Current balance: ${user.balance_usd}')
print(f'Total revenue: ${user.total_revenue_usd}')
print(f'Total analyses: {user.total_analyses_count}')
print('')
print('=== ALL TRANSACTIONS (oldest first) ===')
print(f'{"Date/Time":<18} {"Type":<7} {"Amount":>10} {"Running Bal":>12} Description')
print('-' * 100)

txns = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.created_at.asc()).all()
running_balance = 0
for txn in txns:
    running_balance += float(txn.amount_usd)
    date_str = txn.created_at.strftime('%m/%d %H:%M')
    type_str = txn.transaction_type.upper()
    amt_str = f'${float(txn.amount_usd):>7.2f}'
    bal_str = f'${running_balance:>7.2f}'
    print(f'{date_str:<18} {type_str:<7} {amt_str:>10} {bal_str:>12}   {txn.description}')

print('-' * 100)
print(f'Final balance should be: ${running_balance:.2f}')
print(f'Actual balance in DB:    ${user.balance_usd}')
if abs(running_balance - float(user.balance_usd)) > 0.01:
    print('❌ MISMATCH DETECTED!')
else:
    print('✅ Balance matches transaction history')
