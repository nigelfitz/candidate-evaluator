"""
Quick script to refund a user for incorrect charges
Run with: flask shell < refund_user.py
Or copy/paste into: flask shell
"""

from database import db, User, Transaction
from decimal import Decimal

# Find the user
user = User.query.filter_by(email='testlocal3@test.com').first()

if not user:
    print("User not found!")
else:
    print(f"Found user: {user.email}")
    print(f"Current balance: ${user.balance_usd}")
    
    # Refund amount: They were charged $6 twice instead of $10 once
    # So they were overcharged by: ($6 + $6) - $10 = $2
    refund_amount = Decimal('2.00')
    
    print(f"\nRefunding ${refund_amount} for duplicate/incorrect charge")
    
    user.balance_usd += refund_amount
    
    # Create refund transaction
    refund = Transaction(
        user_id=user.id,
        amount_usd=refund_amount,
        transaction_type='credit',
        description='Refund for duplicate charge - pricing model update'
    )
    db.session.add(refund)
    db.session.commit()
    
    print(f"New balance: ${user.balance_usd}")
    print("Refund complete!")
