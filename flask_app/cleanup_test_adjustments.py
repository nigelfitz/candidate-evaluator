"""Clean up problematic adjustment transactions from test account"""
from database import db, User, Transaction
from app import create_app
from decimal import Decimal

app = create_app()

with app.app_context():
    print("=" * 80)
    print("CLEANUP: Removing problematic adjustment transactions")
    print("=" * 80)
    
    # Find the test user
    test_user = User.query.filter_by(email='testlocal3@test.com').first()
    
    if not test_user:
        print("❌ Test user not found")
        exit(1)
    
    print(f"\n✅ Found user: {test_user.email}")
    print(f"   Current balance: ${test_user.balance_usd}")
    
    # Find and remove ALL adjustment transactions (old and new format)
    problem_txns = Transaction.query.filter(
        Transaction.user_id == test_user.id
    ).filter(
        db.or_(
            Transaction.description.like('%Admin Balance Adjustment%'),
            Transaction.description.like('[ADJUSTMENT]%')
        )
    ).all()
    
    print(f"\n📋 Found {len(problem_txns)} adjustment transactions to review:")
    
    total_adjustment = Decimal('0')
    for txn in problem_txns:
        print(f"   - {txn.created_at}: {txn.transaction_type} ${txn.amount_usd} - {txn.description}")
        total_adjustment += Decimal(str(txn.amount_usd))
    
    print(f"\n💰 Total adjustment amount: ${total_adjustment}")
    
    # Calculate what balance SHOULD be (from non-adjustment transactions)
    all_txns = Transaction.query.filter_by(user_id=test_user.id).order_by(Transaction.created_at).all()
    correct_balance = Decimal('0')
    
    for txn in all_txns:
        if 'Admin Balance Adjustment' not in (txn.description or ''):
            correct_balance += Decimal(str(txn.amount_usd))
    
    print(f"\n🧮 Calculated balance from normal transactions: ${correct_balance}")
    print(f"   Current user.balance_usd: ${test_user.balance_usd}")
    print(f"   Discrepancy: ${test_user.balance_usd - correct_balance}")
    
    response = input(f"\n⚠️  Delete {len(problem_txns)} adjustment transactions and set balance to ${correct_balance}? (yes/no): ")
    
    if response.lower() == 'yes':
        # Delete adjustment transactions
        for txn in problem_txns:
            db.session.delete(txn)
        
        # Set correct balance
        test_user.balance_usd = correct_balance
        
        db.session.commit()
        
        print(f"\n✅ Cleanup complete!")
        print(f"   Deleted {len(problem_txns)} transactions")
        print(f"   Reset balance to ${test_user.balance_usd}")
    else:
        print("\n❌ Cleanup cancelled")
