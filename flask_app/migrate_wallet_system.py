"""
Migration: Add welcome_bonus_claimed field to User table

This migration adds a new field to track whether a user has received
their signup bonus, ensuring we don't accidentally double-credit existing users.

To run this migration on Railway:
1. Open Railway project dashboard
2. Go to your flask_app service
3. Click on "Settings" then "Variables"
4. Add a temporary variable: RUN_MIGRATION=wallet_system
5. Deploy the code
6. Remove the RUN_MIGRATION variable after deployment

Or run manually via Railway CLI:
railway run python migrate_wallet_system.py
"""

from database import db, User
from sqlalchemy import text

def migrate():
    """Add welcome_bonus_claimed field to users table"""
    print("🔄 Starting wallet system migration...")
    
    try:
        # Check if column already exists
        result = db.session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='users' AND column_name='welcome_bonus_claimed';
        """))
        
        if result.fetchone():
            print("✅ Column 'welcome_bonus_claimed' already exists. Skipping.")
            return True
        
        # Add the column with default False
        print("📝 Adding 'welcome_bonus_claimed' column to users table...")
        db.session.execute(text("""
            ALTER TABLE users 
            ADD COLUMN welcome_bonus_claimed BOOLEAN DEFAULT FALSE NOT NULL;
        """))
        
        # Set welcome_bonus_claimed=True for all existing users
        # (They already have their starting balance, so we don't want to double-credit them)
        print("📝 Marking existing users as already having received signup bonus...")
        db.session.execute(text("""
            UPDATE users 
            SET welcome_bonus_claimed = TRUE 
            WHERE balance_usd >= 10.00;
        """))
        
        db.session.commit()
        print("✅ Migration completed successfully!")
        print("   - Added 'welcome_bonus_claimed' column")
        print("   - Marked existing users with balance >= $10 as already credited")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        db.session.rollback()
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    from app import create_app
    app = create_app()
    
    with app.app_context():
        success = migrate()
        exit(0 if success else 1)
