"""
DANGEROUS: Wipe all user accounts and data from database
This script deletes ALL users and their associated data.
Use this ONLY for fresh testing starts.
"""
import os
import sys
from database import db, User, Analysis, Transaction, Draft, DraftResume, CandidateFile, UserSettings, AdminAuditLog, AdminLoginAttempt, Feedback

def wipe_all_users():
    """Delete ALL users and their data from the database"""
    
    print("\n" + "="*70)
    print("⚠️  WARNING: DATABASE WIPE OPERATION")
    print("="*70)
    print("\nThis will DELETE:")
    print("  • All user accounts")
    print("  • All analyses and job history")
    print("  • All transactions and balance history")
    print("  • All drafts and draft resumes")
    print("  • All candidate files")
    print("  • All user settings")
    print("  • All admin audit logs and login attempts")
    print("  • All feedback")
    print("\nThis operation CANNOT be undone!")
    print("="*70)
    
    # Count current data
    user_count = User.query.count()
    analysis_count = Analysis.query.count()
    transaction_count = Transaction.query.count()
    draft_count = Draft.query.count()
    file_count = CandidateFile.query.count()
    
    print(f"\nCurrent database state:")
    print(f"  Users: {user_count}")
    print(f"  Analyses: {analysis_count}")
    print(f"  Transactions: {transaction_count}")
    print(f"  Drafts: {draft_count}")
    print(f"  Candidate Files: {file_count}")
    
    if user_count == 0:
        print("\n✅ Database is already empty. Nothing to delete.")
        return
    
    # Triple confirmation required
    print("\n" + "="*70)
    response1 = input("Type 'DELETE ALL USERS' to continue (or anything else to cancel): ")
    if response1 != "DELETE ALL USERS":
        print("\n❌ Operation cancelled.")
        return
    
    response2 = input("Are you absolutely sure? Type 'YES' to proceed: ")
    if response2 != "YES":
        print("\n❌ Operation cancelled.")
        return
    
    response3 = input("Final confirmation. Type 'WIPE DATABASE' to proceed: ")
    if response3 != "WIPE DATABASE":
        print("\n❌ Operation cancelled.")
        return
    
    print("\n🔥 Starting database wipe...")
    
    try:
        # Delete in order to respect foreign key constraints
        print("\n1. Deleting candidate files...")
        deleted = CandidateFile.query.delete()
        print(f"   ✓ Deleted {deleted} candidate files")
        
        print("2. Deleting analyses...")
        deleted = Analysis.query.delete()
        print(f"   ✓ Deleted {deleted} analyses")
        
        print("3. Deleting transactions...")
        deleted = Transaction.query.delete()
        print(f"   ✓ Deleted {deleted} transactions")
        
        print("4. Deleting drafts...")
        deleted = Draft.query.delete()
        print(f"   ✓ Deleted {deleted} drafts")
        
        print("5. Deleting draft resumes...")
        deleted = DraftResume.query.delete()
        print(f"   ✓ Deleted {deleted} draft resumes")
        
        print("6. Deleting user settings...")
        deleted = UserSettings.query.delete()
        print(f"   ✓ Deleted {deleted} user settings")
        
        print("7. Deleting admin login attempts...")
        deleted = AdminLoginAttempt.query.delete()
        print(f"   ✓ Deleted {deleted} admin login attempts")
        
        print("8. Deleting admin audit logs...")
        deleted = AdminAuditLog.query.delete()
        print(f"   ✓ Deleted {deleted} admin audit logs")
        
        print("9. Deleting feedback...")
        deleted = Feedback.query.delete()
        print(f"   ✓ Deleted {deleted} feedback entries")
        
        print("10. Deleting all users...")
        deleted = User.query.delete()
        print(f"   ✓ Deleted {deleted} users")
        
        db.session.commit()
        
        print("\n" + "="*70)
        print("✅ DATABASE WIPED SUCCESSFULLY")
        print("="*70)
        print("\nAll users and data have been permanently deleted.")
        print("The database is now completely clean.")
        print("\nYou can now:")
        print("  • Register new test accounts")
        print("  • Start fresh testing")
        print("  • Open registrations to real users")
        print("="*70 + "\n")
        
    except Exception as e:
        db.session.rollback()
        print(f"\n❌ ERROR during wipe: {e}")
        print("Database changes have been rolled back.")
        sys.exit(1)

if __name__ == '__main__':
    from app import create_app
    
    app = create_app()
    
    with app.app_context():
        # Check which database we're using
        db_url = app.config.get('SQLALCHEMY_DATABASE_URI', '')
        
        print("\n" + "="*70)
        if 'postgresql' in db_url.lower():
            print("🌐 CONNECTED TO: PostgreSQL (Railway/Production)")
        else:
            print("💻 CONNECTED TO: SQLite (Local)")
        print("="*70)
        
        wipe_all_users()
