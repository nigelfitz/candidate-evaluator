# Wallet System Deployment Guide for Railway

## Overview
This guide will help you deploy the new Promotional & Wallet System to your Railway production environment.

## What's New

### 1. **$10 Test Drive for New Users**
- Every new signup automatically receives $10 credit
- Tracked via `welcome_bonus_claimed` field in database
- Transaction ledger records "Sign-up Bonus"

### 2. **New Wallet Page**
- Modern 3-card design: Single Job ($10), Hiring Sprint ($40→$50), Custom Amount
- Replaces old buy_credits page
- All site links updated to point to `/wallet`

### 3. **Bundle Promotions**
- Hiring Sprint: Pay $40, receive $50 balance (20% bonus)
- Stripe metadata tracks promotional purchases
- Transaction descriptions show full details

### 4. **Enhanced Transaction Tracking**
- Types: 'Sign-up Bonus', 'Stripe Purchase', 'Volume Bonus', 'Analysis Spend'
- Better accounting and revenue tracking

### 5. **Smart UI**
- Run Analysis button shows: "Run Analysis ($10.00)" or "Top Up Balance to Run ($10.00)"
- Welcome banner for new users
- Balance in header links to wallet

---

## Database Migration Required

### Step 1: Add New Column to `users` Table

The system needs a new `welcome_bonus_claimed` column to track signup bonuses.

**Option A: Automatic Migration (Recommended)**

1. Open your Railway project dashboard
2. Navigate to your `flask_app` service
3. Go to Settings → Deploy Triggers
4. Click "Deploy" to trigger a new deployment
5. Once deployed, open the service's shell/terminal
6. Run the migration script:
   ```bash
   python migrate_wallet_system.py
   ```

**Option B: Manual SQL (If you prefer direct database access)**

```sql
-- Add the new column
ALTER TABLE users 
ADD COLUMN welcome_bonus_claimed BOOLEAN DEFAULT FALSE NOT NULL;

-- Mark existing users as already having received signup bonus
-- (They already have their starting balance, so we don't want to double-credit)
UPDATE users 
SET welcome_bonus_claimed = TRUE 
WHERE balance_usd >= 10.00;
```

---

## Deployment Steps

### 1. **Pre-Deployment Checklist**
- [ ] Code has been pushed to main branch
- [ ] Railway is connected to your GitHub repository
- [ ] Stripe API keys are configured in Railway environment variables
- [ ] PostgreSQL database is healthy

### 2. **Deploy to Railway**

Railway should automatically detect the push and start deploying. Monitor the deployment:

```bash
# If using Railway CLI
railway logs

# Or watch in Railway dashboard
# Go to: Project → Service → Deployments
```

### 3. **Run Database Migration**

After successful deployment, run the migration:

```bash
# Option 1: Via Railway CLI
railway run python flask_app/migrate_wallet_system.py

# Option 2: Via Railway Dashboard
# Go to: Service → Shell (terminal icon)
# Then run:
cd flask_app
python migrate_wallet_system.py
```

Expected output:
```
🔄 Starting wallet system migration...
📝 Adding 'welcome_bonus_claimed' column to users table...
📝 Marking existing users as already having received signup bonus...
✅ Migration completed successfully!
   - Added 'welcome_bonus_claimed' column
   - Marked existing users with balance >= $10 as already credited
```

### 4. **Verify Deployment**

1. **Check Health**
   - Visit your site: `https://your-app.railway.app`
   - Verify the site loads without errors

2. **Test New User Registration**
   - Create a test account with a new email
   - Verify you see the welcome banner on dashboard
   - Check balance shows $10.00
   - Go to `/wallet` and verify the 3-card layout displays

3. **Test Wallet Page**
   - Click on your balance in the header
   - Verify you see: Single Job, Hiring Sprint, Custom Amount cards
   - "BEST VALUE • SAVE $10" badge should appear on Hiring Sprint

4. **Test Run Analysis Page**
   - Go to Start New Analysis workflow
   - On Step 4, verify button text matches your balance:
     - If balance >= $10: "Run Analysis ($10.00)"
     - If balance < $10: "Top Up Balance to Run ($10.00)"

5. **Test Stripe Integration (if enabled)**
   - Click "Hiring Sprint" card
   - Complete Stripe checkout for $40
   - After redirect, verify:
     - Balance increased by $50 (not $40)
     - Transaction ledger shows proper description
     - Balance in header updated immediately (no page refresh needed)

6. **Check Transaction History**
   - Go to Account page
   - Verify signup bonus transaction appears as "Sign-up Bonus - Welcome to Candidate Evaluator!"
   - If you tested Stripe, verify it shows correct amounts

---

## Rollback Plan (If Needed)

If something goes wrong, you can rollback:

### 1. **Rollback Code**
```bash
# In your local repository
git revert HEAD
git push origin main

# Railway will automatically deploy the previous version
```

### 2. **Rollback Database (if migration was run)**
```sql
-- Remove the new column (existing users won't be affected)
ALTER TABLE users DROP COLUMN IF EXISTS welcome_bonus_claimed;

-- Note: This doesn't affect balances or transactions, just removes the tracking field
```

---

## Testing Checklist

### New User Flow
- [ ] Registration gives $10 automatic credit
- [ ] Welcome banner appears on dashboard
- [ ] Transaction ledger shows "Sign-up Bonus"
- [ ] Balance in header shows $10.00
- [ ] Clicking balance opens wallet page

### Wallet Page
- [ ] Single Job card displays ($10)
- [ ] Hiring Sprint card displays ($40 → $50) with "BEST VALUE" badge
- [ ] Custom Amount card accepts input
- [ ] All cards link to Stripe checkout (or dev mode test)
- [ ] Card hover effects work

### Bundle Purchase (Hiring Sprint)
- [ ] Stripe charges $40
- [ ] User receives $50 credit
- [ ] Transaction description shows "paid $40, received $50"
- [ ] Balance updates immediately on return

### UI Balance Awareness
- [ ] Run Analysis button text changes based on balance
- [ ] "Top Up Balance to Run" links to wallet page
- [ ] Low balance warnings link to wallet (not buy_credits)

### Existing Users
- [ ] Existing users not double-credited
- [ ] Transaction history preserved
- [ ] Balances unchanged
- [ ] All features work normally

---

## Monitoring After Deployment

### Key Metrics to Watch

1. **User Signups**
   ```sql
   -- Check new signups today
   SELECT COUNT(*), AVG(balance_usd) as avg_starting_balance
   FROM users 
   WHERE DATE(created_at) = CURRENT_DATE;
   ```

2. **Bundle Adoption**
   ```sql
   -- Check how many Hiring Sprint purchases
   SELECT COUNT(*), SUM(amount_usd) as total_credited
   FROM transactions 
   WHERE description LIKE '%Hiring Sprint%'
   AND DATE(created_at) = CURRENT_DATE;
   ```

3. **Transaction Types**
   ```sql
   -- Breakdown of transaction types
   SELECT 
     CASE 
       WHEN description LIKE '%Sign-up Bonus%' THEN 'Sign-up Bonus'
       WHEN description LIKE '%Stripe Purchase%' AND description LIKE '%Volume Bonus%' THEN 'Bundle Purchase'
       WHEN description LIKE '%Stripe Purchase%' THEN 'Regular Purchase'
       WHEN description LIKE '%Analysis Spend%' THEN 'Analysis'
       ELSE 'Other'
     END as type,
     COUNT(*) as count,
     SUM(amount_usd) as total_amount
   FROM transactions 
   WHERE DATE(created_at) = CURRENT_DATE
   GROUP BY type;
   ```

---

## Troubleshooting

### Issue: New users not getting $10 credit

**Check:**
1. Look in Railway logs for registration errors
2. Verify `system_settings.json` has `new_user_welcome_credit` set to 10
3. Check database: `SELECT * FROM users ORDER BY created_at DESC LIMIT 5;`

**Fix:**
```python
# Manually credit a user (run in Railway shell)
python
from app import create_app
from database import db, User, Transaction
from decimal import Decimal

app = create_app()
with app.app_context():
    user = User.query.filter_by(email='test@example.com').first()
    user.balance_usd += Decimal('10.00')
    user.welcome_bonus_claimed = True
    
    transaction = Transaction(
        user_id=user.id,
        amount_usd=Decimal('10.00'),
        transaction_type='credit',
        description='Sign-up Bonus - Welcome to Candidate Evaluator!'
    )
    db.session.add(transaction)
    db.session.commit()
```

### Issue: Bundle not crediting $50

**Check:**
1. Look in Stripe webhook logs (if webhooks configured)
2. Check Railway logs for fulfill_order errors
3. Verify metadata in Stripe dashboard

**Fix:**
The `fulfill_order` function now checks for `credit_amount` in metadata. If using old Stripe sessions, they won't have this field and will default to `charge_amount`.

### Issue: Migration fails with "column already exists"

This is safe to ignore. The migration script checks if the column exists before adding it.

### Issue: Welcome banner not showing

**Check:**
1. User must have `welcome_bonus_claimed = True`
2. User must have `total_analyses_count = 0` (first-time user)
3. Check browser console for JavaScript errors

---

## Success Criteria

✅ **Deployment is successful when:**
- New users receive $10 automatically
- Wallet page displays correctly with 3 cards
- Hiring Sprint bundle works ($40 → $50)
- All transaction types are properly recorded
- Existing users unaffected
- No errors in Railway logs
- Stripe integration works (if enabled)

---

## Support

If you encounter issues:
1. Check Railway logs: `railway logs --tail 100`
2. Check database state: `railway run psql`
3. Review transaction history in Account page
4. Test in dev mode first (FLASK_ENV=development)

---

## Summary of Files Changed

### Backend
- `flask_app/database.py` - Added `welcome_bonus_claimed` field
- `flask_app/auth.py` - Auto-credit on registration
- `flask_app/payments.py` - Bundle logic + wallet route
- `flask_app/migrate_wallet_system.py` - Migration script

### Templates
- `flask_app/templates/wallet.html` - New 3-card wallet page
- `flask_app/templates/dashboard.html` - Welcome banner
- `flask_app/templates/run_analysis.html` - Dynamic button text
- `flask_app/templates/base.html` - Header balance link
- `flask_app/templates/account.html` - Wallet links
- `flask_app/templates/history.html` - Wallet links
- `flask_app/templates/insights.html` - Wallet links

All changes are backward compatible and safe for production deployment.
