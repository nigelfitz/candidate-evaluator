# Balance Monitoring System - Complete Guide

## Overview
The balance monitoring system automatically detects and alerts when user account balances don't match their transaction history. This helps catch bugs early before they compound.

## How It Works

### Automatic Detection
- Every time a user views their account/ledger page, the system validates their balance
- If the actual database balance doesn't match the calculated balance from transaction history, an alert is triggered
- Alerts are sent once per user per session to avoid spam

### Email Alerts
When a balance mismatch is detected:
- Admin receives an email with:
  - User's name and email
  - Actual balance (from database)
  - Calculated balance (from transaction history)
  - Discrepancy amount
  - Number of transactions
  - Direct link to Balance Audit dashboard

## Using the Balance Audit Dashboard

### Accessing the Dashboard
Navigate to: **Admin Panel → Finance → Balance Audit**
URL: `/admin/balance-audit`

### What You'll See
- **Green Badge**: "All Accounts Balanced" - Everything is good!
- **Red Badge**: "X Accounts with Issues" - Attention needed

### Account Table Columns
1. **User** - Name and email
2. **Actual Balance** - What's in the database (`user.balance_usd`)
3. **Calculated Balance** - Sum of all transaction amounts
4. **Discrepancy** - The difference (positive = database is higher)
5. **Transactions** - Number of transactions for this user
6. **Actions** - "Adjust Balance" button

## Adjusting Balances Manually

### When to Use Manual Adjustment
- After investigating a discrepancy and determining the root cause
- When you need to reconcile accounts after fixing a bug
- To create a correcting transaction with a documented reason

### How to Adjust
1. Click "Adjust Balance" for the affected user
2. Review the balance information displayed:
   - Actual Balance (database)
   - Calculated Balance (from transactions)
   - Discrepancy
3. Enter the adjustment amount:
   - **Negative value** to decrease balance (e.g., -7.00)
   - **Positive value** to increase balance (e.g., +7.00)
   - Default is pre-filled with the suggested correction
4. Enter a detailed reason (required) - this is for the audit trail
5. Click "Create Adjustment Transaction"

### What Happens After Adjustment
- A new transaction is created with type "Adjustment"
- The user's balance is updated
- The adjustment is logged in the admin audit trail
- The user no longer appears in the Balance Audit dashboard

## Example Scenario

### Problem: Missing Unlock Transactions
**Situation**: User's test account shows:
- Actual Balance: $45.00
- Calculated Balance: $52.00
- Discrepancy: +$7.00

**Investigation**: After reviewing the ledger, you discover that unlock insight transactions ($1.50 each) weren't being recorded due to a bug (now fixed).

**Resolution**:
1. Go to Balance Audit dashboard
2. Click "Adjust Balance" for the test user
3. Enter amount: `-7.00` (to reduce balance by $7)
4. Reason: "Reconciling missing unlock transactions from testing phase - bug now fixed"
5. Submit adjustment

**Result**: Account now shows $45.00 in both actual and calculated balance. Future unlocks will be properly recorded.

## Threshold Settings
- Current threshold: **$0.01** (1 cent)
- Accounts are flagged only if discrepancy > $0.01
- This prevents false alerts from floating-point rounding

## Technical Details

### Balance Calculation
The system calculates balance by:
1. Starting from zero
2. Adding each transaction amount chronologically
3. Comparing final total to `user.balance_usd`

### Why This Matters
The old system calculated backward from the current balance, which hid missing transactions by adjusting the opening balance. The new forward calculation exposes discrepancies immediately.

### Email Configuration
- Uses existing SendGrid configuration
- Recipient: `ADMIN_EMAIL` environment variable (defaults to admin@candidateevaluator.com)
- Works in both development and production
- Degrades gracefully if email is not configured

## Best Practices

### Regular Monitoring
- Check the Balance Audit dashboard regularly (weekly recommended)
- Review email alerts as soon as they arrive
- Investigate discrepancies before adjusting

### Investigation Steps
1. Check the user's transaction history (Account page)
2. Look for missing transactions
3. Review recent changes to balance-affecting code
4. Verify the bug is fixed before adjusting
5. Document findings in the adjustment reason

### Documentation
- Always provide detailed reasons for adjustments
- Reference bug fixes or issue numbers
- Include date ranges if relevant
- These reasons appear in the admin audit log

## Troubleshooting

### "No email alert received"
- Check SendGrid configuration in Railway
- Verify `ADMIN_EMAIL` environment variable is set
- Check spam folder
- Check Flask logs for email sending errors

### "Alert sent multiple times"
- Alerts should only send once per user per session
- If you restart Flask, the session resets
- This is intentional to prevent spam during active development

### "Adjustment didn't work"
- Verify the amount sign is correct (negative to decrease, positive to increase)
- Check that Flask restarted after making the adjustment
- Clear browser cache and reload the Balance Audit page

## Future Enhancements to Consider
- Scheduled daily balance audit (cron job)
- Balance check before processing payments (preventative)
- Dashboard widget showing overall balance health
- Configurable threshold (currently hardcoded at $0.01)
- Bulk adjustment tool for fixing multiple accounts

## Files Modified
- `app.py` - Added balance checking functions and admin routes
- `admin_balance_audit.html` - Dashboard template
- `admin_balance_adjustment.html` - Adjustment form template
- `admin_*.html` - Added Balance Audit link to Finance dropdown
- Email infrastructure uses existing `email_utils.py`

## Summary
This system shifts from reactive bug fixing to proactive error detection. Instead of hiding discrepancies or forcing accounts to balance, it exposes problems early and provides tools to investigate and fix them properly with full audit trails.
