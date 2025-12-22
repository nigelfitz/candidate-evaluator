# Database Migration: Remove deleted_at Column

**Date:** December 22, 2025
**Migration Type:** Schema Change - Remove Column
**Impact:** Medium (requires database update)

## Changes Made

### Removed from Analysis Model:
- `deleted_at` column (DateTime, nullable, indexed)

### Reason:
Switching from soft delete to hard delete for better data privacy compliance.

## SQL Migration Commands

### For PostgreSQL (Production - Railway):
```sql
-- Remove the deleted_at column from analyses table
ALTER TABLE analyses DROP COLUMN IF EXISTS deleted_at;
```

### For SQLite (Local Development):
```sql
-- SQLite doesn't support DROP COLUMN easily, so we need to:
-- 1. Create new table without deleted_at
-- 2. Copy data
-- 3. Drop old table
-- 4. Rename new table

-- However, SQLite is local only, so easiest is to delete the .db file and recreate

-- OR use SQLAlchemy to recreate:
-- Delete candidate_evaluator.db file and restart Flask to auto-create new schema
```

## How to Apply

### Production (Railway):
1. Connect to Railway PostgreSQL:
   ```bash
   railway run psql $DATABASE_URL
   ```

2. Run migration:
   ```sql
   ALTER TABLE analyses DROP COLUMN IF EXISTS deleted_at;
   ```

3. Verify:
   ```sql
   \d analyses
   ```

### Local Development:
Option 1 (Recommended): Delete and recreate database
```bash
# Delete the SQLite database
rm candidate_evaluator.db

# Restart Flask - it will auto-create the new schema
python -m flask run
```

Option 2: Use Flask-Migrate (if installed)
```bash
flask db migrate -m "Remove deleted_at from Analysis model"
flask db upgrade
```

## Code Changes

- `database.py`: Removed `deleted_at` field from Analysis model
- `app.py`: Removed all `deleted_at.is_(None)` filters
- `app.py`: Added `/delete-analysis/<id>` route for hard deletes
- `job_history.html`: Added delete button
- `security.html`: Updated to reflect proper hard delete policy

## Transaction History Preservation

The `Transaction` model has `analysis_id` with `ondelete='SET NULL'`:
```python
analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id', ondelete='SET NULL'))
```

This means when an analysis is deleted:
- ✅ Transaction record is preserved
- ✅ User's balance history remains intact
- ✅ `analysis_id` is set to NULL (orphaned but preserved)
- ✅ Transaction description still shows job title

## Testing Checklist

- [ ] Apply migration to local database
- [ ] Test deleting an analysis
- [ ] Verify transaction history is preserved
- [ ] Verify candidate files are cascade deleted
- [ ] Apply migration to production
- [ ] Test on production with test account
