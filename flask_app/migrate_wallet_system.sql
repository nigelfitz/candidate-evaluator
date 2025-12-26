-- Migration: Add welcome_bonus_claimed field to User table
-- Run this on Railway using: railway run psql $DATABASE_URL -f migrate_wallet_system.sql

-- Check if column exists first
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name='users' AND column_name='welcome_bonus_claimed'
    ) THEN
        -- Add the column with default FALSE
        ALTER TABLE users 
        ADD COLUMN welcome_bonus_claimed BOOLEAN DEFAULT FALSE NOT NULL;
        
        -- Mark all existing users as already having received their signup bonus
        -- (They already have their starting balance, so we don't want to double-credit them)
        UPDATE users 
        SET welcome_bonus_claimed = TRUE 
        WHERE created_at < NOW();
        
        RAISE NOTICE 'Successfully added welcome_bonus_claimed column to users table';
    ELSE
        RAISE NOTICE 'Column welcome_bonus_claimed already exists, skipping migration';
    END IF;
END
$$;
