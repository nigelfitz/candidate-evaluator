#!/bin/bash
# Railway startup script - runs migration then starts the app

echo "🔄 Running database migration..."
python add_welcome_bonus_column.py

if [ $? -eq 0 ]; then
    echo "✅ Migration completed or already applied"
else
    echo "⚠️  Migration had issues but continuing startup..."
fi

echo "🚀 Starting Flask app..."
exec gunicorn --bind 0.0.0.0:8080 --workers 2 --timeout 600 "app:create_app()"
