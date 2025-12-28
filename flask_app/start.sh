#!/bin/bash
# Railway startup script - runs migrations then starts the app

echo "🔄 Running database migrations..."
python add_welcome_bonus_column.py
python migrate_add_suspension.py
python migrate_add_resumes_processed.py

if [ $? -eq 0 ]; then
    echo "✅ Migrations completed or already applied"
else
    echo "⚠️  Migrations had issues but continuing startup..."
fi

echo "🚀 Starting Flask app..."
exec gunicorn --bind 0.0.0.0:8080 --workers 2 --timeout 600 "app:create_app()"
