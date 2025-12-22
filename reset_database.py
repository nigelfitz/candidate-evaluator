import psycopg2
import os

# IMPORTANT: Never hardcode database credentials in files committed to git!
# Use environment variables instead
database_url = os.environ.get('DATABASE_URL')

if not database_url:
    print("❌ ERROR: DATABASE_URL environment variable not set")
    print("Set it using: railway run python reset_database.py")
    exit(1)

print("Connecting to postgres database...")
# Connect to 'postgres' database to drop and recreate 'railway'
conn_url_parts = database_url.rsplit('/', 1)
postgres_url = conn_url_parts[0] + '/postgres'
conn = psycopg2.connect(postgres_url)
conn.autocommit = True
cursor = conn.cursor()

print("Terminating existing connections to railway database...")
cursor.execute("""
    SELECT pg_terminate_backend(pg_stat_activity.pid)
    FROM pg_stat_activity
    WHERE pg_stat_activity.datname = 'railway'
    AND pid <> pg_backend_pid();
""")

print("Dropping railway database...")
cursor.execute("DROP DATABASE IF EXISTS railway;")

print("Creating railway database...")
cursor.execute("CREATE DATABASE railway;")

print("✅ Database reset successfully!")

cursor.close()
conn.close()
