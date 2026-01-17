import sqlite3

conn = sqlite3.connect('candidate_evaluator.db')
cursor = conn.cursor()

print('Tables in database:')
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
for table in tables:
    print(f'  - {table[0]}')
    cursor.execute(f'PRAGMA table_info({table[0]})')
    cols = cursor.fetchall()
    cost_cols = [col[1] for col in cols if 'cost' in col[1].lower()]
    if cost_cols:
        print(f'    Cost columns: {cost_cols}')

conn.close()
