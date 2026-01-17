import sqlite3

conn = sqlite3.connect('instance/candidate_evaluator.db')
cursor = conn.cursor()
cursor.execute('PRAGMA table_info(analyses)')
cols = [row[1] for row in cursor.fetchall()]
print('Cost tracking columns:')
print('  - openai_cost_usd:', 'openai_cost_usd' in cols)
print('  - ranker_cost_usd:', 'ranker_cost_usd' in cols)
print('  - insight_cost_usd:', 'insight_cost_usd' in cols)

cursor.execute('SELECT id, openai_cost_usd, ranker_cost_usd, insight_cost_usd, num_candidates FROM analyses ORDER BY id DESC LIMIT 1')
row = cursor.fetchone()
print(f'\nLast job (ID {row[0]}):')
print(f'  Resumes: {row[4]}')
print(f'  openai_cost_usd: ${row[1] if row[1] else 0:.4f}')
print(f'  ranker_cost_usd: ${row[2] if row[2] else 0:.4f}')
print(f'  insight_cost_usd: ${row[3] if row[3] else 0:.4f}')
conn.close()
