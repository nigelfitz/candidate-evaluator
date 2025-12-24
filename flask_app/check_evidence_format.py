import sqlite3
import json

conn = sqlite3.connect('instance/candidate_evaluator.db')
cursor = conn.cursor()

cursor.execute('SELECT id, created_at, evidence_data FROM analyses WHERE id = 21')
row = cursor.fetchone()

if row:
    analysis_id, created_at, evidence_data_json = row
    print(f'Analysis ID: {analysis_id}')
    print(f'Created: {created_at}')
    
    evidence = json.loads(evidence_data_json)
    print(f'\nTotal evidence keys: {len(evidence)}')
    
    # Check key formats
    pipe_keys = [k for k in evidence.keys() if '|||' in k]
    tuple_keys = [k for k in evidence.keys() if '|||' not in k]
    
    print(f'Pipe format keys (|||): {len(pipe_keys)}')
    print(f'Tuple format keys: {len(tuple_keys)}')
    
    print(f'\nSample pipe keys (first 3):')
    for k in pipe_keys[:3]:
        print(f'  {k}')
    
    print(f'\nSample tuple keys (first 3):')
    for k in tuple_keys[:3]:
        print(f'  {k}')
else:
    print('Analysis 21 not found')

conn.close()
