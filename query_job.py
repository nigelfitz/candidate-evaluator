import sqlite3
import json

conn = sqlite3.connect('flask_app/instance/candidate_evaluator.db')
cursor = conn.cursor()

# Find Job #0104
cursor.execute("SELECT job_id, job_title, created_at FROM jobs WHERE job_id LIKE '%0104%' ORDER BY created_at DESC LIMIT 1")
job = cursor.fetchone()

if job:
    job_id, job_title, created_at = job
    print(f"\nJob ID: {job_id}")
    print(f"Job Title: {job_title}")
    print(f"Created: {created_at}")
    
    # Get candidate named Nigel
    cursor.execute("""
        SELECT c.candidate_id, c.name, c.overall_score 
        FROM candidates c 
        WHERE c.job_id = ? AND c.name LIKE '%Nigel%'
        ORDER BY c.overall_score DESC
    """, (job_id,))
    
    candidate = cursor.fetchone()
    if candidate:
        cand_id, name, overall_score = candidate
        print(f"\nCandidate: {name}")
        print(f"Overall Score: {overall_score}")
        
        # Get all scores with justifications
        cursor.execute("""
            SELECT criterion, score, justification 
            FROM scores 
            WHERE candidate_id = ?
            ORDER BY score DESC
        """, (cand_id,))
        
        scores = cursor.fetchall()
        print(f"\n=== INDIVIDUAL SCORES ({len(scores)} criteria) ===")
        for criterion, score, justification in scores[:10]:  # Show top 10
            print(f"\n{criterion}: {score}/100")
            print(f"  → {justification}")
        
        # Get insights
        cursor.execute("""
            SELECT top_strengths, gaps_risks, overall_notes 
            FROM insights 
            WHERE candidate_id = ?
        """, (cand_id,))
        
        insight = cursor.fetchone()
        if insight:
            top_strengths, gaps_risks, overall_notes = insight
            
            print("\n=== TOP STRENGTHS ===")
            if top_strengths:
                strengths = json.loads(top_strengths)
                for i, s in enumerate(strengths, 1):
                    print(f"{i}. {s}")
            
            print("\n=== GAPS/RISKS ===")
            if gaps_risks:
                gaps = json.loads(gaps_risks)
                for i, g in enumerate(gaps, 1):
                    print(f"{i}. {g}")
            
            print("\n=== OVERALL NOTES ===")
            print(overall_notes)
    else:
        print("\nNo candidate named Nigel found in this job")
else:
    print("Job #0104 not found")

conn.close()
