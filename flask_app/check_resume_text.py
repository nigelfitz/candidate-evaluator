"""Quick diagnostic to check what text was extracted from resumes"""
from database import db, Draft, DraftResume
from app import app

with app.app_context():
    # Get the most recent draft
    draft = Draft.query.order_by(Draft.created_at.desc()).first()
    
    if not draft:
        print("No drafts found")
    else:
        print(f"\nDraft ID: {draft.id}")
        print(f"Job Title: {draft.job_title}")
        print(f"\nResumes in this draft:")
        print("=" * 80)
        
        resumes = DraftResume.query.filter_by(draft_id=draft.id).all()
        for resume in resumes:
            text = resume.extracted_text or ""
            text_length = len(text)
            
            # Calculate whitespace ratio
            non_whitespace = len(text.strip().replace('\n', '').replace(' ', ''))
            whitespace_ratio = 1.0 - (non_whitespace / max(text_length, 1))
            
            print(f"\nCandidate: {resume.candidate_name}")
            print(f"Filename: {resume.file_name}")
            print(f"Total chars: {text_length}")
            print(f"Non-whitespace chars: {non_whitespace}")
            print(f"Whitespace ratio: {whitespace_ratio:.2%}")
            
            if text_length < 500:
                print(f"First 200 chars of extracted text:")
                print(repr(text[:200]))
            
            print("-" * 80)
