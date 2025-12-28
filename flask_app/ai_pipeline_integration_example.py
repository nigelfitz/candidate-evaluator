"""
Example Integration: How to use the new AI Pipeline in Flask routes
This shows the "pipes" - how data flows through the system
"""

# Example Flask route integration (pseudocode)

from ai_service import run_global_ranking, run_deep_insights
from database import Analysis, db
import asyncio


def analyze_candidates_route():
    """
    Main analysis route showing the new pipeline flow
    """
    
    # ==========================================
    # PHASE 0: Criteria already extracted and edited by user
    # ==========================================
    criteria = [
        {"criterion": "Bachelor's degree", "is_anchor": True, "weight": 1.0, "category": "Qualifications"},
        {"criterion": "5+ years Python", "is_anchor": False, "weight": 1.0, "category": "Experience"},
        # ... more criteria
    ]
    
    # ==========================================
    # Prepare candidate data
    # ==========================================
    candidates = [
        ("John Doe", "full resume text here..."),
        ("Jane Smith", "full resume text here..."),
        # ... up to 200 candidates
    ]
    
    jd_text = "Full job description text..."
    
    
    # ==========================================
    # PHASE 1: Global Ranking (All Candidates)
    # ==========================================
    print("Phase 1: Ranking all candidates...")
    
    # Progress tracking callback
    def update_progress(completed, total):
        """Update database with progress for frontend"""
        # In real implementation: update Analysis record in database
        print(f"Progress: {completed}/{total} candidates scored")
        # analysis.resumes_processed = completed
        # db.session.commit()
    
    # Run async pipeline
    evaluations = asyncio.run(
        run_global_ranking(
            candidates=candidates,
            jd_text=jd_text,
            criteria=criteria,
            progress_callback=update_progress
        )
    )
    
    # Results: evaluations is a sorted list of CandidateEvaluation objects
    # Each contains:
    #   - candidate_name
    #   - overall_score (0-100)
    #   - criterion_scores (list of CriterionScore objects)
    #   - missing_anchors (list of critical gaps)
    
    print(f"Phase 1 complete. Top candidate: {evaluations[0].candidate_name} ({evaluations[0].overall_score:.1f})")
    
    
    # ==========================================
    # PHASE 2: Deep Insights (Top N Only)
    # ==========================================
    num_insights = 5  # User choice: top 3, 5, 10, or all
    
    print(f"Phase 2: Generating deep insights for top {num_insights}...")
    
    insights_data = asyncio.run(
        run_deep_insights(
            candidates=candidates,
            jd_text=jd_text,
            evaluations=evaluations,
            top_n=num_insights
        )
    )
    
    # Results: insights_data is a dict mapping candidate_name -> insights
    # Each insights dict contains:
    #   - top: [strengths]
    #   - gaps: [concerns]
    #   - notes: overall assessment
    #   - justifications: {criterion: refined_justification}
    #   - interview_questions: [questions]
    
    
    # ==========================================
    # Save to Database
    # ==========================================
    # Convert evaluations to DataFrame format for existing UI
    import pandas as pd
    
    coverage_records = []
    for eval in evaluations:
        row = {"Candidate": eval.candidate_name}
        
        # Add criterion scores
        for score in eval.criterion_scores:
            row[score.criterion] = score.score / 100.0  # Convert to 0-1 for compatibility
        
        row["Overall"] = eval.overall_score / 100.0
        coverage_records.append(row)
    
    coverage = pd.DataFrame(coverage_records)
    
    # Save evidence map (for UI compatibility)
    evidence_map = {}
    for eval in evaluations:
        for score in eval.criterion_scores:
            # Store justification as evidence
            evidence_map[(eval.candidate_name, score.criterion)] = (
                score.justification,  # Use justification as "snippet"
                score.score / 100.0,  # score
                1  # density count (not applicable in new system)
            )
    
    # Create Analysis record
    analysis = Analysis(
        user_id=current_user.id,
        job_title=job_title,
        job_description_text=jd_text[:5000],
        num_candidates=len(candidates),
        num_criteria=len(criteria),
        coverage_data=coverage.to_json(orient='records'),
        insights_data=json.dumps(insights_data),
        evidence_data=json.dumps({f"{k[0]}|||{k[1]}": v for k, v in evidence_map.items()}),
        criteria_list=json.dumps(criteria),
        cost_usd=calculated_cost
    )
    
    db.session.add(analysis)
    db.session.commit()
    
    print("Analysis complete and saved!")
    
    return analysis.id


# ==========================================
# Key Differences from Old System
# ==========================================
"""
OLD SYSTEM (Semantic Similarity):
1. Chunk resumes into 1200-char pieces
2. Embed chunks with sentence-transformers
3. Compute cosine similarity for each criterion
4. Apply manual heuristics (non-linear curves, qualification boosts)
5. Generate GPT insights for top N

NEW SYSTEM (Pure AI Pipeline):
1. No chunking, no embeddings, no similarity math
2. AI scores entire resume for each criterion
3. Parallel processing with asyncio (25 concurrent calls)
4. Automatic anchor detection with harsh penalties
5. Two-tier: RANKER_AGENT for speed, INSIGHT_AGENT for quality

Benefits:
- More accurate: AI understands context better than cosine similarity
- Faster: Parallel processing (200 resumes in ~30 seconds)
- Maintainable: No brittle heuristics
- Scalable: Easy to add more sophisticated prompts
- Future-proof: Swap models via config variables
"""
