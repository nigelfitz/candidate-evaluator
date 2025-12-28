"""
Production-Grade AI Pipeline for Candidate Scoring
Replaces semantic similarity with pure AI evaluation
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
from config import Config


# ============================================
# AI Model Configuration (Admin-Configurable)
# ============================================
RANKER_AGENT = 'gpt-4o-mini'    # Fast, cost-effective for bulk scoring
INSIGHT_AGENT = 'gpt-4o'        # Premium model for deep analysis


@dataclass
class CriterionScore:
    """Single criterion evaluation result"""
    criterion: str
    score: int  # 0-100
    justification: str
    is_anchor: bool = False  # Binary requirement (degree, certification, etc.)


@dataclass
class CandidateEvaluation:
    """Complete evaluation for one candidate"""
    candidate_name: str
    overall_score: float  # 0-100 weighted average
    criterion_scores: List[CriterionScore]
    missing_anchors: List[str]  # Critical gaps


class AIService:
    """
    Production AI pipeline for candidate evaluation.
    Handles all AI interactions with configurable models.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.ranker_model = RANKER_AGENT
        self.insight_model = INSIGHT_AGENT
    
    
    # ============================================
    # PHASE 0: Criteria Extraction
    # ============================================
    async def extract_criteria(self, jd_text: str) -> List[Dict[str, Any]]:
        """
        Extract job criteria from JD using RANKER_AGENT.
        
        Returns list of criteria with metadata:
        [
            {
                "criterion": "Bachelor's degree in Computer Science",
                "category": "Qualifications",
                "is_anchor": True,  # Binary requirement
                "weight": 1.0
            },
            ...
        ]
        """
        system_prompt = """You are an expert recruiter. Extract all job requirements from the Job Description.

For each requirement, determine:
1. Is it an "anchor" (binary requirement like degree, certification, license)? These are mandatory.
2. What category does it belong to? (Skills, Experience, Qualifications, etc.)

Return JSON:
{
    "criteria": [
        {
            "criterion": "exact requirement text",
            "category": "category name",
            "is_anchor": true/false,
            "weight": 1.0
        }
    ]
}"""

        user_prompt = f"""Job Description:
{jd_text}

Extract all requirements and identify which are anchors (mandatory binary requirements)."""

        response = await self.client.chat.completions.create(
            model=self.ranker_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("criteria", [])
    
    
    # ============================================
    # PHASE 1: Global Ranking (The Engine)
    # ============================================
    async def score_candidate_single_criterion(
        self,
        resume_text: str,
        jd_text: str,
        criterion: str,
        is_anchor: bool
    ) -> CriterionScore:
        """
        Score ONE candidate against ONE criterion using RANKER_AGENT.
        
        This is the atomic scoring unit - called in parallel for all criteria.
        """
        system_prompt = """You are a senior recruiter scoring candidates. Evaluate the candidate's resume against a specific job requirement.

ANCHOR CRITERIA (degrees, certifications, licenses):
- If the candidate HAS the requirement: score 100
- If the candidate LACKS the requirement: score 0-20 (harsh penalty for missing binary requirement)

NON-ANCHOR CRITERIA:
- Score 0-100 based on evidence quality
- Multiple roles showing the skill = higher score
- Recent relevant experience = higher score
- Outdated/weak evidence = lower score

Return JSON:
{
    "score": integer (0-100),
    "justification": "1-sentence evidence-based reasoning"
}"""

        anchor_note = "\n\nIMPORTANT: This is an ANCHOR criterion (binary requirement). Score 100 if present, 0-20 if missing." if is_anchor else ""
        
        user_prompt = f"""Criterion: {criterion}{anchor_note}

Job Description Context:
{jd_text[:1000]}

Candidate Resume:
{resume_text}

Provide your evaluation."""

        response = await self.client.chat.completions.create(
            model=self.ranker_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return CriterionScore(
            criterion=criterion,
            score=max(0, min(100, int(result.get("score", 0)))),
            justification=result.get("justification", "No justification provided"),
            is_anchor=is_anchor
        )
    
    
    async def score_candidate_all_criteria(
        self,
        candidate_name: str,
        resume_text: str,
        jd_text: str,
        criteria: List[Dict[str, Any]],
        semaphore: asyncio.Semaphore
    ) -> CandidateEvaluation:
        """
        Score ONE candidate against ALL criteria in parallel.
        Uses semaphore to limit concurrent API calls.
        """
        async with semaphore:
            # Create tasks for all criteria
            tasks = [
                self.score_candidate_single_criterion(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    criterion=crit["criterion"],
                    is_anchor=crit.get("is_anchor", False)
                )
                for crit in criteria
            ]
            
            # Execute all criteria scoring in parallel
            criterion_scores = await asyncio.gather(*tasks)
            
            # Calculate overall score (weighted average)
            weights = [crit.get("weight", 1.0) for crit in criteria]
            weighted_scores = [
                score.score * weights[i] 
                for i, score in enumerate(criterion_scores)
            ]
            overall_score = sum(weighted_scores) / sum(weights) if sum(weights) > 0 else 0.0
            
            # Identify missing anchors
            missing_anchors = [
                score.criterion 
                for score in criterion_scores 
                if score.is_anchor and score.score < 50
            ]
            
            return CandidateEvaluation(
                candidate_name=candidate_name,
                overall_score=overall_score,
                criterion_scores=criterion_scores,
                missing_anchors=missing_anchors
            )
    
    
    # ============================================
    # PHASE 2: Deep Insights (The Upgrade)
    # ============================================
    async def generate_deep_insights(
        self,
        candidate_name: str,
        resume_text: str,
        jd_text: str,
        evaluation: CandidateEvaluation
    ) -> Dict[str, Any]:
        """
        Generate high-quality insights using INSIGHT_AGENT.
        Includes refined justifications, strengths, gaps, and interview questions.
        """
        # Build context from Phase 1 scores
        scores_context = "\n".join([
            f"- {score.criterion}: {score.score}/100 - {score.justification}"
            for score in evaluation.criterion_scores
        ])
        
        system_prompt = """You are a senior hiring manager conducting a deep candidate assessment.

Generate comprehensive insights:
1. Top Strengths (3-6 bullet points)
2. Key Gaps/Concerns (3-6 bullet points)
3. Overall Assessment (2-4 sentences)
4. Refined Justifications (rewrite the draft justifications with more polish and evidence)
5. Suggested Interview Questions (3-5 questions to probe key areas)

Return JSON:
{
    "top": ["strength 1", "strength 2", ...],
    "gaps": ["gap 1", "gap 2", ...],
    "notes": "overall assessment",
    "justifications": {"criterion": "polished justification", ...},
    "interview_questions": ["question 1", "question 2", ...]
}"""

        user_prompt = f"""Candidate: {candidate_name}
Overall Score: {evaluation.overall_score:.1f}/100

Job Description:
{jd_text}

Resume:
{resume_text}

Phase 1 Draft Scores:
{scores_context}

Provide deep insights with refined justifications."""

        response = await self.client.chat.completions.create(
            model=self.insight_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result


# ============================================
# Orchestration: Async Pipeline with Progress Tracking
# ============================================
async def run_global_ranking(
    candidates: List[Tuple[str, str]],  # [(name, resume_text), ...]
    jd_text: str,
    criteria: List[Dict[str, Any]],
    progress_callback: Optional[callable] = None
) -> List[CandidateEvaluation]:
    """
    PHASE 1 ORCHESTRATOR: Score all candidates in parallel with progress tracking.
    
    Args:
        candidates: List of (name, resume_text) tuples
        jd_text: Full job description
        criteria: Extracted criteria with anchor flags
        progress_callback: Optional callback(completed, total) for progress updates
    
    Returns:
        List of CandidateEvaluation objects sorted by overall_score
    """
    ai_service = AIService()
    semaphore = asyncio.Semaphore(25)  # Limit concurrent API calls
    
    # Create tasks for all candidates
    tasks = []
    for candidate_name, resume_text in candidates:
        task = ai_service.score_candidate_all_criteria(
            candidate_name=candidate_name,
            resume_text=resume_text,
            jd_text=jd_text,
            criteria=criteria,
            semaphore=semaphore
        )
        tasks.append(task)
    
    # Execute with progress tracking
    results = []
    for i, task in enumerate(asyncio.as_completed(tasks)):
        evaluation = await task
        results.append(evaluation)
        
        # Progress callback for database update
        if progress_callback:
            progress_callback(i + 1, len(tasks))
    
    # Sort by overall score (descending)
    results.sort(key=lambda x: x.overall_score, reverse=True)
    
    return results


async def run_deep_insights(
    candidates: List[Tuple[str, str]],  # [(name, resume_text), ...]
    jd_text: str,
    evaluations: List[CandidateEvaluation],
    top_n: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    PHASE 2 ORCHESTRATOR: Generate deep insights for top N candidates.
    
    Args:
        candidates: List of (name, resume_text) tuples
        jd_text: Full job description
        evaluations: Phase 1 results
        top_n: Number of top candidates to analyze
    
    Returns:
        Dict mapping candidate_name -> deep_insights
    """
    ai_service = AIService()
    
    # Get top N candidates
    top_evaluations = evaluations[:top_n]
    
    # Create resume lookup
    resume_map = {name: text for name, text in candidates}
    
    # Generate insights in parallel
    tasks = [
        ai_service.generate_deep_insights(
            candidate_name=eval.candidate_name,
            resume_text=resume_map[eval.candidate_name],
            jd_text=jd_text,
            evaluation=eval
        )
        for eval in top_evaluations
    ]
    
    insights_list = await asyncio.gather(*tasks)
    
    # Build results dict
    insights_data = {
        eval.candidate_name: insights
        for eval, insights in zip(top_evaluations, insights_list)
    }
    
    return insights_data
