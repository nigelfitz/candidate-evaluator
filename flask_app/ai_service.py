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
# AI Model Configuration (from config/gpt_settings.json)
# ============================================

def load_ai_config() -> Dict[str, Any]:
    """Load AI agent configuration from gpt_settings.json"""
    settings_path = os.path.join(os.path.dirname(__file__), 'config', 'gpt_settings.json')
    with open(settings_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load configuration on module import
_ai_config = load_ai_config()

RANKER_AGENT = _ai_config['ranker_model']['value']          # Fast, cost-effective for bulk scoring
INSIGHT_AGENT = _ai_config['insight_model']['value']        # Premium model for deep analysis
RANKER_TEMPERATURE = _ai_config['ranker_temperature']['value']  # Consistency for scoring
INSIGHT_TEMPERATURE = _ai_config['insight_temperature']['value']  # Readability for insights
RANKER_MAX_TOKENS = _ai_config['advanced_api_settings']['ranker_max_tokens']['value']
INSIGHT_MAX_TOKENS = _ai_config['advanced_api_settings']['insight_max_tokens']['value']
PRESENCE_PENALTY = _ai_config['advanced_api_settings']['presence_penalty']['value']
FREQUENCY_PENALTY = _ai_config['advanced_api_settings']['frequency_penalty']['value']


@dataclass
class CriterionScore:
    """Single criterion evaluation result"""
    criterion: str
    score: int  # 0-100
    justification: str
    raw_evidence: str  # Verbatim quotes from resume supporting the score
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
            temperature=RANKER_TEMPERATURE,
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

CRITICAL: You must extract verbatim quotes from the resume as proof.

Return JSON:
{
    "score": integer (0-100),
    "justification": "1-sentence reasoning explaining your score",
    "raw_evidence": "Direct verbatim quotes from resume that prove this criterion. If evidence spans multiple jobs, include quotes from each separated by double line breaks."
}

Example raw_evidence for multi-job evidence:
"Advanced Excel skills including VLOOKUP, pivot tables, and macros (Finance Manager, 2020-2023)

Created complex financial models using Excel (Senior Analyst, 2018-2020)"

If no direct quote exists, extract the most relevant sentence that implies the skill."""

        anchor_note = "\n\nIMPORTANT: This is an ANCHOR criterion (binary requirement). Score 100 if present, 0-20 if missing." if is_anchor else ""
        
        user_prompt = f"""Criterion: {criterion}{anchor_note}

Job Description Context:
{jd_text[:1000]}

Candidate Resume:
{resume_text}

Provide your evaluation."""

        # Retry logic for rate limits
        max_retries = 5
        retry_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.ranker_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=RANKER_TEMPERATURE,
                    max_tokens=RANKER_MAX_TOKENS,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                return CriterionScore(
                    criterion=criterion,
                    score=max(0, min(100, int(result.get("score", 0)))),
                    justification=result.get("justification", "No justification provided"),
                    raw_evidence=result.get("raw_evidence", "No evidence extracted"),
                    is_anchor=is_anchor
                )
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Rate limit hit for {criterion}. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"ERROR: Max retries exceeded for {criterion}: {error_str}")
                        return CriterionScore(
                            criterion=criterion,
                            score=0,
                            justification=f"Rate limit error after {max_retries} retries",
                            raw_evidence="Unable to extract evidence due to rate limiting",
                            is_anchor=is_anchor
                        )
                else:
                    # Non-rate-limit error - fail immediately
                    print(f"ERROR scoring {criterion}: {error_str}")
                    return CriterionScore(
                        criterion=criterion,
                        score=0,
                        justification=f"Evaluation error: {error_str[:100]}",
                        raw_evidence="Unable to extract evidence due to evaluation error",
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
4. Refined Justifications - CRITICAL: Rewrite ALL draft justifications with more polish and evidence. You MUST provide a refined justification for EVERY criterion listed in Phase 1 Draft Scores.
5. Suggested Interview Questions (3-5 questions to probe key areas)

Return ONLY valid JSON in this exact format:
{
    "top": ["strength 1", "strength 2", ...],
    "gaps": ["gap 1", "gap 2", ...],
    "notes": "overall assessment",
    "justifications": {"criterion name": "polished justification", ...},
    "interview_questions": ["question 1", "question 2", ...]
}

CRITICAL JSON RULES:
- Escape all quotes inside strings using backslash: "He said \\"hello\\""
- Escape all backslashes: use \\\\ for single backslash
- Do not use smart quotes or unicode quotes - use only straight quotes
- Ensure all strings are properly terminated with closing quotes
- Do not include any text before or after the JSON object
- The "justifications" object must contain an entry for EVERY criterion from Phase 1, using the exact criterion name as the key."""

        user_prompt = f"""Candidate: {candidate_name}
Overall Score: {evaluation.overall_score:.1f}/100

Job Description:
{jd_text}

Resume:
{resume_text}

Phase 1 Draft Scores:
{scores_context}

Provide deep insights with refined justifications."""

        # Retry logic with exponential backoff for rate limit handling
        max_retries = 5
        retry_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.insight_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=INSIGHT_TEMPERATURE,
                    max_tokens=INSIGHT_MAX_TOKENS,
                    presence_penalty=PRESENCE_PENALTY,
                    frequency_penalty=FREQUENCY_PENALTY,
                    response_format={"type": "json_object"}
                )
                
                # Parse JSON with aggressive error recovery
                raw_content = response.choices[0].message.content
                
                try:
                    result = json.loads(raw_content)
                except json.JSONDecodeError as e:
                    # JSON parsing failed - log details and try sanitization
                    print(f"WARNING: JSON parse error for {candidate_name}: {str(e)}")
                    print(f"WARNING: Error at line {e.lineno} column {e.colno} (char {e.pos})")
                    if e.pos and e.pos < len(raw_content):
                        start = max(0, e.pos - 150)
                        end = min(len(raw_content), e.pos + 150)
                        print(f"WARNING: Context around error: ...{raw_content[start:end]}...")
                    
                    # Try sanitization strategies
                    sanitized = raw_content
                    
                    # Strategy 1: Fix smart quotes
                    sanitized = sanitized.replace('\u2018', "'").replace('\u2019', "'")
                    sanitized = sanitized.replace('\u201c', '"').replace('\u201d', '"')
                    sanitized = sanitized.replace('\u201a', "'").replace('\u201e', '"')
                    
                    # Strategy 2: Extract JSON object if GPT added extra text
                    first_brace = sanitized.find('{')
                    last_brace = sanitized.rfind('}')
                    
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        json_only = sanitized[first_brace:last_brace+1]
                        
                        try:
                            result = json.loads(json_only)
                            print(f"SUCCESS: JSON parsing recovered by extracting object")
                        except json.JSONDecodeError as e2:
                            # Strategy 3: More aggressive sanitization - fix unterminated strings
                            print(f"WARNING: Trying aggressive string termination fix...")
                            
                            # Try to detect and fix unterminated strings by ensuring proper closure
                            import re
                            
                            # Replace problematic patterns that cause unterminated strings
                            fixed = json_only
                            
                            # Fix newlines inside strings (common GPT issue)
                            # Find strings and replace literal newlines with \n
                            def fix_string_content(match):
                                quote = match.group(1)  # Opening quote
                                content = match.group(2)  # String content
                                # Replace literal newlines with escaped version
                                content = content.replace('\n', '\\n')
                                content = content.replace('\r', '\\r')
                                content = content.replace('\t', '\\t')
                                return f'{quote}{content}{quote}'
                            
                            # This regex finds quoted strings and fixes newlines inside them
                            fixed = re.sub(r'(["\'])([^"\'\}]*?)\1', fix_string_content, fixed, flags=re.DOTALL)
                            
                            try:
                                result = json.loads(fixed)
                                print(f"SUCCESS: JSON parsing recovered after newline fix")
                            except json.JSONDecodeError as e3:
                                # Final fallback: Log everything and RAISE exception
                                # We do NOT want to return fake data - fail the analysis
                                print(f"CRITICAL: All JSON recovery strategies failed for {candidate_name}")
                                print(f"FULL RESPONSE ({len(raw_content)} chars):")
                                print(raw_content)
                                print(f"Original error: {str(e)}")
                                print(f"After sanitization: {str(e2)}")
                                print(f"After newline fix: {str(e3)}")
                                # RAISE the exception instead of returning error message
                                raise Exception(f"Failed to parse GPT response as JSON for {candidate_name}. {str(e)}") from e
                    else:
                        # No JSON object found at all
                        print(f"CRITICAL: No JSON object found in response")
                        print(f"FULL RESPONSE: {raw_content[:500]}...")
                        raise Exception(f"No valid JSON object found in GPT response for {candidate_name}")
                
                return result
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                        print(f"Rate limit hit for insights generation ({candidate_name}). Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # All retries exhausted - RAISE exception instead of returning error message
                        error_msg = f"Rate limit error after {max_retries} retries for {candidate_name}"
                        print(f"CRITICAL: {error_msg}")
                        raise Exception(error_msg) from e
                else:
                    # Non-rate-limit error - RAISE immediately instead of returning error message
                    print(f"CRITICAL: Insights generation failed for {candidate_name}")
                    print(f"ERROR: {error_str}")
                    # Re-raise the exception to trigger analysis failure and rollback
                    raise


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
    
    # Create tasks for all candidates with error handling wrapper
    async def score_with_error_handling(candidate_name: str, resume_text: str):
        """Wrap scoring with error handling to prevent one bad candidate from killing entire batch"""
        try:
            return await ai_service.score_candidate_all_criteria(
                candidate_name=candidate_name,
                resume_text=resume_text,
                jd_text=jd_text,
                criteria=criteria,
                semaphore=semaphore
            )
        except Exception as e:
            print(f"ERROR scoring candidate {candidate_name}: {str(e)}")
            # Return a failed evaluation with zero scores
            from dataclasses import dataclass, field
            from typing import List as TypeList
            
            # Create zero-score evaluation for failed candidate
            zero_scores = [
                CriterionScore(
                    criterion=crit["criterion"],
                    score=0.0,
                    justification=f"Analysis failed: Unable to process resume (possibly corrupted, scanned image, or invalid format)",
                    raw_evidence="[Unable to extract evidence - processing error]",
                    is_anchor=crit.get("is_anchor", False)
                )
                for crit in criteria
            ]
            
            return CandidateEvaluation(
                candidate_name=candidate_name,
                overall_score=0.0,
                criterion_scores=zero_scores,
                missing_anchors=[score.criterion for score in zero_scores if score.is_anchor]
            )
    
    tasks = [
        score_with_error_handling(candidate_name, resume_text)
        for candidate_name, resume_text in candidates
    ]
    
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
