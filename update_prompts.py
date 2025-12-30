import json

# Read existing prompts.json
with open('flask_app/config/prompts.json', 'r', encoding='utf-8') as f:
    prompts = json.load(f)

# Add RANKER scoring prompt section
prompts["ranker_scoring"] = {
    "name": "RANKER Agent - Bulk Candidate Scoring",
    "description": "Scores individual candidates against specific criteria (0-100). Called once per candidate × criterion combination.",
    "developer_notes": {
        "purpose": "This is your HIGHEST-VOLUME operation. For 50 candidates × 20 criteria, this runs 1,000 times per analysis. Keep it fast and consistent.",
        "model_used": "RANKER_AGENT (gpt-4o-mini by default)",
        "temperature": "0.1 (very low for consistency - same resume+criterion should always get same score)",
        "critical_features": [
            "Returns 0-100 integer score",
            "Provides 1-sentence justification explaining the score",
            "Extracts verbatim resume quotes as raw_evidence (proof of the score)",
            "Handles 'anchor' criteria (binary requirements like degrees) with harsh penalties if missing"
        ],
        "cost_per_call": "~$0.0003 with gpt-4o-mini (500 tokens in, 100 tokens out)",
        "pro_tip": "If scores seem random/inconsistent, lower ranker_temperature to 0.05. If evidence seems incomplete, increase ranker_max_tokens to 400."
    },
    "system_prompt": {
      "value": """You are a senior recruiter scoring candidates. Evaluate the candidate's resume against a specific job requirement.

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

If no direct quote exists, extract the most relevant sentence that implies the skill.""",
      "label": "System Prompt",
      "description": "Defines the RANKER's role and scoring methodology",
      "variables_used": "None - this is the foundational instruction"
    },
    "user_prompt_template": {
      "value": """Criterion: {criterion}{anchor_note}

Job Description Context:
{jd_text}

Candidate Resume:
{resume_text}

Provide your evaluation.""",
      "label": "User Prompt Template",
      "description": "The actual scoring request sent per candidate-criterion pair",
      "placeholders": [
        "{criterion}",
        "{anchor_note}",
        "{jd_text}",
        "{resume_text}"
      ],
      "variable_details": {
        "{criterion}": "The specific requirement being scored (e.g., 'Python programming experience')",
        "{anchor_note}": "Empty string for normal criteria. For anchors, adds: 'IMPORTANT: This is an ANCHOR criterion (binary requirement). Score 100 if present, 0-20 if missing.'",
        "{jd_text}": "Truncated job description (first 1000 chars) for context",
        "{resume_text}": "Full candidate resume text"
      },
      "CRITICAL_REMINDER": "⚠️ The response must match the exact JSON schema defined in system_prompt. Changing field names will break ai_service.py parsing."
    }
}

# Add INSIGHT agent prompt section (move from candidate_insights with better documentation)
prompts["insight_generation"] = {
    "name": "INSIGHT Agent - Deep Candidate Analysis",
    "description": "Generates comprehensive insights (strengths, gaps, notes, justifications) for selected candidates only.",
    "developer_notes": {
        "purpose": "This is your LOW-VOLUME, HIGH-VALUE operation. Only runs for candidates you select for 'Deep Insights' (typically top 3-10).",
        "model_used": "INSIGHT_AGENT (gpt-4o by default)",
        "temperature": "0.4 (moderate for natural, readable phrasing)",
        "critical_features": [
            "Generates 3-6 strength bullet points tied to specific evidence",
            "Generates 3-6 gap/risk bullet points with constructive feedback",
            "Provides 2-4 sentence overall assessment",
            "Refines ALL criterion justifications from Phase 1 for polished, professional phrasing",
            "Optionally generates interview questions (if requested)"
        ],
        "cost_per_call": "~$0.03 with gpt-4o (3000 tokens in, 800 tokens out)",
        "pro_tip": "If insights feel generic/robotic, increase insight_temperature to 0.5. If too creative/inconsistent, lower to 0.3. Use presence_penalty=0.4 to avoid repetitive phrasing across candidates."
    },
    "system_prompt": {
      "value": """You are a senior hiring analyst. You receive Phase 1 draft scores and must:
1. Generate high-quality strengths (3-6 bullets) tied to specific evidence
2. Identify gaps/concerns (3-6 bullets) with constructive feedback
3. Provide overall assessment notes (2-4 sentences)
4. REFINE all criterion justifications to be polished and professional

Return JSON:
{
    "top": ["strength 1", "strength 2", ...],
    "gaps": ["gap 1", "gap 2", ...],
    "notes": "overall assessment",
    "justifications": {"criterion name": "polished justification", ...},
    "interview_questions": ["question 1", "question 2", ...]
}

IMPORTANT: The "justifications" object must contain an entry for EVERY criterion from Phase 1, using the exact criterion name as the key.""",
      "label": "System Prompt",
      "description": "Defines the INSIGHT agent's role and output format",
      "variables_used": "None - this is the foundational instruction"
    },
    "user_prompt_template": {
      "value": """Candidate: {candidate_name}
Overall Score: {overall_score}/100

Job Description:
{jd_text}

Resume:
{resume_text}

Phase 1 Draft Scores:
{scores_context}

Provide deep insights with refined justifications.""",
      "label": "User Prompt Template",
      "description": "The deep analysis request for selected candidates",
      "placeholders": [
        "{candidate_name}",
        "{overall_score}",
        "{jd_text}",
        "{resume_text}",
        "{scores_context}"
      ],
      "variable_details": {
        "{candidate_name}": "Full name of the candidate being analyzed",
        "{overall_score}": "Weighted average score (0-100) from Phase 1 RANKER scoring",
        "{jd_text}": "Full or truncated job description text",
        "{resume_text}": "Full or truncated candidate resume text",
        "{scores_context}": "Formatted list of all Phase 1 scores with draft justifications, e.g.:\n- Python Programming (85/100): Strong evidence across multiple projects...\n- Team Leadership (45/100): Limited management experience..."
      },
      "CRITICAL_REMINDER": "⚠️ The response must match the exact JSON schema. The code expects fields: 'top'/'top_strengths'/'strengths', 'gaps'/'gaps_risks'/'concerns', 'notes'/'overall_notes'/'assessment', 'justifications' (object), 'interview_questions' (array)."
    }
}

# Keep existing jd_extraction and candidate_insights for backward compatibility but mark as legacy
prompts["candidate_insights"]["_LEGACY_NOTICE"] = "This section is kept for backward compatibility with old analysis.py code. New code should use 'insight_generation' section above."

# Update metadata
prompts["_metadata"]["last_updated"] = "2025-12-30 00:00:00 UTC"
prompts["_metadata"]["version"] = "2.0"
prompts["_metadata"]["changelog"].append("v2.0 (2025-12-30): Added ranker_scoring and insight_generation sections for two-agent architecture. Added comprehensive developer notes and pro-tips.")

# Write back to file
with open('flask_app/config/prompts.json', 'w', encoding='utf-8') as f:
    json.dump(prompts, f, indent=2, ensure_ascii=False)

print("✅ prompts.json updated with RANKER and INSIGHT prompt sections!")
print("📋 Added developer notes, variable documentation, and pro-tips for each agent.")
