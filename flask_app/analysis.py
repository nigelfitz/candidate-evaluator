"""
Core analysis logic for candidate evaluation
Ported from Streamlit app
"""

import io
import os
import re
import hashlib
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime

# PDF/DOCX imports
try:
    import PyPDF2
    import pdfplumber
    from docx import Document
    import fitz  # PyMuPDF for layout-aware extraction
except ImportError as e:
    PyPDF2 = pdfplumber = Document = fitz = None

# OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    SentenceTransformer = torch = None

# ML
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = cosine_similarity = None


# -----------------------------
# Admin Configuration Loader
# -----------------------------
def load_gpt_settings():
    """
    Load GPT configuration settings from config/gpt_settings.json (v2.0 Two-Agent Architecture)
    
    This allows admin to adjust parameters without touching code:
    - RANKER model (bulk scoring - gpt-4o-mini recommended)
    - INSIGHT model (deep analysis - gpt-4o recommended)
    - Separate temperatures for each agent (RANKER: 0.1, INSIGHT: 0.4)
    - Advanced API settings (presence/frequency penalties, max tokens)
    - Score thresholds (High/Low cutoffs for UI display)
    
    Note: ai_service.py also loads this config independently for AI calls.
    This function provides settings for non-AI parts of analysis flow.
    
    Returns dict with all settings and their current values.
    """
    settings_path = os.path.join(os.path.dirname(__file__), 'config', 'gpt_settings.json')
    
    # Load settings from JSON file
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)
    
    # Return flattened dict with just the values for easy access (two-agent architecture)
    # Note: Most AI configuration is now handled by ai_service.py load_ai_config()
    return {
        # Two-agent model configuration
        'ranker_model': settings.get('ranker_model', {}).get('value', 'gpt-4o-mini'),
        'insight_model': settings.get('insight_model', {}).get('value', 'gpt-4o'),
        'ranker_temperature': settings.get('ranker_temperature', {}).get('value', 0.1),
        'insight_temperature': settings.get('insight_temperature', {}).get('value', 0.4),
        
        # Advanced API settings
        'presence_penalty': settings.get('advanced_api_settings', {}).get('presence_penalty', {}).get('value', 0.4),
        'frequency_penalty': settings.get('advanced_api_settings', {}).get('frequency_penalty', {}).get('value', 0.3),
        'ranker_max_tokens': settings.get('advanced_api_settings', {}).get('ranker_max_tokens', {}).get('value', 300),
        'insight_max_tokens': settings.get('advanced_api_settings', {}).get('insight_max_tokens', {}).get('value', 1000),
        
        # Text processing settings
        'candidate_text_chars': settings.get('text_processing', {}).get('candidate_text_chars', {}).get('value', 12000),
        'jd_text_chars': settings.get('text_processing', {}).get('jd_text_chars', {}).get('value', 5000),
        'evidence_snippet_chars': settings.get('text_processing', {}).get('evidence_snippet_chars', {}).get('value', 500),
        
        # Insight formatting
        'notes_length': settings.get('insight_formatting', {}).get('notes_length', {}).get('value', 'concise'),
        'insight_tone': settings.get('insight_formatting', {}).get('insight_tone', {}).get('value', 'professional'),
        
        # Report balance (min/max strengths and gaps)
        'min_strengths': settings.get('report_balance', {}).get('min_strengths', {}).get('value', 3),
        'max_strengths': settings.get('report_balance', {}).get('max_strengths', {}).get('value', 6),
        'min_gaps': settings.get('report_balance', {}).get('min_gaps', {}).get('value', 3),
        'max_gaps': settings.get('report_balance', {}).get('max_gaps', {}).get('value', 6),
        
        # Evidence depth
        'top_evidence_items': settings.get('evidence_depth', {}).get('top_evidence_items', {}).get('value', 5),
        
        # Evidence thresholds (for match quality classification)
        'strong_match_threshold': settings.get('evidence_thresholds', {}).get('strong_match_threshold', {}).get('value', 0.75),
        'good_match_threshold': settings.get('evidence_thresholds', {}).get('good_match_threshold', {}).get('value', 0.50),
        'moderate_match_threshold': settings.get('evidence_thresholds', {}).get('moderate_match_threshold', {}).get('value', 0.35),
        'weak_match_threshold': settings.get('evidence_thresholds', {}).get('weak_match_threshold', {}).get('value', 0.15),
        
        # Score thresholds
        'high_threshold': settings.get('score_thresholds', {}).get('high_threshold', {}).get('value', 0.75),
        'low_threshold': settings.get('score_thresholds', {}).get('low_threshold', {}).get('value', 0.35),
        
        # Legacy aliases (for backward compatibility)
        'model': settings.get('ranker_model', {}).get('value', 'gpt-4o-mini'),
        'temperature': settings.get('ranker_temperature', {}).get('value', 0.1),
        'max_tokens': settings.get('advanced_api_settings', {}).get('ranker_max_tokens', {}).get('value', 300),
    }


def load_prompts():
    """
    Load AI prompt templates from config/prompts.json
    
    This allows admin to adjust prompt wording and instructions without touching code:
    - JD extraction prompts (how criteria are extracted from job descriptions)
    - Candidate insights prompts (how strengths/gaps/notes are generated)
    
    Returns dict with all prompts and their current values.
    """
    prompts_path = os.path.join(os.path.dirname(__file__), 'config', 'prompts.json')
    with open(prompts_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Candidate:
    name: str
    file_name: str
    text: str
    hash: str
    raw_bytes: Optional[bytes] = None


@dataclass
class JD:
    file_name: str
    text: str
    hash: str
    raw_bytes: Optional[bytes] = None


@dataclass
class JDSections:
    job_title: str
    key_skills: List[str]
    responsibilities: List[str]
    qualifications: List[str]
    experience_required: List[str]


# -----------------------------
# Hash helpers
# -----------------------------
def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


# -----------------------------
# OpenAI helper
# -----------------------------
def get_openai_client():
    """Get OpenAI client if API key is set"""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    if not api_key or OpenAI is None:
        return None
    try:
        # Create client with just the API key - let OpenAI handle HTTP client
        client = OpenAI(api_key=api_key, timeout=60.0)
        return client
    except Exception as e:
        print(f"ERROR: Failed to create OpenAI client: {e}")
        return None


def call_llm_json(system_prompt: str, user_prompt: str, schema: Dict[str, Any], model: str = "gpt-4o") -> Dict[str, Any]:
    """Call OpenAI to return structured JSON"""
    client = get_openai_client()
    if client is None:
        return {"key_skills": [], "responsibilities": [], "qualifications": [], "experience_required": []}
    
    json_guard = (
        "Return ONLY valid JSON that conforms to the provided JSON Schema. "
        "Do not include backticks or any commentary."
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n\n{json_guard}\n\nSchema:\n{json.dumps(schema)}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        txt = resp.choices[0].message.content or "{}"
        return json.loads(txt)
    except Exception as e:
        print(f"GPT call failed: {e}")
        return {"key_skills": [], "responsibilities": [], "qualifications": [], "experience_required": []}


# -----------------------------
# File reading / extraction
# -----------------------------
def read_file_bytes(file_bytes: bytes, file_name: str) -> str:
    """Extract text from uploaded file (PDF, DOCX, TXT)"""
    lower = file_name.lower()
    
    # TXT
    if lower.endswith(".txt"):
        try:
            return file_bytes.decode("utf-8")
        except Exception:
            return file_bytes.decode("latin-1", errors="ignore")
    
    # DOCX
    if lower.endswith(".docx") and Document is not None:
        try:
            d = Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in d.paragraphs)
        except Exception:
            pass
    
    # PDF - use layout-aware extraction by default
    if lower.endswith(".pdf"):
        # Try layout-aware extraction first (PyMuPDF)
        try:
            text = extract_text_layout_aware(file_bytes, file_name)
            if text.strip():  # If we got text, use it
                return text
        except Exception:
            pass
        
        # Fallback to pdfplumber (better than PyPDF2)
        if pdfplumber is not None:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                return "\n".join(pages)
            except Exception:
                pass
        
        # Last resort: PyPDF2
        if PyPDF2 is not None:
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n".join(pages)
            except Exception:
                pass
    
    # Default: try to decode as text
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_layout_aware(file_bytes: bytes, filename: str) -> str:
    """
    Extract PDF text in a layout-aware order using PyMuPDF (fitz):
    - Read page text as blocks with coordinates
    - Sort by y (top) then x (left) to respect columns/boxes
    - Join blocks with blank lines
    Falls back to empty string on error; caller should handle fallback.
    """
    try:
        if fitz is None:
            return ""
        text_blocks = []
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                blocks = page.get_text("blocks") or []
                # Sort by y coordinate (top to bottom), then x (left to right)
                blocks.sort(key=lambda b: (round(b[1]), round(b[0])))
                for b in blocks:
                    t = (b[4] or "").strip()
                    if t:
                        text_blocks.append(t)
        return "\n\n".join(text_blocks).strip()
    except Exception:
        return ""


# -----------------------------
# Text utils
# -----------------------------
def normalize_ws(s: str) -> str:
    """Normalize whitespace"""
    return re.sub(r"\s+", " ", (s or "")).strip()


# -----------------------------
# GPT JD extraction
# -----------------------------
def normalize_lines(items: List[str]) -> List[str]:
    """Clean and deduplicate lines"""
    out = []
    for it in items or []:
        t = re.sub(r"\s+", " ", (it or "").strip(" •-–·\t")).strip()
        if t:
            out.append(t)
    
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for x in out:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl)
            deduped.append(x)
    return deduped


def extract_jd_sections_with_gpt(jd_text: str) -> JDSections:
    """Extract JD sections using GPT"""
    schema = {
        "type": "object",
        "properties": {
            "job_title": {"type": "string", "description": "The job title/position name"},
            "key_skills": {"type": "array", "items": {"type": "string"}},
            "responsibilities": {"type": "array", "items": {"type": "string"}},
            "qualifications": {"type": "array", "items": {"type": "string"}},
            "experience_required": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["job_title", "key_skills", "responsibilities", "qualifications", "experience_required"],
        "additionalProperties": False,
    }
    
    # Load prompts from admin-configurable file
    prompts = load_prompts()
    jd_extraction = prompts['jd_extraction']
    
    system = jd_extraction['system_prompt']['value']
    user = jd_extraction['user_prompt_template']['value'].format(jd_text=jd_text)
    
    # Use gpt-4o for JD extraction - better quality, and with PyMuPDF 1.26.6 JDs are clean/small
    data = call_llm_json(system, user, schema, model="gpt-4o")
    
    sections = JDSections(
        job_title=data.get("job_title", "Position Not Specified"),
        key_skills=normalize_lines(data.get("key_skills", [])),
        responsibilities=normalize_lines(data.get("responsibilities", [])),
        qualifications=normalize_lines(data.get("qualifications", [])),
        experience_required=normalize_lines(data.get("experience_required", [])),
    )
    return sections


def extract_candidate_name_with_gpt(resume_text: str, filename: str) -> str:
    """Extract candidate name from resume using GPT (fast and cheap with gpt-4o-mini)"""
    # Load admin settings to get candidate text character limit
    settings = load_gpt_settings()
    candidate_text_chars = settings['candidate_text_chars']
    
    schema = {
        "type": "object",
        "properties": {
            "candidate_name": {"type": "string", "description": "The candidate's full name"}
        },
        "required": ["candidate_name"],
        "additionalProperties": False,
    }
    
    system = "You extract the candidate's name from resumes. Return only the person's full name, properly capitalized."
    
    # Use admin-configured character limit (typically 4000 chars)
    user = f"""Extract the candidate's full name from this resume.

Filename: {filename}

Resume (first section):
{resume_text[:candidate_text_chars]}

Return JSON with candidate_name field."""
    
    try:
        # Use gpt-4o-mini for cost efficiency ($0.003 per call vs $0.01 for gpt-4o)
        data = call_llm_json(system, user, schema, model="gpt-4o-mini")
        name = data.get("candidate_name", "").strip()
        
        # Fallback to regex if AI returns empty/invalid
        if not name or len(name) < 3 or len(name) > 100:
            return infer_candidate_name(filename, resume_text)
        
        return name
    except Exception as e:
        print(f"AI name extraction failed: {e}. Falling back to regex.")
        return infer_candidate_name(filename, resume_text)


def build_criteria_from_sections(sections: JDSections, per_section: int = 6, cap_total: int = 30) -> Tuple[List[str], Dict[str, str]]:
    """Build criteria list from JD sections"""
    order = [
        ("key_skills", sections.key_skills),
        ("responsibilities", sections.responsibilities),
        ("qualifications", sections.qualifications),
        ("experience_required", sections.experience_required),
    ]
    
    crits: List[str] = []
    cat_map: Dict[str, str] = {}
    
    for sec_name, items in order:
        picked = items[:per_section]
        for p in picked:
            if p and p not in crits:
                crits.append(p)
                cat_map[p] = sec_name
            if len(crits) >= cap_total:
                break
        if len(crits) >= cap_total:
            break
    
    return crits[:cap_total], cat_map


# -----------------------------
# Candidate name inference
# -----------------------------
def normalize_ws(s: str) -> str:
    """Normalize whitespace in a string"""
    return re.sub(r"\s+", " ", s).strip()

def infer_candidate_name(file_name: str, text: str) -> str:
    """Extract candidate name from filename or resume text using sophisticated pattern matching.
    
    This function uses a three-tier approach:
    1. Look for "Name:" or "Name-" prefixed lines in the text
    2. Use intelligent name detection on text lines
    3. Fall back to filename parsing
    
    Handles:
    - All-caps names ("JIGNESH M DUSARA" → "Jignesh M Dusara")
    - Middle initials ("John A. Smith", "Jignesh M Dusara")
    - Compound names with 2-5 parts
    - Filters out resume keywords and common false positives
    """
    # CRITICAL: Check if text is suspiciously short (likely corrupted/scanned image)
    # Use obvious placeholder name to alert user
    if not text or len(text.strip()) < 100:
        import os
        base_filename = os.path.basename(file_name)
        return f"[UNREADABLE FILE - {base_filename}]"
    
    # Only scan first 10 lines for name patterns (faster, avoids picking up random text later)
    lines = [l.strip() for l in (text or "").splitlines()][:10]
    
    # Pattern 1: Look for "Name:" or "Name-" prefixed lines
    for l in lines:
        m = re.match(r"(?i)^\s*name\s*[:\-]\s*(.+)$", l)
        if m:
            cand = normalize_ws(m.group(1))
            # Split at "Address:" if present to avoid capturing address parts
            if "Address:" in cand or "address:" in cand.lower():
                cand = re.split(r"(?i)\s*address\s*:", cand)[0].strip()
            if 2 <= len(cand) <= 60 and not re.search(r"\d|@|http", cand):
                return cand.title()
    
    def looks_like_name(s: str) -> bool:
        """Check if a string looks like a person's name"""
        if not s or len(s) > 60:
            return False
        if re.search(r"@|http", s):  # Allow digits (for middle initials, etc.)
            return False
        
        # Split at "Address:" if present
        if "Address:" in s or "address:" in s.lower():
            s = re.split(r"(?i)\s*address\s*:", s)[0].strip()
        
        parts = [p for p in re.split(r"\s+", s) if p]
        if len(parts) < 2 or len(parts) > 5:  # Allow up to 5 name parts
            return False
        
        # Expanded bad-word list to avoid false positives
        bad = {
            "curriculum", "vitae", "resume", "cv", "profile", "professional", 
            "summary", "address", "phone", "email", "mobile", "contact",
            "international", "business", "education", "experience", "skills",
            "objective", "personal", "details", "information"
        }
        if any(w.lower() in bad for w in parts):
            return False
        
        # Handle all-caps names (common in resumes) like "JIGNESH M DUSARA"
        all_caps_parts = [p for p in parts if p.isupper() and len(p) > 1]
        if len(all_caps_parts) >= 2:
            # Looks like an all-caps name
            return True
        
        # Recognize middle initials (single uppercase letter, possibly with period)
        # Example: "Jignesh M Dusara" or "John A. Smith"
        has_initial = any(re.match(r"^[A-Z]\.?$", p) for p in parts)
        
        # Standard capitalized name pattern
        cap_like = sum(1 for p in parts if re.match(r"^[A-Z][a-z'\-]+$", p))
        
        # Accept if:
        # - Has middle initial and at least 1 other capitalized word
        # - Or has at least 2 capitalized words (or len(parts)-1, whichever is more)
        if has_initial and cap_like >= 1:
            return True
        return cap_like >= max(2, len(parts) - 1)
    
    # Pattern 2: Look for lines that look like names
    for l in lines:
        if looks_like_name(l):
            # Preserve case for all-caps names, title-case for mixed case
            if l.isupper():
                return l.title()
            return l.title()
    
    # Pattern 3: Fall back to filename parsing
    import os
    base = os.path.splitext(os.path.basename(file_name))[0]
    base = re.sub(r"(?i)\b(cv|resume|curriculum|vitae)\b", " ", base)
    base = re.sub(r"[_\-\.]+", " ", base)
    base = re.sub(r"\d+", " ", base)
    base = normalize_ws(base)
    parts = base.split()
    if 2 <= len(parts) <= 5:  # Allow up to 5 parts from filename
        return " ".join(p.capitalize() for p in parts)
    return (parts[0].capitalize() if parts else "Candidate")


# -----------------------------
# GPT-Powered Candidate Insights (Premium Feature)
# -----------------------------
# GPT Insights Generation
# -----------------------------
def gpt_candidate_insights(candidate_name: str, candidate_text: str, jd_text: str, 
                          coverage_scores: Dict[str, float], criteria: List[str],
                          evidence_map: Dict[Tuple[str, str], Tuple[str, float]],
                          model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Generate AI-powered insights about a candidate's fit for a job.
    
    Uses GPT to analyze:
    - Candidate's resume content
    - Job description requirements
    - Scoring evidence for each criterion
    - Overall coverage scores
    
    Returns structured insights:
    - top: List of 3-6 key strengths
    - gaps: List of 3-6 areas of concern/gaps
    - notes: 2-4 sentence overall assessment
    
    Settings loaded from config/gpt_settings.json (editable via /admin panel):
    - Model, temperature, max tokens
    - Evidence snippet size, candidate/JD context limits
    - All configurable without code changes
    
    Args:
        candidate_name: Full name of candidate
        candidate_text: Full resume text
        jd_text: Full job description text
        coverage_scores: Dict mapping criterion → score (0.0-1.0)
        criteria: List of all criterion names
        evidence_map: Dict mapping (candidate_name, criterion) → (snippet, score)
        model: Override model (if None, uses config setting)
    
    Returns:
        Dict with 'top', 'gaps', 'notes' keys containing insight strings
    """
    if not OpenAI:
        return {"error": "OpenAI not available"}
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Load current settings from config file (admin-configurable)
    try:
        settings = load_gpt_settings()
    except Exception as e:
        print(f"ERROR: Failed to load GPT settings: {str(e)}")
        return {
            "top": [],
            "gaps": [],
            "notes": f"Error loading GPT settings: {str(e)}"
        }
    
    # Use provided model parameter, otherwise use INSIGHT model from config
    model_to_use = model if model != "gpt-4o" else settings['insight_model']
    
    # Build criterion list with scores for GPT context
    # GPT will search the full resume itself rather than pre-selected snippets
    criteria_list = []
    for criterion in criteria:
        score = coverage_scores.get(criterion, 0.0)
        score_percent = int(score * 100)
        
        # Quality label to help GPT understand match strength (using admin-configurable thresholds)
        if score >= settings['strong_match_threshold']:
            quality = "STRONG match"
        elif score >= settings['good_match_threshold']:
            quality = "GOOD match"
        elif score >= settings['moderate_match_threshold']:
            quality = "MODERATE match"
        elif score >= settings['weak_match_threshold']:
            quality = "WEAK match"
        else:
            quality = "MINIMAL match"
        
        criteria_list.append(f"- {criterion} (SCORE: {score_percent}% - {quality})")
    
    criteria_with_scores = "\n".join(criteria_list)
    
    # JSON schema for response - includes justifications dict
    schema = {
        "type": "object",
        "properties": {
            "top": {"type": "array", "items": {"type": "string"}},
            "gaps": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
            "justifications": {
                "type": "object",
                "description": "Dict mapping criterion name to 1-sentence professional justification",
                "additionalProperties": {"type": "string"}
            }
        },
        "required": ["top", "gaps", "notes", "justifications"],
        "additionalProperties": False,
    }
    
    # Load prompts from admin-configurable file
    try:
        prompts = load_prompts()
        insights_prompts = prompts['candidate_insights']
    except Exception as e:
        print(f"ERROR: Failed to load prompts: {str(e)}")
        return {
            "top": [],
            "gaps": [],
            "notes": f"Error loading prompts: {str(e)}"
        }
    
    system_prompt = insights_prompts['system_prompt']['value']
    
    # Map notes_length setting to instruction text
    notes_length_map = {
        'brief': '1-2 sentences',
        'medium': '2-4 sentences',
        'detailed': '4-6 sentences (one paragraph)'
    }
    notes_instruction = notes_length_map.get(settings['notes_length'], '2-4 sentences')
    
    # Map insight_tone setting to instruction text
    tone_map = {
        'professional': 'professional hiring language',
        'conversational': 'friendly, conversational tone',
        'technical': 'technical focus with specific tech details',
        'executive': 'high-level strategic assessment suitable for C-suite'
    }
    tone_instruction = tone_map.get(settings['insight_tone'], 'professional hiring language')
    
    # Build user prompt from template, replacing placeholders
    user_prompt = insights_prompts['user_prompt_template']['value'].format(
        jd_text=jd_text[:settings['jd_text_chars']],
        candidate_name=candidate_name,
        candidate_text=candidate_text,  # Full resume - no artificial limits
        evidence_lines=criteria_with_scores  # Just scores, no snippets - GPT searches full resume
    )
    
    # Add context about what the scores mean
    user_prompt = f"IMPORTANT CONTEXT: The scores (0-100%) represent semantic similarity between the candidate's resume and each job requirement. Higher scores indicate stronger textual evidence of that skill/experience. A 90% score means excellent alignment with substantial relevant content. A 10% score means minimal/weak alignment with very limited relevant content.\n\n" + user_prompt
    
    # Add instructions for notes length and tone (using admin-configurable min/max)
    user_prompt += f"\n\nGenerate between {settings['min_strengths']}-{settings['max_strengths']} key strengths and {settings['min_gaps']}-{settings['max_gaps']} notable gaps or concerns. Write the overall notes in {notes_instruction} using {tone_instruction}."
    
    # Add justification instructions with score-aware guidance  
    user_prompt += "\n\nIMPORTANT JSON FORMAT: Return a JSON object with these exact keys:"
    user_prompt += "\n{'top': [array of strength strings], 'gaps': [array of gap strings], 'notes': 'assessment string', 'justifications': {dict mapping criterion to justification}}"
    user_prompt += "\n\nFor the 'justifications' object, create ONE entry per criterion with the EXACT criterion name as the key (without the SCORE suffix) and the justification sentence as the value."
    user_prompt += "\nExample: If criterion is 'Expert financial analysis (SCORE: 85% - STRONG match)', use key 'Expert financial analysis' NOT the full string with score."
    user_prompt += "\nExample justifications dict: {'Expert financial analysis': 'Demonstrated financial analysis in role X', 'Advanced skills with Microsoft Excel': 'Used Excel extensively at Y'}"
    user_prompt += "\n\nDO NOT embed justifications inside the top/gaps arrays. Keep them separate in the justifications dict."
    user_prompt += "\n\nCRITICAL JSON FORMATTING: Use only straight double quotes for JSON strings. Avoid contractions (use 'does not' instead of 'doesn't'). Avoid possessive apostrophes where possible. If you must use apostrophes or quotes within strings, they will be automatically escaped."
    user_prompt += "\n\nSearch the FULL resume to find ALL relevant evidence for each criterion. Don't limit yourself to one example - cite multiple roles/achievements if the candidate demonstrates the skill across their career."
    user_prompt += "\n\nCALIBRATE YOUR TONE based on the score percentage:"
    user_prompt += f"\n- STRONG match ({int(settings['strong_match_threshold']*100)}-100%): Emphasize depth and breadth of evidence. Cite multiple examples if present."
    user_prompt += f"\n- GOOD match ({int(settings['good_match_threshold']*100)}-{int(settings['strong_match_threshold']*100-1)}%): Note solid evidence while acknowledging it's not exhaustive. Balanced positive tone."
    user_prompt += f"\n- MODERATE match ({int(settings['moderate_match_threshold']*100)}-{int(settings['good_match_threshold']*100-1)}%): Acknowledge limited evidence exists but gaps are notable. Cautious tone."
    user_prompt += f"\n- WEAK match ({int(settings['weak_match_threshold']*100)}-{int(settings['moderate_match_threshold']*100-1)}%): Note minimal evidence found. State what little exists without overstating it."
    user_prompt += f"\n- MINIMAL match (0-{int(settings['weak_match_threshold']*100-1)}%): State clearly that insufficient evidence was found, or that evidence suggests lack of this skill."
    user_prompt += "\n\nEach justification must:"
    user_prompt += "\n1. Reflect the score honestly - don't oversell weak matches or undersell strong ones"
    user_prompt += "\n2. Cite specific facts from the FULL resume (roles, numbers, achievements across multiple jobs if applicable)"
    user_prompt += "\n3. Be unique per criterion - explain WHY it demonstrates THAT specific criterion"
    user_prompt += "\n4. Avoid generic phrases like 'This shows X' or 'The candidate demonstrates Y'"
    user_prompt += "\n5. Be professional and concise (1-2 sentences max)"
    user_prompt += "\n\nExamples:"
    user_prompt += "\n- 85% Budget Management: \"Demonstrated consistent budget oversight across three senior roles: managed $2M annual budgets at SLF Lawyers (2018-2023), directed departmental budgets at Gold Coast Council (2015-2018), and provided budget analysis at Previous Corp (2012-2015).\""
    user_prompt += "\n- 40% Budget Management: \"Resume mentions managing a small team budget at SLF Lawyers, though lacks detail on scope or specific financial responsibilities.\""
    user_prompt += "\n- 8% Budget Management: \"No evidence of budget management responsibilities found; roles focus primarily on technical execution rather than financial oversight.\""
    
    try:
        # Call GPT with INSIGHT agent settings (not RANKER settings!)
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=settings['insight_temperature'],  # INSIGHT temp (0.4)
            max_tokens=settings['insight_max_tokens']      # INSIGHT max tokens (1000)
        )
        
        # Parse JSON response with aggressive error recovery
        raw_content = response.choices[0].message.content
        
        # DEBUG: Always log the raw response for troubleshooting
        print(f"DEBUG: Raw GPT response for {candidate_name}:")
        print(f"DEBUG: Length: {len(raw_content)} characters")
        print(f"DEBUG: First 500 chars: {raw_content[:500]}")
        print(f"DEBUG: Last 200 chars: {raw_content[-200:]}")
        
        try:
            result = json.loads(raw_content)
        except json.JSONDecodeError as e:
            # JSON parsing failed - likely due to unescaped quotes in GPT response
            print(f"ERROR: JSON parsing failed for {candidate_name}: {str(e)}")
            print(f"ERROR: Error location: line {e.lineno} column {e.colno} (char {e.pos})")
            
            # Show context around the error
            if e.pos and e.pos < len(raw_content):
                start = max(0, e.pos - 100)
                end = min(len(raw_content), e.pos + 100)
                print(f"ERROR: Context around error position:")
                print(f"ERROR: ...{raw_content[start:end]}...")
            
            # Try multiple sanitization strategies
            sanitized = raw_content
            
            # Strategy 1: Replace curly quotes with straight quotes
            sanitized = sanitized.replace('\u2018', "'").replace('\u2019', "'")
            sanitized = sanitized.replace('\u201c', '"').replace('\u201d', '"')
            sanitized = sanitized.replace('\u201a', "'").replace('\u201e', '"')
            
            # Strategy 2: Remove any stray newlines inside strings (common GPT error)
            # This is tricky - we need to preserve newlines in the JSON structure
            # but remove them from within string values
            
            # Strategy 3: Try to fix common issues like unescaped backslashes
            # sanitized = sanitized.replace('\\', '\\\\')  # Too aggressive
            
            try:
                # Try parsing again after sanitization
                result = json.loads(sanitized)
                print(f"SUCCESS: JSON parsing recovered after sanitization strategy 1")
            except json.JSONDecodeError as e2:
                # Still failed - try more aggressive fixes
                print(f"ERROR: Sanitization strategy 1 failed: {str(e2)}")
                
                # Strategy 4: Try to extract just the JSON object if GPT added extra text
                # Look for the first { and last }
                first_brace = sanitized.find('{')
                last_brace = sanitized.rfind('}')
                
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_only = sanitized[first_brace:last_brace+1]
                    print(f"INFO: Extracted JSON from position {first_brace} to {last_brace}")
                    
                    try:
                        result = json.loads(json_only)
                        print(f"SUCCESS: JSON parsing recovered by extracting JSON object")
                    except json.JSONDecodeError as e3:
                        # Complete failure - return error to user with details
                        print(f"ERROR: All sanitization strategies failed: {str(e3)}")
                        print(f"ERROR: Full raw response:")
                        print(raw_content)
                        
                        return {
                            "top": ["Unable to generate insights due to error"],
                            "gaps": ["Unable to generate insights due to error"],
                            "notes": f"Evaluation error: {str(e)}",
                            "justifications": {}
                        }
                else:
                    # Can't find valid JSON structure
                    print(f"ERROR: Could not find valid JSON object in response")
                    print(f"ERROR: Full raw response:")
                    print(raw_content)
                    
                    return {
                        "top": ["Unable to generate insights due to error"],
                        "gaps": ["Unable to generate insights due to error"],
                        "notes": f"Evaluation error: {str(e)}",
                        "justifications": {}
                    }
        
        # Normalize key names - GPT sometimes returns different variations
        # Try multiple possible key names for each field
        top_keys = ['top', 'top_strengths', 'Top strengths', 'strengths']
        gaps_keys = ['gaps', 'gaps_risks', 'Gaps / risks', 'risks', 'concerns']
        notes_keys = ['notes', 'overall_notes', 'Overall notes', 'assessment']
        
        # Extract top strengths - handle both string arrays and object arrays
        top_raw = None
        for key in top_keys:
            if key in result:
                top_raw = result[key]
                break
        
        if isinstance(top_raw, list):
            # Handle array of strings OR array of objects with 'criterion'/'evidence' fields
            top_items = []
            for item in top_raw:
                if isinstance(item, str):
                    top_items.append(item.strip())
                elif isinstance(item, dict):
                    # Extract text from object (e.g., {'criterion': 'X', 'evidence': 'Y'})
                    text = item.get('evidence', '') or item.get('criterion', '') or str(item)
                    top_items.append(text.strip())
            top = [s for s in top_items if s]
        else:
            top = []
        
        # Extract gaps - handle both string arrays and object arrays
        gaps_raw = None
        for key in gaps_keys:
            if key in result:
                gaps_raw = result[key]
                break
        
        if isinstance(gaps_raw, list):
            gaps_items = []
            for item in gaps_raw:
                if isinstance(item, str):
                    gaps_items.append(item.strip())
                elif isinstance(item, dict):
                    # Extract text from object (e.g., {'criterion': 'X', 'rationale': 'Y'})
                    text = item.get('rationale', '') or item.get('criterion', '') or str(item)
                    gaps_items.append(text.strip())
            gaps = [s for s in gaps_items if s]
        else:
            gaps = []
        
        # Extract notes - always a string
        notes = ''
        for key in notes_keys:
            if key in result:
                notes = result[key]
                break
        if isinstance(notes, str):
            notes = notes.strip()
        else:
            notes = str(notes).strip()
        
        # Extract justifications - dict mapping criterion → justification sentence
        justifications = result.get('justifications', {})
        if not isinstance(justifications, dict):
            justifications = {}
        
        # FIXED: Normalize justification keys to match criterion names (strip score suffixes)
        # GPT sometimes includes the score suffix like " (SCORE: 20% - WEAK match)"
        normalized_justifications = {}
        for key, value in justifications.items():
            # Remove score suffix if present
            clean_key = re.sub(r'\s*\(SCORE:.*?\)', '', key).strip()
            normalized_justifications[clean_key] = value
        justifications = normalized_justifications
        
        # Fallback: If justifications empty, check if GPT put them inside top_strengths
        if len(justifications) == 0:
            for key in ['top', 'top_strengths']:
                if key in result and isinstance(result[key], list):
                    for item in result[key]:
                        if isinstance(item, dict) and 'criterion' in item and 'justification' in item:
                            clean_key = re.sub(r'\s*\(SCORE:.*?\)', '', item['criterion']).strip()
                            justifications[clean_key] = item['justification']
        
        # DEBUG: Log justification status
        print(f"DEBUG: Justifications extracted: {len(justifications)} items")
        if len(justifications) == 0:
            print(f"DEBUG: GPT response keys: {result.keys()}")
            print(f"DEBUG: Full GPT response (first 500 chars): {str(result)[:500]}")
        
        return {
            "top": top,
            "gaps": gaps,
            "notes": notes,
            "justifications": justifications
        }
        
    except Exception as e:
        print(f"ERROR: GPT insights failed for {candidate_name}: {str(e)}")
        return {
            "top": [],
            "gaps": [],
            "notes": f"Error generating insights: {str(e)}",
            "justifications": {}
        }


# -----------------------------
# Cost estimation
# -----------------------------
def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token"""
    return len(text) // 4


def estimate_analysis_cost(jd_text: str, candidate_texts: List[str], num_criteria: int, model: str = "gpt-4o") -> Dict[str, Any]:
    """Estimate GPT API cost for analysis"""
    # Pricing per 1M tokens (as of Nov 2024)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }
    
    prices = PRICING.get(model, PRICING["gpt-4o"])
    
    # JD extraction
    jd_tokens_in = estimate_tokens(jd_text)
    jd_tokens_out = num_criteria * 20
    
    # Candidate insights (for top 3)
    num_insights = min(3, len(candidate_texts))
    avg_candidate_tokens = sum(estimate_tokens(text) for text in candidate_texts[:num_insights]) // max(1, num_insights)
    total_insight_input = (jd_tokens_in + num_criteria * 15 + avg_candidate_tokens) * num_insights
    total_insight_output = num_insights * 500  # ~500 tokens per insight
    
    # Total
    total_input = jd_tokens_in + total_insight_input
    total_output = jd_tokens_out + total_insight_output
    
    cost_input = (total_input / 1_000_000) * prices["input"]
    cost_output = (total_output / 1_000_000) * prices["output"]
    total_cost = cost_input + cost_output
    
    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "estimated_cost_usd": round(total_cost, 4),
        "model": model
    }
