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
    Load GPT configuration settings from config/gpt_settings.json
    
    This allows admin to adjust parameters without touching code:
    - GPT model selection (gpt-4o, gpt-4o-mini, etc.)
    - Temperature (creativity vs consistency)
    - Token limits (response length)
    - Evidence snippet size (context per criterion)
    - Candidate/JD text limits (overall context)
    - Score thresholds (High/Low cutoffs)
    
    Returns dict with all settings and their current values.
    """
    settings_path = os.path.join(os.path.dirname(__file__), 'config', 'gpt_settings.json')
    
    # Load settings from JSON file
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    
    # Return flattened dict with just the values for easy access
    return {
        'model': settings['gpt_model']['value'],
        'temperature': settings['temperature']['value'],
        'max_tokens': settings['max_tokens']['value'],
        'evidence_snippet_chars': settings['evidence_snippet_chars']['value'],
        'candidate_text_chars': settings['candidate_text_chars']['value'],
        'jd_text_chars': settings['jd_text_chars']['value'],
        'high_threshold': settings['score_thresholds']['high_threshold']['value'],
        'low_threshold': settings['score_thresholds']['low_threshold']['value'],
        'top_evidence_items': settings['advanced_settings']['top_evidence_items']['value']
    }


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
    print(f"DEBUG: OPENAI_API_KEY exists: {bool(api_key)}")
    print(f"DEBUG: OPENAI_API_KEY value (first 10 chars): {api_key[:10] if api_key else 'None'}")
    print(f"DEBUG: OpenAI module available: {OpenAI is not None}")
    
    # Check OpenAI library version
    if OpenAI is not None:
        import openai
        print(f"DEBUG: OpenAI library version: {openai.__version__}")
    
    if not api_key or OpenAI is None:
        print("DEBUG: Returning None - no API key or OpenAI module not available")
        return None
    try:
        client = OpenAI(api_key=api_key)
        print("DEBUG: OpenAI client created successfully")
        return client
    except Exception as e:
        print(f"DEBUG: Failed to create OpenAI client: {e}")
        import traceback
        print(f"DEBUG: Full traceback:\n{traceback.format_exc()}")
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


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks"""
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return [text]
    
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


# -----------------------------
# Embeddings
# -----------------------------
_embedder_cache = None

def load_embedder() -> Dict[str, Any]:
    """Load sentence transformer model (cached)"""
    global _embedder_cache
    
    if _embedder_cache is not None:
        return _embedder_cache
    
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            model.to(device)
            _embedder_cache = {"type": "sbert", "model": model, "device": device}
            return _embedder_cache
        except Exception as e:
            print(f"Failed to load sentence transformer: {e}")
    
    _embedder_cache = {"type": "tfidf"}
    return _embedder_cache


def embed_sbert(texts: List[str], model) -> np.ndarray:
    """Generate embeddings using sentence transformers"""
    return np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity"""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def prepare_corpus_embeddings(corpus_chunks: List[str], embedder_info: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    """Prepare embeddings for corpus chunks"""
    if embedder_info.get("type") == "sbert" and embedder_info.get("model") is not None:
        embs = embed_sbert(corpus_chunks, embedder_info["model"])
        return embs, "sbert"
    else:
        # Fallback to TF-IDF if sentence transformers not available
        if TfidfVectorizer is not None:
            vec = TfidfVectorizer()
            embs = vec.fit_transform(corpus_chunks).toarray()
            return embs, "tfidf"
        else:
            # No embeddings available, return zeros
            return np.zeros((len(corpus_chunks), 1)), "none"


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
            "key_skills": {"type": "array", "items": {"type": "string"}},
            "responsibilities": {"type": "array", "items": {"type": "string"}},
            "qualifications": {"type": "array", "items": {"type": "string"}},
            "experience_required": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["key_skills", "responsibilities", "qualifications", "experience_required"],
        "additionalProperties": False,
    }
    
    system = (
        "You extract criteria from job descriptions into clear bullet items. "
        "Be concise, de-duplicate similar bullets, and keep each item atomic."
    )
    
    user = (
        "Extract the following JD into four sections (key_skills, responsibilities, "
        "qualifications, experience_required). Return ONLY JSON matching the provided schema.\n\n"
        f"JD:\n{jd_text}"
    )
    
    data = call_llm_json(system, user, schema)
    
    return JDSections(
        key_skills=normalize_lines(data.get("key_skills", [])),
        responsibilities=normalize_lines(data.get("responsibilities", [])),
        qualifications=normalize_lines(data.get("qualifications", [])),
        experience_required=normalize_lines(data.get("experience_required", [])),
    )


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
# Main analysis function
# -----------------------------
def analyse_candidates(
    candidates: List[Candidate],
    criteria: List[str],
    weights: Optional[List[float]] = None,
    chunk_chars: int = 1200,
    overlap: int = 150,
) -> Tuple[pd.DataFrame, Dict[str, Dict], Dict[Tuple[str, str], Tuple[str, float]]]:
    """
    Analyze candidates against criteria using semantic similarity
    
    Returns:
        - coverage: DataFrame with scores for each candidate/criterion
        - insights: Dict[candidate_name, dict] (placeholder for GPT insights)
        - evidence_map: Dict[(candidate, criterion), (snippet, score)]
    """
    if weights is None:
        weights = [1.0] * len(criteria)
    
    # Chunk all candidate texts
    all_chunk_texts = []
    chunk_index = []  # (candidate_idx, chunk_idx)
    
    for i, cand in enumerate(candidates):
        chunks = chunk_text(cand.text, chunk_chars, overlap)
        for j, ch in enumerate(chunks):
            all_chunk_texts.append(ch)
            chunk_index.append((i, j))
    
    # Load embedder and create embeddings
    info = load_embedder()
    chunk_embs, vec_type = prepare_corpus_embeddings(all_chunk_texts, info)
    
    # Map candidate_idx -> chunk row indices
    cand_to_rows = {i: [] for i in range(len(candidates))}
    for ridx, (ci, cj) in enumerate(chunk_index):
        cand_to_rows[ci].append(ridx)
    
    # Analyze each candidate
    coverage_records = []
    evidence_map: Dict[Tuple[str, str], Tuple[str, float]] = {}
    
    for cand_idx, cand in enumerate(candidates):
        row = {"Candidate": cand.name}
        chunk_rows = cand_to_rows[cand_idx]
        
        if len(chunk_rows) == 0:
            # No chunks for this candidate
            for crit in criteria:
                row[crit] = 0.0
            row["Overall"] = 0.0
            coverage_records.append(row)
            continue
        
        cand_embs = chunk_embs[chunk_rows]
        
        # Score against each criterion
        criterion_scores = []
        for crit_idx, crit in enumerate(criteria):
            # Embed criterion
            if vec_type == "sbert":
                crit_emb = embed_sbert([crit], info["model"])
            else:
                crit_emb = np.zeros((1, cand_embs.shape[1]))
            
            # Compute similarity to all candidate chunks
            sims = pairwise_cosine(crit_emb, cand_embs)[0]
            max_sim = float(sims.max())
            best_chunk_idx = int(sims.argmax())
            
            row[crit] = max_sim
            criterion_scores.append(max_sim * weights[crit_idx])
            
            # Store evidence (best matching snippet)
            best_chunk_text = all_chunk_texts[chunk_rows[best_chunk_idx]]
            evidence_map[(cand.name, crit)] = (best_chunk_text, max_sim)
        
        # Compute weighted overall score
        row["Overall"] = sum(criterion_scores) / sum(weights) if sum(weights) > 0 else 0.0
        coverage_records.append(row)
    
    # Create DataFrame and sort by Overall score
    coverage = pd.DataFrame(coverage_records)
    coverage = coverage.sort_values("Overall", ascending=False).reset_index(drop=True)
    
    # Insights placeholder (will be filled by GPT if enabled)
    insights = {}
    
    return coverage, insights, evidence_map


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
    settings = load_gpt_settings()
    
    # Use provided model parameter, otherwise use config setting
    model_to_use = model if model != "gpt-4o" else settings['model']
    
    # Sort criteria by score and take top N based on config
    sorted_criteria = sorted(criteria, key=lambda c: coverage_scores.get(c, 0.0), reverse=True)
    top_criteria = sorted_criteria[:settings['top_evidence_items']]
    
    # Build evidence lines from coverage scores and evidence snippets
    # Uses configurable character limit from admin settings
    ev_items = []
    for criterion in top_criteria:
        score = coverage_scores.get(criterion, 0.0)
        # Get evidence snippet from evidence_map
        evidence_key = (candidate_name, criterion)
        snippet = ""
        if evidence_key in evidence_map:
            # Truncate to admin-configured length
            snippet = evidence_map[evidence_key][0][:settings['evidence_snippet_chars']].replace("\n", " ")
        ev_items.append(f"- {criterion} (score {score:.2f}): {snippet}")
    
    ev_lines = "\n".join(ev_items)
    
    # JSON schema for response
    schema = {
        "type": "object",
        "properties": {
            "top": {"type": "array", "items": {"type": "string"}},
            "gaps": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": ["top", "gaps", "notes"],
        "additionalProperties": False,
    }
    
    # Construct prompt (matching Streamlit version EXACTLY)
    system_prompt = (
        "You are a hiring analyst. Produce concise, high-signal insights about a candidate "
        "relative to the provided Job Description and evidence. Be specific and avoid fluff."
    )
    
    # User prompt with admin-configurable text limits for candidate/JD context
    user_prompt = (
        f"Job Description (excerpted):\n{jd_text[:settings['jd_text_chars']]}\n\n"
        f"Candidate: {candidate_name}\n"
        f"Candidate content (excerpted):\n{candidate_text[:settings['candidate_text_chars']]}\n\n"
        f"Evidence by criterion (top {settings['top_evidence_items']} by similarity):\n"
        + ev_lines +
        "\n\nTask:\n"
        "- Provide 3–6 bullet **Top strengths** tied to criteria and tangible evidence.\n"
        "- Provide 3–6 bullet **Gaps / risks** with rationale.\n"
        "- Provide a short **Notes** paragraph (2–4 sentences) with an overall view.\n\n"
        "Return ONLY valid JSON that conforms to the provided JSON Schema. "
        "Do not include backticks or any commentary.\n\n"
        f"Schema:\n{json.dumps(schema)}"
    )
    
    try:
        # Call GPT with admin-configured settings (temperature, max_tokens, model)
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=settings['temperature'],  # Admin-configurable
            max_tokens=settings['max_tokens']      # Admin-configurable
        )
        
        result = json.loads(response.choices[0].message.content)
        # Normalize whitespace
        return {
            "top": [s.strip() for s in result.get("top", []) if s.strip()],
            "gaps": [s.strip() for s in result.get("gaps", []) if s.strip()],
            "notes": result.get("notes", "").strip(),
        }
        
    except Exception as e:
        return {
            "top": [],
            "gaps": [],
            "notes": f"Error generating insights: {str(e)}"
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
