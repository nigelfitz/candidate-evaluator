# Candidate Analyser
from __future__ import annotations

import io, os, re, json, hashlib, textwrap, sys, html
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page Configuration (must be first)
# -----------------------------
st.set_page_config(
    page_title="Candidate Analyser",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to reduce top padding and add app branding
st.markdown("""
    <style>
    /* Reduce top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* App branding header with distinct styling */
    header[data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.95);
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f2f6;
    }
    header[data-testid="stHeader"]::before {
        content: "🎯 Candidate Analyser";
        font-size: 1.5rem;
        font-weight: 700;
        padding-left: 1rem;
        color: #1f77b4;
        white-space: nowrap;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    /* Add spacing after header before main content */
    .main .block-container {
        margin-top: 1rem;
    }
    /* Center text vertically in info boxes */
    .stAlert {
        padding-top: 0.9rem !important;
        padding-bottom: 0.9rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# API Key detection (global)
# -----------------------------
API_KEY_SET = bool(os.getenv('OPENAI_API_KEY'))

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
                blocks.sort(key=lambda b: (round(b[1]), round(b[0])))
                for b in blocks:
                    t = (b[4] or "").strip()
                    if t:
                        text_blocks.append(t)
        return "\n\n".join(text_blocks).strip()
    except Exception:
        return ""


# -----------------------------
# Optional / lazy imports
# -----------------------------
def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

pdfplumber = _try_import("pdfplumber")
docx = _try_import("docx")
sklearn = _try_import("sklearn")
dotenv = _try_import("dotenv")
reportlab = _try_import("reportlab")
_tabulate = _try_import("tabulate")

# OpenAI SDK (support both modern client and legacy)
_openai_mod = _try_import("openai")
try:
    # New SDK style
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

# Robust PDF extractor (PyMuPDF) and optional OCR fallback
fitz = _try_import("fitz")  # PyMuPDF
pdf2image = _try_import("pdf2image")
pytesseract = _try_import("pytesseract")
PIL_mod = _try_import("PIL")
Image = getattr(PIL_mod, "Image", None) if PIL_mod else None

if sklearn:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
else:
    TfidfVectorizer = None
    sk_cosine = None

st_models = _try_import("sentence_transformers")
if st_models:
    from sentence_transformers import SentenceTransformer
    import torch
else:
    SentenceTransformer = None
    torch = None

if reportlab:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.units import cm

if dotenv is not None:
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Candidate:
    name: str
    file_name: str
    text: str
    hash: str
    raw_bytes: Optional[bytes] = None  # keep original bytes for PDF viewing

@dataclass
class JD:
    file_name: str
    text: str
    hash: str
    raw_bytes: Optional[bytes] = None  # keep original bytes for robust extractor if needed

@dataclass
class JDSections:
    key_skills: List[str]
    responsibilities: List[str]
    qualifications: List[str]
    experience_required: List[str]

def _sections_to_dict(secs: JDSections) -> Dict[str, list]:
    return {
        "key_skills": list(secs.key_skills or []),
        "responsibilities": list(secs.responsibilities or []),
        "qualifications": list(secs.qualifications or []),
        "experience_required": list(secs.experience_required or []),
    }
# -----------------------------
# Hash helpers
# -----------------------------
def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

# -----------------------------
# OpenAI helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def _cached_openai_client():
    """
    Returns (mode, client_or_module):
      - ("modern", OpenAI()) if openai>=1.0 client is available
      - ("legacy", openai) if old ChatCompletion API is available
      - (None, None) if nothing usable
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return (None, None)
    # Prefer modern client
    if OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
            return ("modern", client)
        except Exception:
            pass
    # Try legacy module
    if _openai_mod is not None:
        try:
            _openai_mod.api_key = api_key
            return ("legacy", _openai_mod)
        except Exception:
            pass
    return (None, None)

def call_llm_json(system_prompt: str, user_prompt: str, schema: Dict[str, Any], model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Calls OpenAI to return structured JSON adhering to 'schema'.
    Works with both the modern and legacy python SDK variants.
    """
    mode, cli = _cached_openai_client()
    if mode is None or cli is None:
        # No API available
        return {"key_skills": [], "responsibilities": [], "qualifications": [], "experience_required": []}

    # Build a compact instruction to strongly bias JSON-only output
    json_guard = (
        "Return ONLY valid JSON that conforms to the provided JSON Schema. "
        "Do not include backticks or any commentary."
    )

    if mode == "modern":
        # function-free JSON via response_format in new SDK
        try:
            resp = cli.chat.completions.create(
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
        except Exception:
            return {"key_skills": [], "responsibilities": [], "qualifications": [], "experience_required": []}
    else:
        # legacy: best-effort JSON
        try:
            resp = cli.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_prompt}\n\n{json_guard}\n\nSchema:\n{json.dumps(schema)}"}
                ],
                temperature=0.2,
            )
            txt = resp["choices"][0]["message"]["content"] or "{}"
            return json.loads(txt)
        except Exception:
            return {"key_skills": [], "responsibilities": [], "qualifications": [], "experience_required": []}

# -----------------------------
# File reading / extraction
# -----------------------------
@st.cache_data(show_spinner=False)
def read_file_bytes(file_bytes: bytes, file_name: str) -> str:
    lower = file_name.lower()
    if lower.endswith(".txt"):
        try:
            return file_bytes.decode("utf-8")
        except Exception:
            return file_bytes.decode("latin-1", errors="ignore")
    if lower.endswith(".docx") and docx is not None:
        try:
            d = docx.Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in d.paragraphs)
        except Exception:
            pass
    if lower.endswith(".pdf") and pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages)
        except Exception:
            pass
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_pdf_robust(file_bytes: bytes, strong: bool = False, prefer_pdf2image: bool = True) -> str:
    """
    Primary: PyMuPDF text extraction.
    Fallback: rasterize and OCR with pytesseract if very low yield.
    Uses pdf2image if available; otherwise rasterizes via PyMuPDF to avoid Poppler dependency.
    If `strong=True`, render at higher DPI and use stronger thresholding to recover faint text.
    """
    text = ""
    # ---- Pass 1: direct text extraction via PyMuPDF ----
    if fitz is not None:
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                chunks = []
                for p in doc:
                    chunks.append(p.get_text("text"))
                text = "\n".join(chunks)
        except Exception:
            text = ""

    # ---- Pass 2: OCR when text is too small ----
    if len((text or "").strip()) < 200 and pytesseract is not None:
        ocr_chunks = []
        try:
            dpi = 400 if strong else 300
            thresh_cut = 170 if strong else 200  # lower cut -> darker letters kept
            tess_cfg = "--oem 3 --psm 6"
            if pdf2image is not None:
                from pdf2image import convert_from_bytes
                pages = convert_from_bytes(file_bytes, dpi=dpi)
                for img in pages:
                    if Image is not None:
                        gray = img.convert("L")
                        bw = gray.point(lambda x: 0 if x < thresh_cut else 255, "1").convert("L")
                        img_for_ocr = bw
                    else:
                        img_for_ocr = img
                    ocr_chunks.append(pytesseract.image_to_string(img_for_ocr, config=tess_cfg))
            elif fitz is not None:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    mat = fitz.Matrix(dpi/72.0, dpi/72.0)
                    for page in doc:
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        if Image is not None:
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            gray = img.convert("L")
                            bw = gray.point(lambda x: 0 if x < thresh_cut else 255, "1").convert("L")
                            ocr_chunks.append(pytesseract.image_to_string(bw, config=tess_cfg))
                        else:
                            ocr_chunks.append("")
            text = "\n".join([t for t in ocr_chunks if t])
        except Exception:
            pass

    return (text or "").strip()# -----------------------------
# Text utils
# -----------------------------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

# -----------------------------
# Embeddings / scoring
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder() -> Dict[str, Any]:
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            model.to(device)
            return {"type": "sbert", "model": model, "device": device}
        except Exception:
            pass
    return {"type": "tfidf"}

def embed_sbert(texts: List[str], model) -> np.ndarray:
    return np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))

def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
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
# GPT JD extraction (new)
# -----------------------------
def _normalize_lines(items: List[str]) -> List[str]:
    out = []
    for it in items or []:
        t = re.sub(r"\s+", " ", (it or "").strip(" •-–·\t")).strip()
        if t:
            out.append(t)
    # de-dupe while preserving order
    seen = set()
    deduped = []
    for x in out:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl)
            deduped.append(x)
    return deduped

def extract_jd_sections_with_gpt(jd_text: str) -> JDSections:
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
        key_skills=_normalize_lines(data.get("key_skills", [])),
        responsibilities=_normalize_lines(data.get("responsibilities", [])),
        qualifications=_normalize_lines(data.get("qualifications", [])),
        experience_required=_normalize_lines(data.get("experience_required", [])),
    )

@st.cache_resource()

def cached_extract_jd_sections_with_gpt(jd_hash: str, jd_text: str) -> JDSections:
    # Cache key includes jd_hash so repeated runs on the same JD don't re-call GPT
    if 'last_cached_jd_hash' in st.session_state and st.session_state['last_cached_jd_hash'] == jd_hash:
        try: st.toast('JD extracted from cache ✅', icon='💾')
        except Exception: pass
    else:
        st.session_state['last_cached_jd_hash'] = jd_hash
    return extract_jd_sections_with_gpt(jd_text)

def build_criteria_from_gpt_sections(sections: JDSections, per_section: int = 6, cap_total: int = 30) -> Tuple[List[str], Dict[str,str]]:
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
                crits.append(p); cat_map[p] = sec_name
            if len(crits) >= cap_total:
                break
        if len(crits) >= cap_total:
            break
    return crits[:cap_total], cat_map

# -----------------------------
# Embedding & analysis (unchanged)
# -----------------------------
@st.cache_data(show_spinner=False)
def prepare_corpus_embeddings(corpus_chunks: List[str], _embedder_info: Dict[str, Any]):
    info = _embedder_info
    if info.get("type") == "sbert" and info.get("model") is not None:
        embs = embed_sbert(corpus_chunks, info["model"])
        return embs, None
    if TfidfVectorizer is not None and sk_cosine is not None:
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        mat = vec.fit_transform(corpus_chunks)
        return mat, vec
    vocab: Dict[str, int] = {}
    rows: List[Dict[str, int]] = []
    for doc in corpus_chunks:
        counts: Dict[str, int] = {}
        for t in re.findall(r"\b\w+\b", (doc or "").lower()):
            counts[t] = counts.get(t, 0) + 1
        rows.append(counts)
        for t in counts:
            if t not in vocab:
                vocab[t] = len(vocab)
    mat = np.zeros((len(rows), len(vocab)), dtype=float)
    for i, row in enumerate(rows):
        for t, c in row.items():
            mat[i, vocab[t]] = c
    return mat, {"type": "bow", "vocab": vocab}

def infer_candidate_name(file_name: str, text: str) -> str:
    lines = [l.strip() for l in (text or "").splitlines()][:60]
    for l in lines:
        m = re.match(r"(?i)^\s*name\s*[:\-]\s*(.+)$", l)
        if m:
            cand = normalize_ws(m.group(1))
            if 2 <= len(cand) <= 60 and not re.search(r"\d|@|http", cand):
                return cand.title()
    def looks_like_name(s: str) -> bool:
        if not s or len(s) > 60:
            return False
        if re.search(r"\d|@|http", s):
            return False
        parts = [p for p in re.split(r"\s+", s) if p]
        if len(parts) < 2 or len(parts) > 4:
            return False
        bad = {"curriculum", "vitae", "resume", "cv", "profile", "professional", "summary"}
        if any(w.lower() in bad for w in parts):
            return False
        cap_like = sum(1 for p in parts if re.match(r"^[A-Z][a-z'\-]+$", p))
        return cap_like >= max(2, len(parts) - 1)
    for l in lines:
        if looks_like_name(l):
            return l.title()
    base = os.path.splitext(os.path.basename(file_name))[0]
    base = re.sub(r"(?i)\b(cv|resume|curriculum|vitae)\b", " ", base)
    base = re.sub(r"[_\-\.]+", " ", base)
    base = re.sub(r"\d+", " ", base)
    base = normalize_ws(base)
    parts = base.split()
    if 2 <= len(parts) <= 4:
        return " ".join(p.capitalize() for p in parts)
    return (parts[0].capitalize() if parts else "Candidate")

def compute_max_similarity_to_chunks(query_texts, chunk_embs, info, vec_or_meta):
    if info["type"] == "sbert":
        q_embs = embed_sbert(query_texts, info["model"])
        sims = pairwise_cosine(q_embs, chunk_embs)
        argmax = sims.argmax(axis=1)
        return sims.max(axis=1), argmax
    if vec_or_meta is None:
        return np.zeros(len(query_texts)), np.zeros(len(query_texts), dtype=int)
    if hasattr(vec_or_meta, "transform"):
        q_mat = vec_or_meta.transform(query_texts)
        sims = sk_cosine(q_mat, chunk_embs)
        argmax = np.asarray(sims.argmax(axis=1)).ravel()
        return sims.max(axis=1), argmax
    vocab = vec_or_meta["vocab"]
    def bow_vec(s: str):
        v = np.zeros((len(vocab),), dtype=float)
        for t in re.findall(r"\b\w+\b", (s or "").lower()):
            if t in vocab:
                v[vocab[t]] += 1
        return v
    q_mat = np.stack([bow_vec(s) for s in query_texts], axis=0)
    denom = (np.linalg.norm(q_mat, axis=1, keepdims=True) + 1e-12) * (np.linalg.norm(chunk_embs, axis=1, keepdims=True).T + 1e-12)
    sims = (q_mat @ chunk_embs.T) / denom
    argmax = sims.argmax(axis=1)
    return sims.max(axis=1), argmax

def analyse_candidates(
    candidates: List[Candidate],
    criteria: List[str],
    weights: Optional[List[float]] = None,
    chunk_chars: int = 1200,
    overlap: int = 150,
    _progress_parent=None
) -> Tuple[pd.DataFrame, Dict[str, Dict], Dict[str, List[Tuple[str, float]]], Dict[Tuple[str,str], Tuple[str,float]]]:
    criteria = [normalize_ws(c) for c in (criteria or []) if normalize_ws(c)]
    if not criteria or not candidates:
        return pd.DataFrame(), {}, {}, {}

    if weights is None or len(weights) != len(criteria):
        weights = [1.0] * len(criteria)
    weights = np.array(weights, dtype=float)
    weights = weights / (weights.sum() if weights.sum() > 0 else len(weights))

    all_chunk_texts, chunk_index = [], []
    for i, c in enumerate(candidates):
        chunks = chunk_text(c.text, max_chars=chunk_chars, overlap=overlap)
        for j, ch in enumerate(chunks):
            all_chunk_texts.append(ch); chunk_index.append((i, j))

    info = load_embedder()
    chunk_embs, vec_or_meta = prepare_corpus_embeddings(all_chunk_texts, info)

    cand_to_rows = {i: [] for i in range(len(candidates))}
    for ridx, (ci, cj) in enumerate(chunk_index):
        cand_to_rows[ci].append(ridx)

    coverage_records, insights_map, snippets_map = [], {}, {}
    evidence_map: Dict[Tuple[str,str], Tuple[str,float]] = {}

    progress = (_progress_parent.progress(0.0, text="Scoring…") if _progress_parent is not None else None)
    if progress is None:
        class _NoProgress:
            def progress(self, *args, **kwargs):
                return self
            def empty(self):
                pass
        progress = _NoProgress()
    total = max(1, len(candidates))

    for i, cand in enumerate(candidates):
        rows = cand_to_rows[i]
        cand_chunks = [all_chunk_texts[r] for r in rows]
        if len(cand_chunks) == 0:
            crit_scores = np.zeros(len(criteria)); argmax = np.zeros(len(criteria), dtype=int)
        else:
            cand_chunk_embs = chunk_embs[rows] if not isinstance(chunk_embs, np.ndarray) else chunk_embs[rows, :]
            crit_scores, argmax = compute_max_similarity_to_chunks(criteria, cand_chunk_embs, info, vec_or_meta)

        overall = float((crit_scores * weights).sum())
        row = {"Candidate": cand.name, "Overall": overall}
        for c_name, s in zip(criteria, crit_scores):
            row[c_name] = float(s)
        coverage_records.append(row)

        order = np.argsort(-crit_scores)
        top = [f"{criteria[idx]} — strong alignment ({crit_scores[idx]:.2f})" for idx in order[:3]]
        gaps = [f"{criteria[idx]} — weak coverage ({crit_scores[idx]:.2f})" for idx in order[-3:]]
        notes = f"Overall weighted score: {overall:.2f}. Doc length: {len(cand.text):,} chars."
        insights_map[cand.name] = {"top": top, "gaps": gaps, "notes": notes}

        top_snips = []
        for qi, best_idx in enumerate(argmax):
            if cand_chunks:
                ch = cand_chunks[int(best_idx)]
                top_snips.append((ch[:600] + ("…" if len(ch) > 600 else ""), float(crit_scores[qi])))
                evidence_map[(cand.name, criteria[qi])] = (ch[:800], float(crit_scores[qi]))
        seen, uniq_snips = set(), []
        for snip, sc in sorted(top_snips, key=lambda x: -x[1]):
            key = snip[:80]
            if key not in seen:
                uniq_snips.append((snip, sc)); seen.add(key)
            if len(uniq_snips) >= 5: break
        snippets_map[cand.name] = uniq_snips

        progress.progress((i + 1) / total, text=f"Scoring {cand.name}…")

    try:
        progress.progress(1.0, text="Scoring complete.")
        progress.empty()
    except Exception:
        pass

    coverage_df = pd.DataFrame(coverage_records)
    if not coverage_df.empty:
        coverage_df.sort_values(by="Overall", ascending=False, inplace=True)
        coverage_df.reset_index(drop=True, inplace=True)

    return coverage_df, insights_map, snippets_map, evidence_map

# -----------------------------
# Criteria parse & weights
# -----------------------------
def parse_criteria_text(criteria_text: str) -> List[str]:
    items = [normalize_ws(x) for x in (criteria_text or "").splitlines()]
    return [x for x in items if x]

def get_weights(criteria: List[str], mode: str, weights_csv: str) -> List[float]:
    if mode == "Uniform" or not criteria:
        return [1.0] * len(criteria)
    lines = [l.strip() for l in (weights_csv or "").splitlines() if l.strip()]
    if not lines:
        return [1.0] * len(criteria)
    mapping, vals = {}, []
    for line in lines:
        if "," in line:
            k, v = line.split(",", 1)
            try: mapping[normalize_ws(k)] = float(v)
            except Exception: pass
        else:
            try: vals.append(float(line))
            except Exception: pass
    if mapping:
        return [float(mapping.get(c, 1.0)) for c in criteria]
    if vals and len(vals) == len(criteria):
        return [float(x) for x in vals]
    return [1.0] * len(criteria)

# -----------------------------
# Report export
# -----------------------------
def _md_table_or_fallback(df: pd.DataFrame) -> str:
    if _tabulate is None:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)
    return df.to_markdown(index=False)

def to_markdown_report(coverage: pd.DataFrame, insights: Dict[str, Dict], jd_text: str) -> str:
    if coverage.empty:
        return "# Candidate Report\n\n_No analysis available._"
    lines = ["# Candidate Analysis Report", ""]
    lines.append("## Job Description (excerpt)")
    jd_excerpt = (jd_text or "").strip().splitlines()
    jd_excerpt = "\n".join(jd_excerpt[:20])
    lines.append("" if not jd_excerpt else jd_excerpt)
    lines.append("")
    lines.append("## Leaderboard (Overall)")
    top_tbl = coverage[["Candidate", "Overall"]].copy()
    top_tbl["Overall"] = top_tbl["Overall"].map(lambda x: f"{x:.2f}")
    lines.append(_md_table_or_fallback(top_tbl))
    lines.append("")
    lines.append("## Coverage Matrix")
    cov_tbl = coverage.copy()
    for c in cov_tbl.columns:
        if c != "Candidate":
            cov_tbl[c] = cov_tbl[c].map(lambda x: f"{x:.2f}")
    lines.append(_md_table_or_fallback(cov_tbl))
    lines.append("")
    lines.append("## Candidate Insights")
    for name, info in insights.items():
        lines.append(f"### {name}")
        lines.append("**Top strengths**:")
        for t in info.get("top", []): lines.append(f"- {t}")
        lines.append("**Gaps / risks**:")
        for g in info.get("gaps", []): lines.append(f"- {g}")
        notes = info.get("notes", "")
        if notes: lines.append(f"_Notes:_ {notes}")
        lines.append("")
    return "\n".join(lines)

def to_pdf_bytes_from_markdown(md_text: str) -> Optional[bytes]:
    if reportlab is None: return None
    buf = io.BytesIO(); c = rl_canvas.Canvas(buf, pagesize=A4)
    left = 2.0 * cm; top = A4[1] - 2.0 * cm
    wrapped = []
    for para in (md_text or "").splitlines():
        wrapped.extend(textwrap.wrap(para, width=90) or [""])
    y = top
    for line in wrapped:
        if y < 2.0 * cm:
            c.showPage(); y = top
        c.drawString(left, y, line); y -= 12
    c.save(); buf.seek(0)
    return buf.getvalue()

# -------- GPT Insights helpers --------

def _redact(s: str) -> str:
    """Redact emails, phones, URLs. Keeps text readable."""
    if not s:
        return s
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[-email]", s)
    s = re.sub(r"\b(?:\+?\d[\d \-\(\)]{8,}\d)\b", "[-phone]", s)
    s = re.sub(r"https?://\S+|www\.\S+", "[-url]", s)
    return s

def _model_for_engine(engine: str) -> Optional[str]:
    if engine == "gpt-4o":
        return "gpt-4o"
    if engine == "gpt-4o":
        return "gpt-4o"
    return None

def gpt_candidate_insights(
    candidate_name: str,
    candidate_text: str,
    jd_text: str,
    criteria: List[str],
    coverage_row: Dict[str, float],
    evidence_map: Dict[Tuple[str,str], Tuple[str,float]]
) -> Dict[str, Any]:
    """
    Returns dict with keys: {'top': [...], 'gaps': [...], 'notes': str}
    Uses JSON schema to keep output consistent. Always uses GPT-4o.
    """
    model = "gpt-4o"

    # Build evidence lines from coverage scores and evidence_map
    ev_items = []
    for c in criteria:
        sc = float(coverage_row.get(c, 0.0))
        snip = evidence_map.get((candidate_name, c), ("", sc))[0]
        _snip_clean = (snip or "")[:400].replace("\n", " ")
        ev_items.append(f"- {c} (score {sc:.2f}): {_snip_clean}")
    ev_lines = ev_items

    cand_body = candidate_text
    jd_body = jd_text

    system = (
        "You are a hiring analyst. Produce concise, high-signal insights about a candidate "
        "relative to the provided Job Description and evidence. Be specific and avoid fluff."
    )
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
    user = (
        f"Job Description (excerpted):\n{jd_body[:3000]}\n\n"
        f"Candidate: {candidate_name}\n"
        f"Candidate content (excerpted):\n{cand_body[:3000]}\n\n"
        "Evidence by criterion (top 10 by similarity):\n"
        + "\n".join(ev_lines) +
        "\n\nTask:\n"
        "- Provide 3–6 bullet **Top strengths** tied to criteria and tangible evidence.\n"
        "- Provide 3–6 bullet **Gaps / risks** with rationale.\n"
        "- Provide a short **Notes** paragraph (2–4 sentences) with an overall view."
    )

    try:
        data = call_llm_json(system, user, schema, model=model)
        # Normalize
        return {
            "top": [normalize_ws(x) for x in data.get("top", []) if normalize_ws(x)],
            "gaps": [normalize_ws(x) for x in data.get("gaps", []) if normalize_ws(x)],
            "notes": normalize_ws(data.get("notes", "")),
        }
    except Exception:
        return {}

def _update_stage(msg: str):
    """Placeholder for status updates during analysis"""
    pass



with st.sidebar:
    st.markdown("## 🧭 Navigation")
    
    # Initialize page if not set
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Upload & Analyse"
    
    # Radio navigation - better use of vertical space
    pages = ["Upload & Analyse", "Scoring Guide", "Criteria from JD", "Scoring Analysis", "Candidate Insights", "Export", "PDF Tools", "Settings"]
    
    selected_page = st.radio(
        "Navigation",
        pages,
        key="nav_radio",
        label_visibility="collapsed"
    )
    
    # Only update state if changed
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page

# -----------------------------
# File Processing (needs to happen before page rendering)
# -----------------------------
# Initialize variables
jd_file = None
jd_manual = ""
uploaded = []

# Only process uploads on the Upload & Analyse page to keep them in scope
# But we need to maintain state across page navigation

# Only process uploads on the Upload & Analyse page to keep them in scope
# But we need to maintain state across page navigation

# -----------------------------
# Page Rendering Based on Selection
# -----------------------------
current_page = st.session_state.get("current_page", "Upload & Analyse")

if current_page == "Upload & Analyse":
    # Status container for analysis progress (at the very top)
    st.session_state._status_container = st.empty()
    
    # Add spacing below header
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # Simplified, cleaner layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📄 Job Description (JD)")
        jd_file = st.file_uploader("Upload JD", type=["pdf","docx","txt"], key="jd_upl", accept_multiple_files=False, label_visibility="collapsed")
        
        with st.expander("✏️ Paste JD text manually"):
            jd_manual = st.text_area("JD text", value=st.session_state.get("jd_manual_text",""), height=180, label_visibility="collapsed")
            st.session_state.jd_manual_text = jd_manual
    
    with col2:
        st.markdown("#### 👥 Candidate Resumes")
        uploaded = st.file_uploader("Upload resumes", accept_multiple_files=True, type=["pdf","docx","txt"], key="cand_upl", label_visibility="collapsed")
    
    # Show currently loaded files with option to remove
    current_jd = st.session_state.get("jd")
    current_candidates = st.session_state.get("cached_candidates", [])
    
    if current_jd or current_candidates:
        with st.expander("📋 Currently Loaded Files", expanded=False):
            if current_jd:
                st.markdown("**Job Description:**")
                col_jd, col_jd_btn = st.columns([4, 1])
                with col_jd:
                    st.text(current_jd.file_name)
                with col_jd_btn:
                    if st.button("❌", key="remove_jd", help="Remove JD"):
                        st.session_state.jd = None
                        st.session_state.last_coverage = pd.DataFrame()
                        st.session_state.last_insights = {}
                        st.rerun()
            
            if current_candidates:
                st.markdown("**Candidates:**")
                for idx, cand in enumerate(current_candidates):
                    col_name, col_file, col_btn = st.columns([2, 2, 1])
                    with col_name:
                        st.text(cand.name)
                    with col_file:
                        st.caption(cand.file_name)
                    with col_btn:
                        if st.button("❌", key=f"remove_cand_{idx}", help=f"Remove {cand.name}"):
                            # Remove this candidate from the list
                            updated_candidates = [c for c in current_candidates if c.hash != cand.hash]
                            st.session_state.cached_candidates = updated_candidates
                            # Clear analysis results since data changed
                            st.session_state.last_coverage = pd.DataFrame()
                            st.session_state.last_insights = {}
                            st.rerun()
    
    # Analyse button with spacing
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    if st.button("🔍 Analyse / Update", type="primary", width='stretch'):
        st.session_state._trigger_analyse = True
        st.rerun()
    
    # JD assignment from file or manual text
    if jd_file is not None:
        b = jd_file.read()
        if jd_file.name.lower().endswith('.pdf'):
            if st.session_state.get("use_layout_pdf", True):
                try:
                    jd_text = extract_text_layout_aware(b, jd_file.name)
                    if not jd_text.strip():
                        jd_text = read_file_bytes(b, jd_file.name)
                except Exception:
                    jd_text = read_file_bytes(b, jd_file.name)
            else:
                jd_text = read_file_bytes(b, jd_file.name)
        else:
            jd_text = read_file_bytes(b, jd_file.name)
        st.session_state.jd = JD(file_name=jd_file.name, text=jd_text, hash=hash_bytes(b), raw_bytes=b)
    elif jd_manual.strip():
        st.session_state.jd = JD(file_name="Manual JD", text=jd_manual, hash=hash_text(jd_manual), raw_bytes=None)

    # Candidates ingest
    new_candidates: List[Candidate] = []
    if uploaded:
        for uf in uploaded:
            b = uf.read()
            if uf.name.lower().endswith('.pdf'):
                if st.session_state.get("use_layout_pdf", True):
                    try:
                        text = extract_text_layout_aware(b, uf.name)
                        if not text.strip():
                            text = read_file_bytes(b, uf.name)
                    except Exception:
                        text = read_file_bytes(b, uf.name)
                else:
                    text = read_file_bytes(b, uf.name)
            else:
                text = read_file_bytes(b, uf.name)
            name_guess = infer_candidate_name(uf.name, text)
            new_candidates.append(Candidate(name=name_guess, file_name=uf.name, text=text, hash=hash_bytes(b), raw_bytes=b))
    if new_candidates:
        existing = {c.hash: c for c in st.session_state.get("cached_candidates", [])}
        for c in new_candidates: existing[c.hash] = c
        st.session_state.cached_candidates = list(existing.values())
    
    # Overview/Status Section with better visual hierarchy
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # Custom styling for smaller metrics
    st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 1.2rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        jd_status = "📄 Loaded" if st.session_state.get("jd") else "⚠️ Not loaded"
        st.metric("Job Description", jd_status)
        if st.session_state.get("jd"):
            st.caption(f"{len(st.session_state.jd.text):,} characters")
    
    with col_b:
        num_cands = len(st.session_state.get("cached_candidates", []))
        st.metric("Candidates", num_cands)
    
    with col_c:
        covdf = st.session_state.get("last_coverage", pd.DataFrame())
        num_criteria = len([c for c in covdf.columns if c not in ('Candidate','Overall')]) if isinstance(covdf, pd.DataFrame) and not covdf.empty else 0
        st.metric("Criteria", num_criteria)
    
    # Show top results if analysis has been run
    if isinstance(covdf, pd.DataFrame) and not covdf.empty:
        # Subtle info text explaining scores - full width above results
        st.markdown(
            "<div style='padding: 0.5rem 0; color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;'>"
            "Scores indicate the strength of the match between candidate resumes and the JD criteria. 0.00 = no match, 1.00 = perfect match.<br>"
            "Icons indicate score strength in 3 ranges: ✅ Strong (≥0.70) | ⚠️ Moderate (0.45–0.70) | ⛔ Weak (&lt;0.45)"
            "</div>",
            unsafe_allow_html=True
        )
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.markdown("### 🏆 Top Candidates")
            try:
                top3 = covdf[["Candidate","Overall"]].head(3)
                hi = st.session_state.get("cov_hi", 0.70)
                lo = st.session_state.get("cov_lo", 0.45)
                for idx, r in top3.iterrows():
                    score = r['Overall']
                    icon = "✅" if score >= hi else ("⚠️" if score >= lo else "⛔")
                    st.write(f"**{idx+1}.** {r['Candidate']} — score: {score:.2f} {icon}")
            except Exception:
                pass
        
        with result_col2:
            st.markdown("### 📈 Top Matched Criteria")
            try:
                # Calculate average scores by criterion
                crit_cols = [c for c in covdf.columns if c not in ('Candidate','Overall')]
                avg_by_criterion = {}
                for crit in crit_cols:
                    scores = covdf[crit].tolist()
                    avg_by_criterion[crit] = sum(scores) / len(scores) if scores else 0
                
                # Display top 5 criteria
                sorted_crit = sorted(avg_by_criterion.items(), key=lambda x: x[1], reverse=True)
                top_5 = sorted_crit[:5]
                
                hi = st.session_state.get("cov_hi", 0.70)
                lo = st.session_state.get("cov_lo", 0.45)
                for crit, score in top_5:
                    icon = "✅" if score >= hi else ("⚠️" if score >= lo else "⛔")
                    st.write(f"• {crit}: {score:.2f} {icon}")
            except Exception:
                pass

elif current_page == "Criteria from JD":
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("Criteria from Job Description (JD)")
    
    # Check if JD exists
    if not st.session_state.get("jd"):
        st.warning("Upload a JD file or paste JD text on the **Upload & Analyse** tab to get started.")
        st.stop()
    
    jd_obj = st.session_state.jd
    
    # Auto-extract sections and criteria (GPT-only) for current JD if not already present
    if API_KEY_SET:
        need_extract = (
            "extracted_hash" not in st.session_state
            or st.session_state.get("extracted_hash") != jd_obj.hash
        )
        if need_extract:
            secs = cached_extract_jd_sections_with_gpt(jd_obj.hash, jd_obj.text)
            st.session_state["extracted_sections"] = _sections_to_dict(secs)
            crits, cat_map = build_criteria_from_gpt_sections(secs, per_section=999, cap_total=10000)
            st.session_state["cat_map"] = cat_map
            st.session_state["criteria_text"] = "\n".join(crits)
            st.session_state["extracted_hash"] = jd_obj.hash
    
    # Compact, collapsible section with all JD details
    with st.expander("📋 Job Description & Extraction Details", expanded=False):
        # Summary metrics in compact row
        col1, col2, col3, col4 = st.columns(4)
        num_criteria = len([l for l in (st.session_state.get("criteria_text","") or "").splitlines() if l.strip()])
        cat_map = st.session_state.get("cat_map", {})
        num_categories = len(set(cat_map.values())) if cat_map else 0
        
        with col1:
            st.metric("JD File", jd_obj.file_name.split('.')[0][:12] + "..." if len(jd_obj.file_name) > 15 else jd_obj.file_name)
        with col2:
            st.metric("Characters", f"{len(jd_obj.text):,}")
        with col3:
            st.metric("Criteria", num_criteria)
        with col4:
            st.metric("Categories", num_categories if num_categories > 0 else "Uncategorized")
        
        st.markdown("---")
        
        # View original file inline (in collapsible section)
        if jd_obj.raw_bytes:
            if jd_obj.file_name.lower().endswith('.pdf'):
                # PDF viewer in expander
                with st.expander("📄 View Original PDF File", expanded=False):
                    import base64
                    base64_pdf = base64.b64encode(jd_obj.raw_bytes).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                    st.caption("💡 Use the browser's PDF controls to zoom, navigate pages, or download if needed.")
            elif jd_obj.file_name.lower().endswith(('.docx', '.doc')):
                with st.expander("📝 Original DOCX/DOC File", expanded=False):
                    st.info("DOCX/DOC files cannot be previewed inline. View the 'Full Text Extract' below, or download the file to open in Word.", icon="ℹ️")
                    # Provide download button for DOCX
                    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if jd_obj.file_name.lower().endswith('.docx') else "application/msword"
                    st.download_button(
                        "📥 Download to View in Word",
                        data=jd_obj.raw_bytes,
                        file_name=jd_obj.file_name,
                        mime=mime_type,
                        help="Download to view in Microsoft Word"
                    )
            else:
                # Plain text files in expander
                with st.expander("📄 View Original Text File", expanded=False):
                    st.code(jd_obj.text, language=None, line_numbers=False)
        
        st.markdown("---")
        
        # Extracted sections (if any)
        secs_dict = st.session_state.get("extracted_sections", None)
        if secs_dict:
            st.markdown("**📋 Extracted Sections (auto from GPT)**")
            sec_col1, sec_col2 = st.columns(2)
            with sec_col1:
                st.markdown("**🎯 Key skills**")
                for x in secs_dict.get("key_skills", []): st.markdown(f"- {x}")
                st.markdown("**📌 Responsibilities**")
                for x in secs_dict.get("responsibilities", []): st.markdown(f"- {x}")
            with sec_col2:
                st.markdown("**🎓 Qualifications**")
                for x in secs_dict.get("qualifications", []): st.markdown(f"- {x}")
                st.markdown("**💼 Experience required**")
                for x in secs_dict.get("experience_required", []): st.markdown(f"- {x}")
            
            st.markdown("---")
        elif not API_KEY_SET:
            st.info("💡 Set `OPENAI_API_KEY` in your environment to auto-extract sections and build criteria.", icon="ℹ️")
            st.markdown("---")
        
        # Full text extract
        st.markdown("**📄 Full Text Extract**")
        st.text_area("Full text extracted from JD", value=jd_obj.text, height=200, label_visibility="collapsed", key="jd_text_preview")

    # --- Review & edit criteria ---
    st.markdown("### ✏️ Review & Edit Criteria")
    
    init_lines = [l for l in (st.session_state.get("criteria_text","") or "").splitlines() if l.strip()]
    cat_map = st.session_state.get("cat_map", {})
    
    # Show count and category breakdown
    num_criteria = len(init_lines)
    categories = set(cat_map.get(c, "Uncategorized") for c in init_lines)
    
    st.caption(f"Editing **{num_criteria}** criteria across **{len(categories)}** categories")
    
    st.info("""💡 **Tip:** Add, remove, or reword criteria. Uncheck **Use** to exclude items. Changes take effect after clicking "Save & Re-run Analysis".""", icon="ℹ️")

    # Prepare editable table (Use, Category, Criterion)
    import pandas as pd
    rows = []
    for c in init_lines:
        rows.append({"Use": True, "Category": cat_map.get(c, ""), "Criterion": c})
    if not rows:
        rows = [{"Use": True, "Category": "", "Criterion": ""}]
    df = pd.DataFrame(rows, columns=["Use","Category","Criterion"])

    edited = st.data_editor(
        df,
        width='stretch',
        num_rows="dynamic",
        column_config={
            "Use": st.column_config.CheckboxColumn("Use", help="Uncheck to exclude from analysis."),
            "Category": st.column_config.TextColumn("Category", help="Optional grouping label for display order."),
            "Criterion": st.column_config.TextColumn("Criterion", help="Edit the text of the criterion."),
        },
        hide_index=True,
    )

    
    # --- Action buttons in card-style layout ---
    st.markdown("#### ⚙️ Actions")

    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        st.markdown("**📤 Export Criteria**")
        st.caption("Save selected criteria to CSV")
        # Export selected rows as CSV with Criterion,Category
        import pandas as _pd
        try:
            _rows = [
                {"Criterion": str(c).strip(),
                "Category": (str(cat).strip() if str(cat).strip() else "")}
                for u, c, cat in zip(
                    edited["Use"],
                    edited["Criterion"],
                    edited.get("Category", [""] * len(edited)),
                )
                if u and str(c).strip()
            ]
        except Exception:
            _rows = [
                {"Criterion": ln, "Category": ""}
                for ln in (st.session_state.get("criteria_text", "") or "").splitlines()
                if ln.strip()
            ]
        _csv = _pd.DataFrame(_rows, columns=["Criterion", "Category"]).to_csv(index=False)
        st.download_button(
            "Export to CSV",
            data=_csv,
            file_name="criteria.csv",
            mime="text/csv",
            use_container_width=True
        )

    with action_col2:
        st.markdown("**📥 Import Criteria**")
        st.caption("Load from CSV file")
        _uploaded = st.file_uploader(
            "Upload CSV", type=["csv"], accept_multiple_files=False, key="crit_csv_upl", label_visibility="collapsed"
        )
        if st.button("Import & Replace", key="import_replace_btn", use_container_width=True):
            import pandas as _pd
            if _uploaded is None:
                st.warning("Upload a CSV file first.")
            else:
                try:
                    _df = _pd.read_csv(_uploaded)

                    # Normalize expected columns
                    rename_map = {}
                    for cn in _df.columns:
                        cl = str(cn).lower().strip()
                        if cl in ("criterion", "criteria"):
                            rename_map[cn] = "Criterion"
                        elif cl in ("category", "cat"):
                            rename_map[cn] = "Category"
                    if rename_map:
                        _df = _df.rename(columns=rename_map)

                    if "Criterion" not in _df.columns:
                        st.error("CSV must include a 'Criterion' column.")
                    else:
                        _df["Criterion"] = (
                            _df["Criterion"].fillna("").astype(str).str.strip()
                        )
                        if "Category" not in _df.columns:
                            _df["Category"] = ""
                        _df["Category"] = (
                            _df["Category"].fillna("").astype(str).str.strip()
                        )
                        _df = _df[_df["Criterion"] != ""]
                        crit_list = _df["Criterion"].tolist()
                        cat_map = {
                            r["Criterion"]: r["Category"]
                            for _, r in _df.iterrows()
                            if str(r["Category"]).strip()
                        }
                        if crit_list:
                            st.session_state["criteria_text"] = "\n".join(crit_list)
                            st.session_state["cat_map"] = cat_map
                            st.session_state["_trigger_analyse"] = True
                            st.rerun()
                        else:
                            st.warning("No valid criteria found in CSV.")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")

    with action_col3:
        st.markdown("**🔄 Reset Criteria**")
        st.caption("Restore to auto-generated from JD")
        if st.button("Reset to Auto-Generated", use_container_width=True):
            if st.session_state.get("jd") and API_KEY_SET:
                _jd = st.session_state.jd
                _secs = cached_extract_jd_sections_with_gpt(_jd.hash, _jd.text)
                _crits, _cat_map = build_criteria_from_gpt_sections(_secs, per_section=999, cap_total=10000)
                st.session_state["extracted_sections"] = _sections_to_dict(_secs)
                st.session_state["criteria_text"] = "\n".join(_crits)
                st.session_state["cat_map"] = _cat_map
                st.session_state["_trigger_analyse"] = True
                st.rerun()
            else:
                st.warning("Cannot reset without a JD and API key.")

 
    # Primary save button at bottom
    save_col1, save_col2 = st.columns([1, 3])
    with save_col1:
        if st.button("💾 Save & Re-run Analysis", type="primary", use_container_width=True):
            # Persist edited criteria & categories
            new_use = list(edited["Use"])
            new_crits = list(edited["Criterion"])
            new_cats = list(edited["Category"]) if "Category" in edited else [""] * len(new_crits)

            kept = []
            new_cat_map = {}
            for used, crit, cat in zip(new_use, new_crits, new_cats):
                crit_s = str(crit).strip()
                if used and crit_s:
                    kept.append(crit_s)
                    if cat and str(cat).strip():
                        new_cat_map[crit_s] = str(cat).strip()
            st.session_state["criteria_text"] = "\n".join(kept)
            # Merge existing cat_map to retain known mappings unless overridden
            old_map = st.session_state.get("cat_map", {})
            old_map.update(new_cat_map)
            st.session_state["cat_map"] = {k:v for k,v in old_map.items() if k in kept}

            # Trigger fresh analysis
            st.session_state["_trigger_analyse"] = True
            st.rerun()
    
    with save_col2:
        pass

elif current_page == "Scoring Analysis":
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    coverage_df = st.session_state.get("last_coverage", pd.DataFrame())
    
    # Header row with summary and top candidates in corner
    header_col1, header_col2 = st.columns([2, 1])
    
    with header_col1:
        st.subheader("Scoring Analysis")
        if not isinstance(coverage_df, pd.DataFrame) or coverage_df.empty:
            st.warning("Upload JD/criteria and candidate resumes, then click Analyse.")
        else:
            # Compact summary line
            num_candidates = len(coverage_df)
            num_criteria = len([c for c in coverage_df.columns if c not in ("Candidate","Overall")])
            st.markdown(f"Analyzing **{num_candidates} candidates** across **{num_criteria} criteria**")
    
    with header_col2:
        # Compact top candidates list in corner (only if we have data)
        # Will be populated after checkbox is rendered
        top_candidates_placeholder = st.empty()
    
    # Only continue if we have data
    if not isinstance(coverage_df, pd.DataFrame) or coverage_df.empty:
        st.stop()
    
    # Info box with scoring guide - full width, below header, smaller font
    st.markdown(
        "<div style='font-size: 0.9rem;'>"
        "<div style='padding: 0.75rem; background-color: #e7f3ff; border-left: 4px solid #1f77b4; border-radius: 0.25rem; margin-bottom: 1rem;'>"
        "<strong>ℹ️ Scores:</strong> Each score (0.00–1.00) measures how well a candidate's resume matches a criterion. "
        "Colors/icons indicate strength: ✅ Strong (≥0.70) | ⚠️ Moderate (0.45–0.70) | ⛔ Weak (&lt;0.45)"
        "</div></div>",
        unsafe_allow_html=True
    )
    
    # Get threshold settings (used throughout this page)
    hi = st.session_state.get("cov_hi", 0.70)
    lo = st.session_state.get("cov_lo", 0.45)
    
    # Compact single-row controls (no header needed - matrix is obvious)
    st.markdown("""
        <style>
        .compact-controls {
            font-size: 0.85rem;
            color: #666;
            margin: 0.5rem 0;
        }
        /* Make controls more compact */
        div[data-testid="stHorizontalBlock"] > div {
            gap: 0.5rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([2, 1.5, 1.5, 1, 0.8])
    
    with ctrl_col1:
        sort_options = ["Name (A-Z)", "Overall Score (High-Low)", "Overall Score (Low-High)"]
        # Reset index if it's out of range (e.g., from previous session with more options)
        default_idx = st.session_state.get("cov_sort_idx", 1)
        if default_idx >= len(sort_options):
            default_idx = 1
        sort_by = st.selectbox("Sort candidates by", 
                              sort_options,
                              index=default_idx,
                              key="sort_select",
                              label_visibility="visible")
        st.session_state.cov_sort_idx = sort_options.index(sort_by)
    
    with ctrl_col2:
        filter_options = ["All", "Top 3", "Top 5", "Top 10", "Custom Selection"]
        show_top_n = st.selectbox("Show", 
                                 filter_options,
                                 index=0,
                                 key="filter_select",
                                 label_visibility="visible")
    
    # Custom candidate selection (only shown if "Custom Selection" is chosen)
    selected_candidates = None
    if show_top_n == "Custom Selection":
        all_candidates = coverage_df["Candidate"].tolist()
        st.markdown("**👥 Select candidates to compare:**")
        selected_candidates = st.multiselect(
            "Select candidates",
            all_candidates,
            default=all_candidates[:3] if len(all_candidates) >= 3 else all_candidates,
            key="custom_candidate_selector",
            label_visibility="collapsed"
        )
        if not selected_candidates:
            st.warning("⚠️ Select at least one candidate to display.")
    
    with ctrl_col3:
        expand_categories = st.checkbox("Expand categories", value=st.session_state.get("cov_expand_all", True))
        st.session_state.cov_expand_all = expand_categories
    
    with ctrl_col4:
        # Checkbox that syncs with session state - let Streamlit manage the state naturally
        show_scores_checked = st.checkbox(
            "Scores", 
            value=st.session_state.get("cov_show_scores", False),
            help="Show numeric scores",
            key="cov_show_scores_checkbox"
        )
        # Update session state for next render
        st.session_state.cov_show_scores = show_scores_checked
        # Use the checkbox value directly (not session state)
        show_scores = show_scores_checked
    
    with ctrl_col5:
        # CSV download button placeholder - will be populated after sorting
        csv_placeholder = st.empty()
    
    # Now render top candidates with the current show_scores value
    with top_candidates_placeholder.container():
        if isinstance(coverage_df, pd.DataFrame) and not coverage_df.empty:
            st.markdown("##### 🏆 Top Candidates")
            
            # Sort for top candidates display
            top_5_df = coverage_df.sort_values("Overall", ascending=False).head(5)
            
            # Display as simple vertical list using current show_scores value
            for idx, row in top_5_df.iterrows():
                score = row["Overall"]
                icon = "✅" if score >= hi else ("⚠️" if score >= lo else "⛔")
                # Get position (1-5)
                pos = top_5_df.index.get_loc(idx) + 1
                if show_scores:
                    st.markdown(f"**{pos}.** {row['Candidate']} — {score:.2f} {icon}")
                else:
                    st.markdown(f"**{pos}.** {row['Candidate']} — {icon}")

    # Sort candidates based on selection
    if sort_by == "Overall Score (High-Low)":
        sorted_coverage = coverage_df.sort_values("Overall", ascending=False)
    elif sort_by == "Overall Score (Low-High)":
        sorted_coverage = coverage_df.sort_values("Overall", ascending=True)
    else:  # Name (A-Z)
        sorted_coverage = coverage_df.sort_values("Candidate")
    
    # Filter based on selection type
    if show_top_n == "Custom Selection":
        # Use custom multiselect
        if selected_candidates:
            sorted_coverage = sorted_coverage[sorted_coverage["Candidate"].isin(selected_candidates)]
        else:
            # No candidates selected - show nothing (warning already displayed above)
            sorted_coverage = pd.DataFrame()
    elif show_top_n != "All":
        # Use Top N filter
        n = int(show_top_n.split()[-1])
        sorted_coverage = sorted_coverage.head(n)
    
    # Stop here if no candidates to display (must check before accessing columns)
    if sorted_coverage.empty:
        st.stop()
    
    candidates = sorted_coverage["Candidate"].tolist()
    
    # Now populate the CSV download button in the placeholder
    with csv_placeholder:
        csv_data = sorted_coverage.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥",
            data=csv_data,
            file_name="coverage_matrix.csv",
            mime="text/csv",
            help="Download CSV",
            key="csv_download_btn"
            )

    # Hint text for sorting
    st.markdown(
        "<p style='font-size: 0.85rem; color: #666; margin: 0.5rem 0 1rem 0;'>"
        "💡 Hint: If you want to see which Criteria achieved the best (or worst) average scores, "
        "just click on the column header \"Average\" to sort by that column."
        "</p>",
        unsafe_allow_html=True
    )
    
    # Build coverage matrix with sorted/filtered candidates
    cat_map = st.session_state.get("cat_map", {})
    crit_cols = [c for c in sorted_coverage.columns if c not in ("Candidate","Overall")]
    mat = sorted_coverage.set_index("Candidate")[crit_cols].T
    ordered = sorted(crit_cols, key=lambda c: (cat_map.get(c, "zz"), c.lower()))
    
    candidates = sorted_coverage["Candidate"].tolist()
    overall_scores = {row["Candidate"]: row["Overall"] for _, row in sorted_coverage.iterrows()}
    
    # Build multi-index for rows including Overall as first row
    # Overall row will use ("📊 Overall", "Overall Score") as the index tuple
    disp = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [("📊 Overall", "Overall Score")] + [(cat_map.get(c,"uncategorised"), c) for c in ordered], 
            names=["Category","Criterion"]
        ), 
        columns=["Average"] + candidates
    )
    
    # Fill Overall row (no average for Overall, just candidate scores)
    for cand in candidates:
        overall = overall_scores[cand]
        icon = "✅" if overall >= hi else ("⚠️" if overall >= lo else "⛔")
        disp.loc[("📊 Overall", "Overall Score"), cand] = f"{overall:.2f} {icon}"  # Always show score (score first for proper sorting)
    disp.loc[("📊 Overall", "Overall Score"), "Average"] = ""  # Blank average for Overall
    
    # Fill in the matrix values with icons (and optional scores) for criteria
    for crit in ordered:
        # Calculate average score for this criterion across all displayed candidates
        scores_for_crit = [float(mat.loc[crit, cand]) for cand in candidates]
        avg_score = sum(scores_for_crit) / len(scores_for_crit) if scores_for_crit else 0.0
        
        # Format average column (score first for proper sorting)
        avg_icon = "✅" if avg_score >= hi else ("⚠️" if avg_score >= lo else "⛔")
        disp.loc[(cat_map.get(crit,"uncategorised"), crit), "Average"] = f"{avg_score:.2f} {avg_icon}" if show_scores else avg_icon
        
        # Format candidate columns (score first for proper sorting)
        for c in candidates:
            val = float(mat.loc[crit, c])
            icon = "✅" if val >= hi else ("⚠️" if val >= lo else "⛔")
            disp.loc[(cat_map.get(crit,"uncategorised"), crit), c] = f"{val:.2f} {icon}" if show_scores else icon
    
    # Display matrix by category with expandable sections
    if expand_categories:
        st.dataframe(disp, width='stretch')
    else:
        # Group by category for collapsible display
        categories = disp.index.get_level_values(0).unique()
        for category in categories:
            cat_data = disp.loc[category]
            with st.expander(f"📁 {category}", expanded=False):
                st.dataframe(cat_data, width='stretch')
    
    # Evidence Explorer - more prominent styling
    st.markdown("---")
    st.markdown("#### 🔍 Evidence Explorer")
    st.caption("Select a candidate and criterion to view the evidence snippet used for scoring")
    
    names = coverage_df["Candidate"].tolist()
    ordered = ordered  # ensure exists
    
    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        cand_choice = st.selectbox("👤 Candidate", names, index=0 if names else None)
    with ev_col2:
        crit = st.selectbox("📋 Criterion", ordered, index=0 if ordered else None)
    
    if crit and cand_choice:
        ev_map = st.session_state.get("evidence_map", {})
        snip, _ = ev_map.get((cand_choice, crit), (None, None))
        
        # Get the actual score from coverage_df, not evidence_map
        actual_score = None
        if crit in coverage_df.columns:
            cand_row = coverage_df[coverage_df["Candidate"] == cand_choice]
            if not cand_row.empty:
                actual_score = float(cand_row[crit].iloc[0])
        
        if snip is None or actual_score is None:
            st.info("No cached evidence yet for this pair. Re-run Analyse to refresh evidence.")
        else:
            # Show score with color coding (using actual score from coverage_df)
            score_color = "#28a745" if actual_score >= hi else ("#ffc107" if actual_score >= lo else "#dc3545")
            st.markdown(f"**Score:** <span style='color: {score_color}; font-size: 1.2rem; font-weight: bold;'>{actual_score:.2f}</span>", unsafe_allow_html=True)
            st.markdown("**Evidence from resume:**")
            st.code(snip, language=None)

elif current_page == "Candidate Insights":
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("Candidate Insights")
    
    if not API_KEY_SET:
        st.error("OpenAI API key is required for Insights. Set `OPENAI_API_KEY` to enable this tab.")
        st.stop()
    
    insights_map = st.session_state.get("last_insights", {})
    covdf = st.session_state.get("last_coverage", pd.DataFrame())
    
    if not insights_map or (isinstance(covdf, pd.DataFrame) and covdf.empty):
        st.info("Run analysis first to view candidate insights.")
        st.stop()
    
    # Get threshold settings for color coding
    hi = st.session_state.get("cov_hi", 0.70)
    lo = st.session_state.get("cov_lo", 0.45)
    cat_map = st.session_state.get("cat_map", {})
    
    # Sort candidates by Overall score (descending) and add rank
    covdf_sorted = covdf.sort_values("Overall", ascending=False).reset_index(drop=True)
    covdf_sorted["Rank"] = range(1, len(covdf_sorted) + 1)
    
    # Build candidate options with rank and score: "Name (Score: 0.85) ⭐ #1"
    candidate_options = []
    for _, row in covdf_sorted.iterrows():
        name = row["Candidate"]
        score = row["Overall"]
        rank = row["Rank"]
        option_text = f"{name} (Score: {score:.2f}) ⭐ #{rank}"
        candidate_options.append(option_text)
    
    # Initialize or get the current selection index
    if "insights_selected_idx" not in st.session_state:
        st.session_state.insights_selected_idx = 0
    
    # Compact selector with navigation
    selector_col1, nav_col1, nav_col2 = st.columns([4, 1, 1])
    
    with nav_col1:
        if st.button("⬅️ Previous", disabled=(st.session_state.insights_selected_idx == 0), use_container_width=True):
            st.session_state.insights_selected_idx = max(0, st.session_state.insights_selected_idx - 1)
            st.rerun()
    
    with nav_col2:
        if st.button("Next ➡️", disabled=(st.session_state.insights_selected_idx == len(candidate_options) - 1), use_container_width=True):
            st.session_state.insights_selected_idx = min(len(candidate_options) - 1, st.session_state.insights_selected_idx + 1)
            st.rerun()
    
    # Candidate selector with search (selectbox is searchable by default)
    with selector_col1:
        selected_option = st.selectbox(
            "Select a candidate to view their detailed insights",
            options=candidate_options,
            index=st.session_state.insights_selected_idx,
            help="Search by name or browse by rank. Sorted by overall score (best first)."
        )
    
    # Update the index if user manually selected from dropdown
    selected_idx = candidate_options.index(selected_option)
    if selected_idx != st.session_state.insights_selected_idx:
        st.session_state.insights_selected_idx = selected_idx
    
    # Use the session state index to get candidate info (this is the source of truth)
    current_idx = st.session_state.insights_selected_idx
    selected_name = covdf_sorted.iloc[current_idx]["Candidate"]
    selected_score = covdf_sorted.iloc[current_idx]["Overall"]
    selected_rank = covdf_sorted.iloc[current_idx]["Rank"]
    
    # Calculate criteria breakdown
    crit_cols = [c for c in covdf.columns if c not in ("Candidate", "Overall")]
    selected_row = covdf[covdf["Candidate"] == selected_name].iloc[0]
    strong_count = sum(1 for c in crit_cols if selected_row[c] >= hi)
    moderate_count = sum(1 for c in crit_cols if lo <= selected_row[c] < hi)
    weak_count = sum(1 for c in crit_cols if selected_row[c] < lo)
    
    # Compact header with all key info in one line
    score_color = "#28a745" if selected_score >= hi else ("#ffc107" if selected_score >= lo else "#dc3545")
    st.markdown(f"""
    <div style='background-color: rgba(128,128,128,0.1); padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
        <h3 style='margin: 0 0 0.5rem 0;'>👤 {selected_name} — At a Glance</h3>
        <div style='display: flex; gap: 2rem; align-items: center;'>
            <div>
                <span style='font-size: 0.9rem; color: #666;'>Score:</span>
                <span style='color: {score_color}; font-size: 1.8rem; font-weight: bold; margin-left: 0.5rem;'>{selected_score:.2f}</span>
            </div>
            <div>
                <span style='font-size: 0.9rem; color: #666;'>Rank:</span>
                <span style='font-size: 1.8rem; font-weight: bold; margin-left: 0.5rem;'>#{selected_rank}</span>
                <span style='font-size: 0.85rem; color: #666;'> of {len(covdf_sorted)}</span>
            </div>
            <div>
                <span style='font-size: 0.9rem; color: #666;'>Criteria:</span>
                <span style='margin-left: 0.5rem;'>✅ {strong_count}</span>
                <span style='margin-left: 0.5rem;'>⚠️ {moderate_count}</span>
                <span style='margin-left: 0.5rem;'>⛔ {weak_count}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== GPT INSIGHTS (MAIN FEATURE) ==========
    st.markdown("### 💡 GPT-Powered Insights")
    
    info = insights_map.get(selected_name, {})
    
    insight_col1, insight_col2 = st.columns([1, 1])
    
    with insight_col1:
        st.markdown("#### ✅ Top Strengths")
        strengths = info.get("top", [])
        if strengths:
            for strength in strengths:
                st.markdown(f"- {strength}")
        else:
            st.info("No strengths identified yet. Click 'Refresh GPT Insights' above.")
    
    with insight_col2:
        st.markdown("#### ⚠️ Gaps / Risks")
        gaps = info.get("gaps", [])
        if gaps:
            for gap in gaps:
                st.markdown(f"- {gap}")
        else:
            st.info("No gaps identified yet. Click 'Refresh GPT Insights' above.")
    
    # Additional notes
    notes = info.get("notes", "")
    if notes:
        st.markdown("#### 📝 Additional Notes")
        st.info(notes)
    
    # ========== DETAILED SCORES ==========
    with st.expander("📈 Detailed Scores for All Criteria", expanded=False):
        st.markdown(
            "View individual criteria scores with color-coding "
            "(<span style='color: #28a745; font-weight: bold;'>strong</span>  -  "
            "<span style='color: #ffc107; font-weight: bold;'>moderate</span>  -  "
            "<span style='color: #dc3545; font-weight: bold;'>weak or missing</span>)",
            unsafe_allow_html=True
        )
        
        # Build scores dataframe for this candidate
        scores_data = []
        for crit in crit_cols:
            score = selected_row[crit]
            category = cat_map.get(crit, "Uncategorized")
            scores_data.append({"Category": category, "Criterion": crit, "Score": score})
        
        scores_df = pd.DataFrame(scores_data)
        
        # Group by category if categories exist
        if cat_map:
            st.markdown("**Scores grouped by category:**")
            for category in sorted(scores_df["Category"].unique()):
                cat_scores = scores_df[scores_df["Category"] == category]
                st.markdown(f"**📁 {category}**")
                for _, row in cat_scores.iterrows():
                    score = row["Score"]
                    crit = row["Criterion"]
                    color = "#28a745" if score >= hi else ("#ffc107" if score >= lo else "#dc3545")
                    st.markdown(f"- {crit}: <span style='color: {color}; font-weight: bold;'>{score:.2f}</span>", unsafe_allow_html=True)
                st.markdown("")
        else:
            # No categories, just list all
            for _, row in scores_df.iterrows():
                score = row["Score"]
                crit = row["Criterion"]
                color = "#28a745" if score >= hi else ("#ffc107" if score >= lo else "#dc3545")
                st.markdown(f"- {crit}: <span style='color: {color}; font-weight: bold;'>{score:.2f}</span>", unsafe_allow_html=True)
        
        # Show evidence snippets inline
        st.markdown("---")
        st.markdown("**🔍 View Evidence for Specific Criterion**")
        
        ev_crit = st.selectbox("Select criterion to see evidence snippet", crit_cols, key="insights_evidence_crit")
        
        if ev_crit:
            ev_map = st.session_state.get("evidence_map", {})
            snip, _ = ev_map.get((selected_name, ev_crit), (None, None))
            actual_score = selected_row[ev_crit]
            
            if snip:
                score_color_ev = "#28a745" if actual_score >= hi else ("#ffc107" if actual_score >= lo else "#dc3545")
                st.markdown(f"**Score:** <span style='color: {score_color_ev}; font-size: 1.2rem; font-weight: bold;'>{actual_score:.2f}</span>", unsafe_allow_html=True)
                st.markdown("**Evidence from resume:**")
                st.code(snip, language=None)
            else:
                st.info("No cached evidence for this criterion. Re-run analysis to refresh.")
        
        # Collapse helper at bottom
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: #666; font-size: 0.9rem;'>👆 Scroll to top to collapse this section</div>", unsafe_allow_html=True)
    
    # ========== RESUME ==========
    with st.expander("📄 Resume / CV", expanded=False):
        st.markdown("View the candidate's resume - both original file and extracted text")
        
        # Find the candidate object to get resume text
        cands = st.session_state.get("cached_candidates", [])
        cand_obj = None
        for c in cands:
            if c.name == selected_name:
                cand_obj = c
                break
        
        if cand_obj:
            st.markdown(f"**Filename:** {cand_obj.file_name}")
            
            # Show original file if it's a PDF
            if cand_obj.file_name.lower().endswith('.pdf'):
                with st.expander("📄 View Original PDF File", expanded=False):
                    import base64
                    base64_pdf = base64.b64encode(cand_obj.raw_bytes).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                    st.caption("💡 Use the browser's PDF controls to zoom, navigate pages, or download if needed.")
            elif cand_obj.file_name.lower().endswith(('.docx', '.doc')):
                with st.expander("📥 Original DOCX/DOC File", expanded=False):
                    st.info("DOCX/DOC files cannot be previewed inline. View the 'Extracted Text' below, or download the file to open in Word.", icon="ℹ️")
                    # Provide download button for DOCX
                    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if cand_obj.file_name.lower().endswith('.docx') else "application/msword"
                    st.download_button(
                        "📥 Download to View in Word",
                        data=cand_obj.raw_bytes,
                        file_name=cand_obj.file_name,
                        mime=mime_type,
                        help="Download to view in Microsoft Word"
                    )
            else:
                # Plain text files in expander
                with st.expander("📄 View Original Text File", expanded=False):
                    st.code(cand_obj.text, language=None, line_numbers=False)
            
            st.markdown("---")
            st.markdown("**📝 Extracted Text**")
            st.code(cand_obj.text, language=None)
        else:
            st.warning("Resume text not available for this candidate.")
        
        # Collapse helper at bottom
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: #666; font-size: 0.9rem;'>👆 Scroll to top to collapse this section</div>", unsafe_allow_html=True)

elif current_page == "Compare":
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("Compare Candidates vs JD")
    covdf = st.session_state.get("last_coverage", pd.DataFrame())
    names = covdf["Candidate"].tolist() if isinstance(covdf, pd.DataFrame) and not covdf.empty else []
    if not names:
        st.info("Run analysis to compare candidates.")
    else:
        sel = st.multiselect("Select 2–3 candidates", names, default=names[:2])
        if len(sel) < 2:
            st.warning("Select at least two candidates.")
        else:
            sub = covdf[covdf["Candidate"].isin(sel)].set_index("Candidate")
            st.write("**Comparison (scores)**")
            st.table(sub.reset_index().style.format({c: "{:.2f}" for c in sub.columns if c != "Candidate"}))

elif current_page == "Candidates":
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("Candidates")
    cands = st.session_state.get("cached_candidates", [])
    if not cands:
        st.info("Upload candidate resumes to view details.")
    else:
        # Build display options: "Label — Filename"
        options = [f"{c.name} — {c.file_name}" for c in cands]

        # Stronger, clearer header for the selector
        st.markdown(
            "<div class='cand-select-header'><strong>Select a resume</strong></div>",
            unsafe_allow_html=True
        )
        sel = st.selectbox("Select a resume", options=options, index=0, key="cand_selector", label_visibility="collapsed")
        sel_idx = options.index(sel) if sel in options else 0
        cand = cands[sel_idx]

        # CSS tweaks to improve look & spacing and align heights
        st.markdown(
            """
            <style>
            .cand-select-header { font-size: 1.05rem; font-weight: 700; margin: 0 0 0.25rem 0; }
            .readonly-input, .readonly-textarea {
                width: 100%;
                box-sizing: border-box;
                border: 1px solid rgba(49,51,63,0.2);
                border-radius: 0.5rem;
                padding: 0.5rem 0.75rem;
                background: var(--background-color);
                color: inherit;
                font: inherit;
            }
            /* Match Streamlit text_input height more closely */
            .readonly-input { height: 2.5rem; line-height: 1.2; }
            .readonly-textarea { height: 320px; resize: vertical; }
            .resume-section-gap { height: 0.75rem; }
            .field-label { font-weight: 700; margin-bottom: 0.25rem; display:block; }
            </style>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown("<span class='field-label'>Filename</span>", unsafe_allow_html=True)
            st.markdown(
                f"<input class='readonly-input' type='text' value='{html.escape(cand.file_name)}' readonly/>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown("<span class='field-label'>Label (Edit if desired)</span>", unsafe_allow_html=True)
            lbl_key = f"cand_label_{cand.hash[:12]}"
            # Collapse the label on the Streamlit input (we show our own bold heading above)
            new_label = st.text_input("Label (Edit if desired)", value=cand.name, key=lbl_key, label_visibility="collapsed")
            if new_label != cand.name:
                cand.name = new_label

        # Add a little extra gap before the resume text
        st.markdown("<div class='resume-section-gap'></div>", unsafe_allow_html=True)
        st.markdown("<span class='field-label'>Resume Text Extracted</span>", unsafe_allow_html=True)
        st.code(cand.text)

        # Persist any label edits back to session_state
        st.session_state.cached_candidates = cands

elif current_page == "Scoring Guide":
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("📊 Scoring Guide")
    st.markdown("### What do the scores mean?")
    st.markdown("""
Each score is a **semantic similarity** between a JD **criterion** and the **most relevant snippet** in a candidate's resume.
It ranges from **0.00** (not similar) to **1.00** (nearly identical in meaning). It is **not** a percentage of criteria met.

**Color Coding:**
- ✅ **Green (Strong):** ≥0.70 — Excellent to good alignment
- ⚠️ **Yellow (Moderate):** 0.45–0.70 — Partial or indirect match
- ⛔ **Red (Weak):** <0.45 — Likely not covered

**Quick ranges:**
- **0.75–1.00:** Excellent alignment
- **0.60–0.74:** Good
- **0.45–0.59:** Moderate (partial/indirect)
- **0.30–0.44:** Weak
- **<0.30:** Likely not covered

**Overall** is a weighted average of all criteria scores.
""")

    st.markdown("### How it’s calculated (plain English)")
    st.markdown("""
1) The resume is split into chunks (e.g., ~1,200 chars).
2) For each **criterion**, we compare it to every chunk and keep the **best match**.
3) The similarity comes from vector math (cosine similarity on embeddings/TF‑IDF), which captures meaning overlap, not just the same words.
""")

    st.markdown("### Examples")
    st.markdown("""
- **Criterion:** “Project management with Agile/Scrum.”  
  **Resume:** “Led a 10‑person team delivering sprints using Scrum ceremonies.”  
  → **Score ≈ 0.75–0.85** (strong, direct alignment)

- **Criterion:** “Python data analysis.”  
  **Resume:** “Automated Excel reports; some VBA macros.”  
  → **Score ≈ 0.35–0.50** (partial/weak — tooling mismatch)

- **Criterion:** “Stakeholder communication.”  
  **Resume:** “Presented monthly performance updates to executives and coordinated cross‑team work.”  
  → **Score ≈ 0.60–0.75** (good evidence)
""")

    st.markdown("### How to use the scores")
    st.markdown("""
- Treat scores as **evidence strength**, not pass/fail. Use the Coverage matrix to spot strengths and gaps.
- Adjust **weights** to emphasise must‑haves; Overall will reflect your priorities.
- Use the **Evidence Explorer** (Coverage tab) to view the exact resume snippet behind each score.
""")

elif current_page == "Export":
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("Export")
    covdf = st.session_state.get("last_coverage", pd.DataFrame())
    if not isinstance(covdf, pd.DataFrame) or covdf.empty:
        st.info("Run analysis to enable exports.")
    else:
        jd_text = st.session_state.jd.text if st.session_state.get("jd") else ""
        md_text = to_markdown_report(covdf, st.session_state.get("last_insights", {}), jd_text)
        st.markdown("**Preview (Markdown)**")
        st.code(md_text, language="markdown")
        st.download_button("⬇️ Download report (Markdown)", data=md_text.encode("utf-8"), file_name="candidate_report.md")
        if reportlab is not None:
            if st.button("Generate PDF preview"):
                pdf_bytes = to_pdf_bytes_from_markdown(md_text)
                if pdf_bytes:
                    st.download_button("⬇️ Download PDF report", data=pdf_bytes, file_name="candidate_report.pdf", mime="application/pdf")
                else:
                    st.error("Failed to generate PDF.")
        else:
            st.caption("Install 'reportlab' for simple PDF export, or print the Markdown to PDF in your editor/browser.")

elif current_page == "PDF Tools":
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("PDF Tools — Convert Scanned PDFs to Text")

    st.info("""
    ### 🧩 Developer Notes — PDF Tools (OCR)
    ⚠️ **Current status:** The PDF Tools feature is **not fully working yet.**
    
    **Observed behavior:** OCR always returns 'produced no text', even for simple image PDFs.  
    This happens because `pdf2image` depends on Poppler binaries, which are not installed or accessible
    in the current Windows environment.

    **Next steps (when resuming work):**
    - Confirm Poppler and Tesseract are installed and available in PATH.
    - If not, modify extraction to use **PyMuPDF rasterization only** (no Poppler dependency).
    - When deploying, ensure the server container includes Tesseract and optionally Poppler, or switch to a cloud OCR API (Google Vision / AWS Textract).
    - Add better runtime diagnostics to show which backend (pdf2image vs PyMuPDF) was used and if OCR produced text.

    **Design intention:** This page will eventually allow admins or users to upload scanned PDFs
    and get readable text + a simple text-only PDF output.

    **Reminder:** When Nigel says “Let's finish the PDF Tools page based on the notes on that page,”
    resume work from these notes.
    """)

    # Diagnostics
    tesseract_ok = False
    tess_version = ""
    try:
        if pytesseract is not None:
            tess_version = str(pytesseract.get_tesseract_version())
            tesseract_ok = True
    except Exception:
        tesseract_ok = False
    fitz_ok = (fitz is not None)
    p2i_ok = (pdf2image is not None)
    if not tesseract_ok:
        st.warning("Tesseract OCR not detected. Install Tesseract and ensure it's on PATH.", icon="⚠️")
    else:
        st.caption(f"Tesseract detected: {tess_version}")
    st.caption(f"Backends — PyMuPDF: {'✅' if fitz_ok else '❌'} • pdf2image: {'✅' if p2i_ok else '❌'}")
    st.caption("Use this tool when a PDF is image-based or poorly extracted. It will OCR the file and give you text. Optionally, it can generate a simple text-only PDF (not layout-preserving).")

    ocr_ok = (pdf2image is not None) and (pytesseract is not None)
    if not ocr_ok:
        st.warning("OCR dependencies not available. Install 'pdf2image' and 'pytesseract' (and Tesseract engine) to enable this.", icon="⚠️")

    strong_ocr = st.checkbox("Use stronger OCR (for faint or blurry scans)", value=False, help="Renders at higher DPI and stronger thresholding; slower but recovers faint text.")

    conv_files = st.file_uploader("Upload one or more PDFs to OCR", type=["pdf"], accept_multiple_files=True, key="pdf_tools_upl")

    if conv_files and ocr_ok:
        for uf in conv_files:
            st.markdown(f"**{uf.name}**")
            b = uf.read()
            # Reuse robust OCR text extractor (uses PyMuPDF first; falls back to OCR)
            text_ocr = extract_text_from_pdf_robust(b, strong=strong_ocr)
            if not text_ocr:
                text_ocr = extract_text_from_pdf_robust(b, strong=True, prefer_pdf2image=False)
            if not text_ocr:
                st.error("OCR produced no text.")
                continue
            # Download as .txt
            st.download_button(
                "⬇️ Download TXT (OCR)",
                data=text_ocr.encode("utf-8", errors="ignore"),
                file_name=uf.name.rsplit(".", 1)[0] + "_ocr.txt",
                mime="text/plain"
            )
            # Simple text-only PDF (layout not preserved)
            if reportlab is not None:
                pdf_bytes = to_pdf_bytes_from_markdown(text_ocr)
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Download simple PDF (text-only)",
                        data=pdf_bytes,
                        file_name=uf.name.rsplit(".", 1)[0] + "_ocr.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.info("Reportlab PDF generation unavailable.")
            else:
                st.caption("Install 'reportlab' if you want a simple text-only PDF export.")

elif current_page == "Settings":
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("Settings")

    # Scoring Thresholds Section
    st.markdown("### 📊 Scoring Thresholds")
    st.caption("These settings control how scores are displayed throughout the app (Coverage, Compare, etc.)")
    
    st.markdown("**Threshold Values:**")
    thresh_col1, thresh_col2 = st.columns(2)
    
    with thresh_col1:
        hi = st.slider(
            "✅ High threshold (green)", 
            0.5, 0.95, 
            st.session_state.get("cov_hi", 0.70), 
            step=0.01,
            help="Scores at or above this are marked with ✅ (green). Recommended: 0.65-0.75"
        )
        st.session_state.cov_hi = hi
    
    with thresh_col2:
        lo = st.slider(
            "⚠️ Medium threshold (yellow)", 
            0.1, float(hi-0.01), 
            st.session_state.get("cov_lo", 0.45), 
            step=0.01,
            help="Scores at or above this are marked with ⚠️ (yellow). Below this get ⛔ (red). Recommended: 0.40-0.50"
        )
        st.session_state.cov_lo = lo
    
    # Display current thresholds and reset button
    info_col, reset_col = st.columns([3, 1])
    with info_col:
        st.info(f"Current thresholds: ✅ High ≥ {hi:.2f} | ⚠️ Medium ≥ {lo:.2f} | ⛔ Low < {lo:.2f}")
    with reset_col:
        if st.button("Reset to Defaults", help="Reset thresholds to recommended defaults (High: 0.70, Medium: 0.45)"):
            st.session_state.cov_hi = 0.70
            st.session_state.cov_lo = 0.45
            st.rerun()
    
    st.markdown("---")

    # Layout-aware PDF extraction toggle (default ON)
    st.session_state["use_layout_pdf"] = st.checkbox(
        "Use layout-aware PDF extraction",
        value=st.session_state.get("use_layout_pdf", True),
        help="Disable if PDFs extract with jumbled order; falls back to standard extractor."
    )
# Insight Engine selection (default gpt-4o)
    # Insight Engine selection (GPT-only with descriptions)
    st.markdown("### GPT Model")
    st.markdown("This app uses OpenAI’s `gpt-4o` model for all GPT-powered analysis and insights.")

    with st.popover("🔒 Privacy & data handling"):
        st.markdown(
            "> **When using OpenAI/GPT**, data from Job Descriptions and Candidate Resumes is **securely transmitted** to OpenAI’s servers via their API."
        )
        st.markdown(
            "- ✅ **API data is *not* used to train OpenAI models** (unless you explicitly opt in). We have **not opted in**.\n"
            "- ✅ **API data is retained up to 30 days** for abuse monitoring, then deleted automatically.\n"
            "- ❗ If this is unacceptable, do not use this program."
        )

    st.markdown("---")
    st.markdown("### Weights")
    st.session_state["weights_mode"] = st.radio("Weights", ["Uniform","Custom"],
        index=(0 if st.session_state.get("weights_mode","Uniform") == "Uniform" else 1),
        help="Use uniform weights or specify custom weights per criterion.")
    if st.session_state["weights_mode"] == "Custom":
        st.caption("Provide either 'criterion, weight' per line or one weight per line (aligned with criteria order).")
        st.session_state["weights_csv"] = st.text_area("Custom weights",
            value=st.session_state.get("weights_csv","Python, 2\nData analysis, 1\nStakeholder communication, 1"),
            height=120)

    st.markdown("---")
    st.markdown("### Performance")
    st.session_state["chunk_chars"] = st.slider("Chunk size (characters)", 600, 2000, st.session_state.get("chunk_chars",1200), step=100)
    st.session_state["overlap"] = st.slider("Chunk overlap", 50, 400, st.session_state.get("overlap",150), step=25)

# -----------------------------
# Orchestration
# -----------------------------
if st.session_state.get("_trigger_analyse", False):
    # Get status container if it exists
    status_container = st.session_state.get("_status_container", st.empty())
    
    crit_text = st.session_state.get("criteria_text", "").strip()
    if not crit_text and st.session_state.get("jd") is not None:
        # default build if nothing present: GPT-only
        jd_txt = st.session_state.jd.text
        if _cached_openai_client()[0] is None:
            st.error("GPT extraction requires an OpenAI API key. Set OPENAI_API_KEY or paste criteria manually on the JD tab.")
        else:
            secs = cached_extract_jd_sections_with_gpt(st.session_state.jd.hash, jd_txt)
            crits, cat_map = build_criteria_from_gpt_sections(secs, per_section=999, cap_total=10000)
            st.session_state.criteria_text = "\n".join(crits)
            st.session_state.cat_map = cat_map
            crit_text = st.session_state.criteria_text

    

    # Warn if JD text is suspiciously short — extraction may have failed
    try:
        jd_len = len(st.session_state.jd.text or "")
        if jd_len < 200:
            st.warning("JD text looks very short. Try enabling the robust PDF extractor or use 'Extract with GPT' to confirm sections.", icon="⚠️")
    except Exception:
        pass

    criteria = parse_criteria_text(crit_text)
    weights = get_weights(criteria, st.session_state.get("weights_mode","Uniform"), st.session_state.get("weights_csv",""))

    # --- Auto-build criteria if still empty (fallback on Analyze) ---
    if (not criteria) and st.session_state.get("jd"):
        jd_txt_fb = st.session_state.jd.text
        if _cached_openai_client()[0] is not None:
            secs_fb = cached_extract_jd_sections_with_gpt(st.session_state.jd.hash, jd_txt_fb)
            crits_fb, cat_map_fb = build_criteria_from_gpt_sections(secs_fb, per_section=999, cap_total=10000)
        else:
            st.error("GPT extraction requires an OpenAI API key. Set OPENAI_API_KEY or paste criteria manually on the JD tab.")
            crits_fb, cat_map_fb = [], {}
        if crits_fb:
            st.session_state.criteria_text = "\n".join(crits_fb)
            st.session_state.cat_map = cat_map_fb
            criteria = parse_criteria_text(st.session_state.criteria_text)
            try:
                st.toast(f"Built {len(criteria)} criteria automatically.", icon="🧩")
            except Exception:
                pass

    if not criteria:
        st.warning("No criteria to analyse. Build criteria (Job Description tab) then click Analyse.")
    elif not st.session_state.get("cached_candidates"):
        st.warning("Please upload candidate resumes before running Analyze.")
    else:
        status_container.info("🔍 Extracting JD & building criteria...")
        
        with st.spinner("Running analysis…"):
            cov, ins_local, snips, ev_map = analyse_candidates(
                st.session_state.cached_candidates, criteria, weights,
                chunk_chars=st.session_state.get("chunk_chars",1200),
                overlap=st.session_state.get("overlap",150)
            )
            
            status_container.info("📊 Scoring candidates complete. Preparing insights...")
            
            st.session_state.last_coverage = cov
            st.session_state.last_insights = ins_local
            st.session_state.last_snippets = snips
            st.session_state.evidence_map = ev_map

                        # ---- GPT insight generation ----
            api_ok = _cached_openai_client()[0] is not None

            if api_ok and isinstance(cov, pd.DataFrame) and not cov.empty:
                status_container.info("✨ Generating GPT insights...")
                try:
                    with st.spinner("Generating GPT insights…"):
                        jd_text_full = st.session_state.jd.text if st.session_state.get("jd") else ""
                        upgraded = {}
                        crit_cols = [c for c in cov.columns if c not in ("Candidate","Overall")]
                        for _, row in cov.iterrows():
                            cand_name = row["Candidate"]
                            # Row dict: {criterion: score}
                            row_dict = {c: float(row[c]) for c in crit_cols}
                            # Retrieve cached candidate text
                            cand_text = ""
                            for c_obj in st.session_state.get("cached_candidates", []):
                                if c_obj.name == cand_name:
                                    cand_text = c_obj.text
                                    break
                            upgraded[cand_name] = gpt_candidate_insights(
                                candidate_name=cand_name,
                                candidate_text=cand_text,
                                jd_text=jd_text_full,
                                criteria=crit_cols,
                                coverage_row=row_dict,
                                evidence_map=st.session_state.get("evidence_map", {}),
                            ) or st.session_state["last_insights"].get(cand_name, {})
                        # Replace insights with GPT versions where available
                        st.session_state["last_insights"] = upgraded
                        status_container.success("✅ Analysis complete!")
                        try: st.toast("Insights enhanced with GPT.", icon="✨")
                        except Exception: pass
                except Exception:
                    st.warning("GPT insight generation failed — using local insights.")
                    status_container.success("✅ Analysis complete (without GPT insights)")
            elif not api_ok:
                st.info("OpenAI API key not configured. Keeping baseline (non-GPT) insights.")
                status_container.success("✅ Analysis complete!")
            else:
                status_container.success("✅ Analysis complete!")
                
            st.session_state._trigger_analyse = False
            st.rerun()

        if isinstance(st.session_state.last_coverage, pd.DataFrame) and not st.session_state.last_coverage.empty:
            try: st.toast("Analysis complete.", icon="✅")
            except Exception: pass
        else:
            st.warning("Analysis ran but produced no rows. Check criteria and document text.")

# -----------------------------
# Insights-only refresh orchestration (no rescoring)
# -----------------------------
if st.session_state.get("_trigger_refresh_insights", False):
    cov = st.session_state.get("last_coverage", pd.DataFrame())
    if not isinstance(cov, pd.DataFrame) or cov.empty:
        st.warning("No coverage table available. Run full analysis first.")
        st.session_state._trigger_refresh_insights = False
    else:
        api_ok = _cached_openai_client()[0] is not None
        if not api_ok:
            st.info("OpenAI API not configured. Set OPENAI_API_KEY to refresh GPT insights.")
            st.session_state._trigger_refresh_insights = False
        else:
            _update_stage("✨ Regenerating GPT insights…")
            try:
                with st.spinner("Regenerating GPT insights…"):
                    jd_text_full = st.session_state.jd.text if st.session_state.get("jd") else ""
                    upgraded = {}
                    crit_cols = [c for c in cov.columns if c not in ("Candidate","Overall")]
                    for _, row in cov.iterrows():
                        cand_name = row["Candidate"]
                        row_dict = {c: float(row[c]) for c in crit_cols}
                        cand_text = ""
                        for c_obj in st.session_state.get("cached_candidates", []):
                            if c_obj.name == cand_name:
                                cand_text = c_obj.text
                                break
                        upgraded[cand_name] = gpt_candidate_insights(
                            f"{cand_name}",
                            cand_text,
                            jd_text_full,
                            crit_cols,
                            row_dict,
                            st.session_state.get("evidence_map", {}),
                        ) or st.session_state.get("last_insights", {}).get(cand_name, {})
                    st.session_state["last_insights"] = upgraded
                    st.success("✅ GPT insights refreshed.")
            finally:
                st.session_state._trigger_refresh_insights = False
            st.rerun()