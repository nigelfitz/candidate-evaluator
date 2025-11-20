# Candidate Pack Summariser — pt4 (JD extraction quality restoration merged)
from __future__ import annotations

import io, os, re, json, hashlib, textwrap, sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

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

def call_llm_json(system_prompt: str, user_prompt: str, schema: Dict[str, Any], model: str = "gpt-4o-mini") -> Dict[str, Any]:
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
# Legacy JD parser (kept)
# -----------------------------
LEGACY_HEADINGS = [
    "requirements","selection criteria","criteria",
    "key skills","technical skills","skills",
    "responsibilities","duties",
    "experience","qualifications",
    "about you","nice to have","preferred",
    "role","role responsibilities",
]

def _is_heading_legacy(line: str) -> Optional[str]:
    s = normalize_ws(line)
    if not s:
        return None
    s_l = s.lower().rstrip(":")
    for h in LEGACY_HEADINGS:
        if s_l == h or s_l.startswith(h):
            return h
    if s.endswith(":") and len(s) <= 48:
        return s_l
    return None

def parse_jd_legacy(jd_text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    cur = "uncategorised"
    for raw in (jd_text or "").splitlines():
        h = _is_heading_legacy(raw)
        if h:
            cur = h.replace(" ", "_")
            sections.setdefault(cur, [])
            continue
        m = re.match(r"^(\s*[-•·▪◦*]|\s*\d+[\.\)])\s+(.*)$", raw)
        if m:
            bullet = normalize_ws(m.group(2))
            if bullet:
                sections.setdefault(cur, []).append(bullet)
    if not any(sections.values()):
        bullets = []
        for l in (jd_text or "").splitlines():
            s = normalize_ws(l)
            if re.search(r"(experience|responsibil|require|skills|criteria|competenc)", s.lower()) and len(s) > 25:
                bullets.append(s)
        if bullets:
            sections["requirements"] = bullets
        else:
            sections["uncategorised"] = []
    return sections

def clean_for_display_legacy(b: str) -> str:
    s = b.strip("•-*–—· ").strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s

def build_criteria_legacy(sections: Dict[str, List[str]], per_section: int = 6, cap_total: int = 30) -> Tuple[List[str], Dict[str,str]]:
    order = [
        "requirements","selection_criteria","criteria",
        "key_skills","technical_skills","skills",
        "responsibilities","duties","experience","qualifications",
        "about_you","nice_to_have","preferred","role","role_responsibilities","uncategorised"
    ]
    crits: List[str] = []
    cat_map: Dict[str,str] = {}
    for sec in order + [k for k in sections.keys() if k not in order]:
        bullets = sections.get(sec, [])
        if not bullets:
            continue
        picked = [clean_for_display_legacy(x) for x in bullets[:per_section]]
        for p in picked:
            if p and p not in crits:
                crits.append(p)
                cat_map[p] = sec
            if len(crits) >= cap_total:
                break
        if len(crits) >= cap_total:
            break
    return crits[:cap_total], cat_map

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

@st.cache_data(show_spinner=False)
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

def analyze_candidates(
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
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", s)
    s = re.sub(r"\b(?:\+?\d[\d \-\(\)]{8,}\d)\b", "[redacted-phone]", s)
    s = re.sub(r"https?://\S+|www\.\S+", "[redacted-url]", s)
    return s

def _model_for_engine(engine: str) -> Optional[str]:
    if engine == "gpt-4o":
        return "gpt-4o"
    if engine == "gpt-4o-mini":
        return "gpt-4o-mini"
    return None

def gpt_candidate_insights(
    engine: str,
    candidate_name: str,
    candidate_text: str,
    jd_text: str,
    criteria: List[str],
    coverage_row: Dict[str, float],
    evidence_map: Dict[Tuple[str,str], Tuple[str,float]],
    redacted: bool = False
) -> Dict[str, Any]:
    """
    Returns dict with keys: {'top': [..], 'gaps': [..], 'notes': str}
    Uses JSON schema to keep output consistent.
    """
    model = _model_for_engine(engine)
    if model is None:
        return {}

    # Build compact evidence pack: top 10 criteria by score (descending)
    crit_scores = [(c, float(coverage_row.get(c, 0.0))) for c in criteria]
    crit_scores.sort(key=lambda x: -x[1])
    top_items = crit_scores[:10]

    ev_lines = []
    for c, sc in top_items:
        snip, _ = evidence_map.get((candidate_name, c), ("", sc))
        snip = snip or ""
        if redacted:
            snip = _redact(snip)
        _snip_clean = snip[:400].replace("\n", " ")
        ev_lines.append(f"- {c} (score {sc:.2f}): {_snip_clean}")

    # Possibly redact candidate/JD text if requested
    cand_body = _redact(candidate_text) if redacted else candidate_text
    jd_body   = _redact(jd_text) if redacted else jd_text

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
        "- Provide a 1–2 sentence **Notes** summary (e.g., risk/fit, seniority, missing essentials).\n"
        "Return ONLY JSON matching the schema."
    )

    try:
        data = call_llm_json(system, user, schema, model=model)
        # light normalization
        return {
            "top": [normalize_ws(x) for x in data.get("top", []) if normalize_ws(x)],
            "gaps": [normalize_ws(x) for x in data.get("gaps", []) if normalize_ws(x)],
            "notes": normalize_ws(data.get("notes","")),
        }
    except Exception:
        return {}

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Job Candidates Summariser", layout="wide")
st.title("📦 Job Candidates Summariser")
# Top-of-page status/progress container
_top_container = st.container()
# Stage label placeholder (sequential updates)
_stage_ph = _top_container.empty()

def _update_stage(msg: str):
    try:
        _stage_ph.info(msg)
    except Exception:
        pass


with st.sidebar:
    st.markdown("<h2 style='font-size:1.5em; font-weight:700; margin-bottom:0.4em;'>📂 Uploads</h2>", unsafe_allow_html=True)

    st.markdown("<h4 style='margin-top:0.8em; font-weight:600;'>Job Description</h4>", unsafe_allow_html=True)
    jd_file = st.file_uploader("Upload JD", type=["pdf","docx","txt"], key="jd_upl", accept_multiple_files=False, label_visibility="collapsed")

    with st.expander("Or paste JD text manually"):
        jd_manual = st.text_area("JD text", value=st.session_state.get("jd_manual_text",""), height=180)
        st.session_state.jd_manual_text = jd_manual

    # Divider between JD and Candidate sections
    st.markdown("<div style='background-color:#d0d0d0; height:2px; border-radius:2px; margin:10px 0;'></div>", unsafe_allow_html=True)

    st.markdown("<h4 style='margin-top:1.2em; font-weight:600;'>Candidate Resumes</h4>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload candidate resumes", accept_multiple_files=True, type=["pdf","docx","txt"], key="cand_upl", label_visibility="collapsed")

    # Build/clear/analyze controls
    # Divider above Analyze / Update button
    st.markdown("<div style='background-color:#d0d0d0; height:2px; border-radius:2px; margin:10px 0;'></div>", unsafe_allow_html=True)
    colb1, colb2, colb3 = st.columns(3)
    if colb1.button("Analyze / Update", type="primary"):
        st.session_state._trigger_analyze = True
        st.rerun()

    # JD assignment from file or manual text
    if jd_file is not None:
        b = jd_file.read()
        if jd_file.name.lower().endswith('.pdf'):
            try:
                jd_text = extract_text_layout_aware(b, jd_file.name)
                if not jd_text.strip():
                    jd_text = read_file_bytes(b, jd_file.name)
            except Exception:
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
        b = uf.read(); text = read_file_bytes(b, uf.name)
        name_guess = infer_candidate_name(uf.name, text)
        new_candidates.append(Candidate(name=name_guess, file_name=uf.name, text=text, hash=hash_bytes(b)))
if new_candidates:
    existing = {c.hash: c for c in st.session_state.get("cached_candidates", [])}
    for c in new_candidates: existing[c.hash] = c
    st.session_state.cached_candidates = list(existing.values())

# Tabs
overview_tab, jd_tab, coverage_tab, insights_tab, compare_tab, labels_tab, scoring_tab, export_tab, pdf_tab, settings_tab = st.tabs(["Overview","Job Description","Coverage","Insights","Compare","Candidate Labels","Scoring Explained","Export","PDF Tools","Settings"])



with overview_tab:
    st.subheader("Overview")
    left, right = st.columns([2, 1])

    with left:
        st.metric("Candidates loaded", len(st.session_state.get("cached_candidates", [])))
        covdf = st.session_state.get("last_coverage", pd.DataFrame())
        st.metric("Criteria", len([c for c in covdf.columns if c not in ('Candidate','Overall')]) if isinstance(covdf, pd.DataFrame) and not covdf.empty else 0)
        if isinstance(covdf, pd.DataFrame) and not covdf.empty:

            st.markdown("#### Top overall scores")
            st.caption("Scoring explained — see the **Scoring Explained** tab.")
            try:
                top3 = covdf[["Candidate","Overall"]].head(3)
                for _, r in top3.iterrows():
                    st.write(f"- **{r['Candidate']}** — {r['Overall']:.2f}")
            except Exception:
                pass

    with right:
        st.empty()

with jd_tab:
    st.subheader("Job Description")
    if st.session_state.get("jd"):
        jd_obj = st.session_state.jd
        st.write(f"**{jd_obj.file_name}** — {len(jd_obj.text):,} characters")
        with st.expander("Preview JD text"):
            st.text_area("JD content", value=jd_obj.text, height=300)
    else:
        st.warning("Upload a JD file or paste JD text in the sidebar.")

    # ---------- JD extraction mode selector ----------
    st.markdown("### Build criteria from JD")
    mode = st.radio("Extraction mode", ["GPT (structured sections)", "Legacy (heuristic)"], index=0,
                    help="GPT mode restores the old structured extraction into key sections.")
    colj1, colj2 = st.columns(2)
    per_sec = colj1.number_input("Max per section", 1, 20, value=6)
    cap_total = colj2.number_input("Total cap", 5, 80, value=30)

    if mode.startswith("GPT"):
        if _cached_openai_client()[0] is None:
            st.warning("OpenAI API not configured or SDK not installed. Set OPENAI_API_KEY and ensure 'openai' package is available.", icon="⚠️")
        if st.button("Extract with GPT"):
            if not st.session_state.get("jd"):
                st.warning("No JD loaded.")
            else:
                secs = cached_extract_jd_sections_with_gpt(st.session_state.jd.hash, st.session_state.jd.text)
                # show sections
                with st.expander("Extracted sections (GPT)"):
                    st.markdown("**Key skills**")
                    for x in secs.key_skills: st.markdown(f"- {x}")
                    st.markdown("**Responsibilities**")
                    for x in secs.responsibilities: st.markdown(f"- {x}")
                    st.markdown("**Qualifications**")
                    for x in secs.qualifications: st.markdown(f"- {x}")
                    st.markdown("**Experience required**")
                    for x in secs.experience_required: st.markdown(f"- {x}")
                crits, cat_map = build_criteria_from_gpt_sections(secs, per_section=int(per_sec), cap_total=int(cap_total))
                st.session_state.criteria_text = "\n".join(crits)
                st.session_state.cat_map = cat_map
                st.success(f"Built {len(crits)} criteria from GPT sections. Go to Coverage and click Analyze.")
    else:
        # Legacy UI (fixed f-string quoting bug)
        if st.session_state.get("jd"):
            secs = parse_jd_legacy(st.session_state.jd.text)
            picks: Dict[str, List[str]] = {}
            preferred_order = [
                "requirements","selection_criteria","criteria",
                "key_skills","technical_skills","skills",
                "responsibilities","duties","experience","qualifications",
                "about_you","nice_to_have","preferred","role","role_responsibilities","uncategorised"
            ]
            for sec in [s for s in preferred_order if s in secs] + [k for k in secs.keys() if k not in preferred_order]:
                bullets = secs.get(sec, [])
                with st.expander(f"{sec.replace('_',' ').title()} ({len(bullets)})"):
                    opts = [clean_for_display_legacy(b) for b in bullets]
                    sel = st.multiselect("Select items", options=opts, default=opts[:min(6, len(opts))], key=f"legacy_{sec}")
                    picks[sec] = sel

            if st.button("Build Criteria (Legacy)"):
                any_sel = any(picks.get(sec) for sec in picks)
                if any_sel:
                    crits, cat_map = [], {}
                    for sec, lst in picks.items():
                        for x in lst[:int(per_sec)]:
                            if x and x not in crits:
                                crits.append(x); cat_map[x] = sec
                            if len(crits) >= int(cap_total): break
                        if len(crits) >= int(cap_total): break
                else:
                    crits, cat_map = build_criteria_legacy(secs, per_section=int(per_sec), cap_total=int(cap_total))
                st.session_state.criteria_text = "\n".join(crits)
                st.session_state.cat_map = cat_map
                st.success(f"Built {len(crits)} criteria. Go to Coverage and click Analyze.")

with coverage_tab:
    st.subheader("Coverage Matrix")
    coverage_df = st.session_state.get("last_coverage", pd.DataFrame())
    if not isinstance(coverage_df, pd.DataFrame) or coverage_df.empty:
        st.warning("Upload JD/criteria and candidate resumes, then click Analyze.")
    else:
        cc1, cc2, cc3 = st.columns([1,1,1])
        show_scores = cc1.checkbox("Show numeric scores", value=st.session_state.get("cov_show_scores", True))
        hi = cc2.slider("High threshold (✅)", 0.5, 0.95, st.session_state.get("cov_hi", 0.70), step=0.01)
        lo = cc3.slider("Medium threshold (⚠️)", 0.1, float(hi-0.01), st.session_state.get("cov_lo", 0.45), step=0.01)
        st.session_state.cov_show_scores, st.session_state.cov_hi, st.session_state.cov_lo = show_scores, hi, lo

        cat_map = st.session_state.get("cat_map", {})
        candidates = coverage_df["Candidate"].tolist()
        crit_cols = [c for c in coverage_df.columns if c not in ("Candidate","Overall")]
        mat = coverage_df.set_index("Candidate")[crit_cols].T
        ordered = sorted(crit_cols, key=lambda c: (cat_map.get(c, "zz"), c.lower()))
        disp = pd.DataFrame(index=pd.MultiIndex.from_tuples([(cat_map.get(c,"uncategorised"), c) for c in ordered], names=["Category","Criterion"]), columns=candidates)
        for c in candidates:
            for crit in ordered:
                val = float(mat.loc[crit, c])
                icon = "✅" if val >= hi else ("⚠️" if val >= lo else "⛔")
                disp.loc[(cat_map.get(crit,"uncategorised"), crit), c] = f"{icon} {val:.2f}" if show_scores else icon
        st.caption("Criteria down the left • Candidates across the top • Icons reflect thresholds")
        st.dataframe(disp, width="stretch")

        st.markdown("---")
        st.markdown("### Evidence Explorer")
        names = coverage_df["Candidate"].tolist()
        ordered = ordered  # ensure exists
        crit = st.selectbox("Criterion", ordered, index=0 if ordered else None)
        cand_choice = st.selectbox("Candidate", names, index=0 if names else None)
        if crit and cand_choice:
            ev_map = st.session_state.get("evidence_map", {})
            snip, sc = ev_map.get((cand_choice, crit), (None, None))
            if snip is None:
                st.info("No cached evidence yet for this pair. Re-run Analyze to refresh evidence.")
            else:
                st.write(f"**Score:** {sc:.2f}")
                st.code(snip)

with insights_tab:
    _engine_label = st.session_state.get("insight_engine", "gpt-4o-mini")
    st.subheader(f"Candidate Insights ({_engine_label})")
    col_ins1, col_ins2 = st.columns([1,1])
    if col_ins1.button("Refresh GPT insights"):
        st.session_state._trigger_refresh_insights = True
        st.rerun()
    insights_map = st.session_state.get("last_insights", {})
    covdf = st.session_state.get("last_coverage", pd.DataFrame())
    if not insights_map or (isinstance(covdf, pd.DataFrame) and covdf.empty):
        st.info("Run analysis first.")
    else:
        order_names = covdf["Candidate"].tolist()
        for name in order_names:
            info = insights_map.get(name, {})
            with st.expander(f"🔎 {name}"):
                st.markdown("**Top strengths**")
                for t in info.get("top", []): st.write("- ", t)
                st.markdown("**Gaps / risks**")
                for g in info.get("gaps", []): st.write("- ", g)
                st.caption(info.get("notes", ""))

with compare_tab:
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

with labels_tab:
    st.subheader("Candidate Labels")
    st.caption("edit label names if desired")
    if st.session_state.get("cached_candidates"):
        for i, c in enumerate(st.session_state.cached_candidates):
            c.name = st.text_input(f"Label for {c.file_name}", value=c.name, key=f"nm_{i}_{c.hash[:12]}")
    else:
        st.info("Upload candidate resumes to edit labels.")

with scoring_tab:
    st.subheader("Scoring Explained")
    st.markdown("### What do the scores mean?")
    st.markdown("""
Each score is a **semantic similarity** between a JD **criterion** and the **most relevant snippet** in a candidate's resume.
It ranges from **0** (not similar) to **1** (nearly identical in meaning). It is **not** a percentage of criteria met.

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


with export_tab:
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


with pdf_tab:
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
with settings_tab:
    st.subheader("Settings")

    # Insight Engine selection (default gpt-4o-mini)
    st.markdown("### Insight Engine")
    st.session_state["insight_engine"] = st.radio(
        "Generate candidate insights using:",
        ["Local", "gpt-4o-mini", "gpt-4o"],
        index=(["Local","gpt-4o-mini","gpt-4o"].index(st.session_state.get("insight_engine","gpt-4o-mini")) if st.session_state.get("insight_engine") in ["Local","gpt-4o-mini","gpt-4o"] else 1),
        help="Local = offline heuristics. GPT options call OpenAI to produce higher-quality insights."
    )

    # Redacted mode (unchecked by default)
    st.session_state["redacted_mode"] = st.checkbox(
        "Redacted mode (redact emails/phones/URLs before sending to GPT)",
        value=st.session_state.get("redacted_mode", False)
    )

    # API key status
    api_set = bool(os.getenv("OPENAI_API_KEY"))
    st.caption(f"OpenAI API key detected: {'✅' if api_set else '❌'} (.env)")

    # Info popovers (stacked vertically)
    with st.popover("ℹ️ What do these engines mean?"):
        st.markdown(
            "- ✅ **Local**: fully offline; fastest; good ranking; **weaker narrative insights**.\n"
            "- ✅ **gpt-4o-mini**: fast, cost-efficient; **much better** plain-English insights.\n"
            "- ✅ **gpt-4o**: highest-quality phrasing and reasoning; **slightly slower**."
        )

    with st.popover("🔒 Privacy & data handling"):
        st.markdown(
            "> **When using GPT-4o or GPT-4o-mini**, data from Job Descriptions and Candidate Resumes is **securely transmitted** to OpenAI’s servers via their API."
        )
        st.markdown(
            "- ✅ **API data is *not* used to train OpenAI models** (unless you explicitly opt in). We have **not opted in**.\n"
            "- ✅ **API data is retained up to 30 days** for abuse monitoring, then deleted automatically.\n"
            "- ❗ If this is unacceptable, use **Local** mode or enable **Redacted mode**."
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
if st.session_state.get("_trigger_analyze", False):
    crit_text = st.session_state.get("criteria_text", "").strip()
    if not crit_text and st.session_state.get("jd") is not None:
        # default build if nothing present: prefer GPT if available, else legacy
        jd_txt = st.session_state.jd.text
        if _cached_openai_client()[0] is not None:
            secs = cached_extract_jd_sections_with_gpt(st.session_state.jd.hash, jd_txt)
            crits, cat_map = build_criteria_from_gpt_sections(secs, per_section=6, cap_total=30)
        else:
            secs_legacy = parse_jd_legacy(jd_txt)
            crits, cat_map = build_criteria_legacy(secs_legacy, per_section=6, cap_total=30)
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
            crits_fb, cat_map_fb = build_criteria_from_gpt_sections(secs_fb, per_section=6, cap_total=30)
        else:
            secs_fb_legacy = parse_jd_legacy(jd_txt_fb)
            crits_fb, cat_map_fb = build_criteria_legacy(secs_fb_legacy, per_section=6, cap_total=30)
        if crits_fb:
            st.session_state.criteria_text = "\n".join(crits_fb)
            st.session_state.cat_map = cat_map_fb
            criteria = parse_criteria_text(st.session_state.criteria_text)
            try:
                st.toast(f"Built {len(criteria)} criteria automatically.", icon="🧩")
            except Exception:
                pass

    if not criteria:
        st.warning("No criteria to analyze. Build criteria (Job Description tab) then click Analyze.")
    elif not st.session_state.get("cached_candidates"):
        st.warning("Please upload candidate resumes before running Analyze.")
    else:
        with st.spinner("Running analysis…"):
            _update_stage("🔍 Extracting JD & building criteria…")
            _update_stage("📊 Scoring candidates…")
            cov, ins_local, snips, ev_map = analyze_candidates(
                st.session_state.cached_candidates, criteria, weights,
                chunk_chars=st.session_state.get("chunk_chars",1200),
                overlap=st.session_state.get("overlap",150)
            )
            st.session_state.last_coverage = cov
            st.session_state.last_insights = ins_local
            st.session_state.last_snippets = snips
            st.session_state.evidence_map = ev_map

            # ---- Optional GPT insight upgrade ----
            engine = st.session_state.get("insight_engine", "gpt-4o-mini")
            use_gpt = engine in ("gpt-4o", "gpt-4o-mini")
            api_ok = _cached_openai_client()[0] is not None

            if use_gpt and api_ok and isinstance(cov, pd.DataFrame) and not cov.empty:
                _update_stage("✨ Generating insights (GPT)…")
                try:
                    with st.spinner("Generating GPT insights…"):
                        jd_text_full = st.session_state.jd.text if st.session_state.get("jd") else ""
                        upgraded = {}
                        crit_cols = [c for c in cov.columns if c not in ("Candidate","Overall")]
                        for _, row in cov.iterrows():
                            cand_name = row["Candidate"]
                            # Row dict: {criterion: score}
                            row_dict = {c: float(row[c]) for c in crit_cols}
                            # Candidate full text (for better context) — use cached candidate list
                            cand_text = ""
                            for c_obj in st.session_state.get("cached_candidates", []):
                                if c_obj.name == cand_name:
                                    cand_text = c_obj.text
                                    break
                            upgraded[cand_name] = gpt_candidate_insights(
                                engine=engine,
                                candidate_name=cand_name,
                                candidate_text=cand_text,
                                jd_text=jd_text_full,
                                criteria=crit_cols,
                                coverage_row=row_dict,
                                evidence_map=st.session_state.get("evidence_map", {}),
                                redacted=st.session_state.get("redacted_mode", False)
                            ) or st.session_state["last_insights"].get(cand_name, {})
                        # Replace insights with GPT versions where available
                        st.session_state["last_insights"] = upgraded
                        try: st.toast("Insights enhanced with GPT.", icon="✨")
                        except Exception: pass
                except Exception:
                    st.warning("GPT insight generation failed — using local insights.")
            elif use_gpt and not api_ok:
                st.info("GPT insight engine selected but no OpenAI API is configured. Using Local insights.")

        st.session_state._trigger_analyze = False
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
        engine = st.session_state.get("insight_engine", "gpt-4o-mini")
        use_gpt = engine in ("gpt-4o", "gpt-4o-mini")
        api_ok = _cached_openai_client()[0] is not None
        if not use_gpt:
            st.info("Insight engine is set to 'Local'. Switch to gpt-4o-mini or gpt-4o to refresh GPT insights.")
            st.session_state._trigger_refresh_insights = False
        elif not api_ok:
            st.info("OpenAI API not configured. Set OPENAI_API_KEY to refresh GPT insights.")
            st.session_state._trigger_refresh_insights = False
        else:
            _update_stage("✨ Generating insights (GPT)…")
            with st.spinner("Generating GPT insights…"):
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
                        engine=engine,
                        candidate_name=cand_name,
                        candidate_text=cand_text,
                        jd_text=jd_text_full,
                        criteria=crit_cols,
                        coverage_row=row_dict,
                        evidence_map=st.session_state.get("evidence_map", {}),
                        redacted=st.session_state.get("redacted_mode", False)
                    ) or st.session_state.get("last_insights", {}).get(cand_name, {})
                st.session_state["last_insights"] = upgraded
            _stage_ph.empty()
            _top_container.success("✅ GPT insights refreshed.")
            st.session_state._trigger_refresh_insights = False
            st.rerun()