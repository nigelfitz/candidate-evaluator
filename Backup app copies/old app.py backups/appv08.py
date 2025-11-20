import os
import re
import io
import json
import tempfile
import zipfile
from math import sqrt
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd
import streamlit as st

# --- Load .env if available ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional imports used in extraction fallbacks
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pdf2image import convert_from_bytes
    import pytesseract
except Exception:
    convert_from_bytes = None
    pytesseract = None

try:
    import docx  # python-docx
except Exception:
    docx = None

# -------------------------------
# OpenAI (modern SDK)
# -------------------------------
from openai import OpenAI

def _get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env.")
    return OpenAI(api_key=key)

def call_gpt(messages, model: str = "gpt-4o-mini", temperature: float = 0.1) -> str:
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content

def _embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    if not texts:
        return []
    client = _get_openai_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def _cos_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a)); nb = sqrt(sum(y*y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot/(na*nb)

# -------------------------------
# Text extraction helpers
# -------------------------------
def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n?|\u00a0", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    extracted = ""
    if fitz is not None:
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                parts = []
                for page in doc:
                    txt = page.get_text("text")
                    if txt:
                        parts.append(txt)
                extracted = "\n".join(parts)
        except Exception:
            extracted = ""
    if (not extracted or len(extracted.strip()) < 50) and convert_from_bytes and pytesseract:
        try:
            images = convert_from_bytes(file_bytes, dpi=300)
            ocr_parts = [pytesseract.image_to_string(img) for img in images]
            extracted = "\n".join(ocr_parts)
        except Exception:
            pass
    return _clean_text(extracted)

essential_docx_warning = (
    "python-docx not installed. Install with `pip install python-docx` to parse .docx files."
)

def extract_text_from_docx(file_bytes: bytes) -> str:
    if docx is None:
        return essential_docx_warning
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            path = tmp.name
        document = docx.Document(path)
        paras = [p.text for p in document.paragraphs]
        return _clean_text("\n".join(paras))
    except Exception as e:
        return f"Error reading .docx: {e}"

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return _clean_text(file_bytes.decode("utf-8", errors="ignore"))
    except Exception:
        return _clean_text(file_bytes.decode("latin-1", errors="ignore"))

def extract_text(file_name: str, file_bytes: bytes) -> str:
    name = (file_name or "").lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif name.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        return extract_text_from_pdf(file_bytes)

# -------------------------------
# Prompt builders + safe JSON extraction
# -------------------------------
def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    while start != -1:
        brace = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                brace += 1
            elif text[i] == "}":
                brace -= 1
                if brace == 0:
                    cand = text[start : i + 1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        break
        start = text.find("{", start + 1)
    try:
        normalized = text.replace("'", '"')
        return json.loads(normalized)
    except Exception:
        return None

MAX_CHARS = 12000

def build_user_prompt(label: str, text: str) -> str:
    text = text or ""
    trimmed = text[:MAX_CHARS]
    return (
        f"BEGIN_{label}_TEXT len={len(text)} trimmed_to={len(trimmed)}\n"
        f"{trimmed}\nEND_{label}_TEXT"
    )

def run_gpt_json(label: str, text: str, model: str, sys_prompt: str) -> Dict[str, Any]:
    user_prompt = build_user_prompt(label, text)
    try:
        raw = call_gpt(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.1,
        )
        data = _extract_json_block(raw)
        return {"data": data, "raw": raw, "error": None}
    except Exception as e:
        return {"data": None, "raw": "", "error": str(e)}

# -------------------------------
# GPT-powered extractors (generic)
# -------------------------------
def extract_candidate(cv_text: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    if not cv_text or len(cv_text.strip()) < 50:
        return {"data": None, "raw": "", "error": "No readable text found in CV."}
    sys_prompt = (
        "You are an expert HR analyst. You are given the candidate CV between explicit tags.\n"
        "Use ONLY that text. Output exactly one JSON object with fields:\n"
        "name, email, phone, summary, education, experience, skills, certifications, linkedin, github.\n"
        "For lists, use arrays. If a field is not present, return an empty string/array.\n"
        "Also include numeric field _read_len = number of characters you read."
    )
    return run_gpt_json("CV", cv_text, model, sys_prompt)

def parse_jd(jd_text: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    if not jd_text or len(jd_text.strip()) < 50:
        return {"data": None, "raw": "", "error": "No readable text found in JD."}
    sys_prompt = (
        "You are an HR assistant. You are given the job description between explicit tags.\n"
        "Use ONLY that text. Output exactly one JSON object with fields:\n"
        "role_title, department, location, employment_type, key_skills, responsibilities, "
        "qualifications, experience_required.\n"
        "Lists are arrays. Include numeric field _read_len."
    )
    return run_gpt_json("JD", jd_text, model, sys_prompt)

# -------------------------------
# Generic normaliser + heuristic scorer
# -------------------------------
def norm_skill_set(values) -> set:
    out = set()
    if not isinstance(values, list):
        return out
    for v in values:
        if not isinstance(v, str):
            continue
        parts = re.split(r"[,/;•|]", v)
        for p in parts:
            s = p.lower().replace("&", " and ")
            s = re.sub(r"[^a-z0-9+.# ]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            if s:
                out.add(s)
    return out

def _tokset(s: str) -> set:
    return set(re.findall(r"[a-z0-9+.#]+", s.lower()))

def score_candidate_heuristic(candidate: dict, jd: dict) -> dict:
    cand_sk = norm_skill_set(candidate.get("skills", []))
    jd_sk   = norm_skill_set(jd.get("key_skills", []))
    exact_overlap = sorted(cand_sk & jd_sk)
    cand_tokens = set().union(*[_tokset(s) for s in cand_sk]) if cand_sk else set()
    jd_tokens   = set().union(*[_tokset(s) for s in jd_sk]) if jd_sk else set()
    token_overlap = sorted(jd_tokens & cand_tokens)
    denom_exact = max(1, len(jd_sk))
    exact_score = len(exact_overlap) / denom_exact
    denom_tok   = max(1, len(jd_tokens))
    token_score = len(token_overlap) / denom_tok
    final = round(100 * (0.7 * exact_score + 0.3 * token_score), 1)
    gaps = sorted([s for s in jd_sk if s not in cand_sk])
    return {
        "score": final,
        "exact_overlap": exact_overlap,
        "token_overlap_count": len(token_overlap),
        "jd_skill_count": len(jd_sk),
        "candidate_skill_count": len(cand_sk),
        "gaps": gaps,
    }

# -------------------------------
# Semantic scorer (embeddings)
# -------------------------------
def score_candidate_semantic(candidate: dict, jd: dict, threshold: float, w_exact: float, w_token: float, w_sem: float) -> dict:
    cand_sk = sorted(norm_skill_set(candidate.get("skills", [])))
    jd_sk   = sorted(norm_skill_set(jd.get("key_skills", [])))
    exact = sorted(set(cand_sk) & set(jd_sk))
    cand_tokens = set().union(*[_tokset(s) for s in cand_sk]) if cand_sk else set()
    jd_tokens   = set().union(*[_tokset(s) for s in jd_sk]) if jd_sk else set()
    token_overlap = sorted(jd_tokens & cand_tokens)
    semantic_matches = []
    if jd_sk and cand_sk:
        jd_vecs   = _embed_texts(jd_sk)
        cand_vecs = _embed_texts(cand_sk)
        for j_idx, jv in enumerate(jd_vecs):
            best_s, best_cand = 0.0, None
            for c_idx, cv in enumerate(cand_vecs):
                s = _cos_sim(jv, cv)
                if s > best_s:
                    best_s, best_cand = s, cand_sk[c_idx]
            if best_s >= threshold:
                semantic_matches.append({"jd": jd_sk[j_idx], "candidate": best_cand, "similarity": round(best_s, 3)})
    denom_exact = max(1, len(jd_sk))
    exact_score = len(exact) / denom_exact
    denom_tok   = max(1, len(jd_tokens))
    token_score = len(token_overlap) / denom_tok
    denom_sem   = max(1, len(jd_sk))
    sem_score   = len(semantic_matches) / denom_sem
    s = max(1e-9, (w_exact + w_token + w_sem))
    wx, wt, ws = w_exact/s, w_token/s, w_sem/s
    final = round(100 * (wx * exact_score + wt * token_score + ws * sem_score), 1)
    gaps_exact = sorted([s for s in jd_sk if s not in set(exact)])
    return {
        "score": final,
        "exact_overlap": exact,
        "token_overlap_count": len(token_overlap),
        "semantic_matches": semantic_matches,
        "jd_skill_count": len(jd_sk),
        "candidate_skill_count": len(cand_sk),
        "gaps_exact": gaps_exact,
    }

# -------------------------------
# Chunking helpers (long docs)
# -------------------------------
def chunk_text(text: str, target: int = 9000, overlap: int = 400) -> List[str]:
    text = text or ""
    if len(text) <= target:
        return [text]
    paras = re.split(r"\n{2,}", text)
    chunks, buf = [], ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= target:
            buf += "\n\n" + p
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + "\n\n" + p).strip()
    if buf:
        chunks.append(buf)
    return chunks

def merge_candidate_chunks(datas: List[dict]) -> dict:
    out = {
        "name": "", "email": "", "phone": "", "summary": "",
        "education": [], "experience": [], "skills": [], "certifications": [],
        "linkedin": "", "github": ""
    }
    for d in datas:
        if not isinstance(d, dict): continue
        for k, v in d.items():
            if k in ("education", "experience", "skills", "certifications"):
                if isinstance(v, list): out[k].extend(v)
            elif k.startswith("_"):
                pass
            else:
                if isinstance(v, str) and v.strip() and not out.get(k):
                    out[k] = v.strip()
    if out["skills"]:
        seen, dedup = set(), []
        for s in out["skills"]:
            if not isinstance(s, str): continue
            key = s.strip().lower()
            if key not in seen:
                seen.add(key); dedup.append(s)
        out["skills"] = dedup
    return out

def merge_jd_chunks(datas: List[dict]) -> dict:
    out = {
        "role_title": "", "department": "", "location": "", "employment_type": "",
        "key_skills": [], "responsibilities": [], "qualifications": [], "experience_required": []
    }
    for d in datas:
        if not isinstance(d, dict): continue
        for k, v in d.items():
            if k in ("key_skills", "responsibilities", "qualifications", "experience_required"):
                if isinstance(v, list): out[k].extend(v)
            elif k.startswith("_"):
                pass
            else:
                if isinstance(v, str) and v.strip() and not out.get(k):
                    out[k] = v.strip()
    for k in ("key_skills","responsibilities","qualifications","experience_required"):
        if out[k]:
            seen, dedup = set(), []
            for s in out[k]:
                if not isinstance(s, str): continue
                key = s.strip().lower()
                if key not in seen:
                    seen.add(key); dedup.append(s)
            out[k] = dedup
    return out

# -------------------------------
# Coverage & evidence helpers
# -------------------------------
def build_jd_criteria(jd_data: dict, max_items_per_section: int = 20) -> list[dict]:
    crits = []
    for section in ["key_skills", "responsibilities", "qualifications", "experience_required"]:
        vals = jd_data.get(section) or []
        if isinstance(vals, list):
            for v in vals[:max_items_per_section]:
                if isinstance(v, str):
                    t = v.strip()
                    if t:
                        crits.append({"section": section, "text": t})
    return crits

def jd_field_completeness(jd_data: dict) -> dict:
    fields = ["role_title","department","location","employment_type",
              "key_skills","responsibilities","qualifications","experience_required"]
    out = {}
    for f in fields:
        v = jd_data.get(f)
        if isinstance(v, list):
            out[f] = len([x for x in v if isinstance(x, str) and x.strip() > ""])
        elif isinstance(v, str):
            out[f] = 1 if v.strip() else 0
        else:
            out[f] = 0
    out["_populated_fields"] = sum(1 for f in fields if out.get(f,0) > 0)
    out["_total_fields"] = len(fields)
    out["_completeness_pct"] = round(100 * out["_populated_fields"] / out["_total_fields"], 1)
    return out

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
def find_evidence_snippet(cv_text: str, criterion: str, window_chars: int = 220) -> str:
    if not isinstance(cv_text, str) or not cv_text.strip():
        return ""
    if not isinstance(criterion, str) or not criterion.strip():
        return ""
    crit = criterion.strip()
    pos = cv_text.lower().find(crit.lower())
    if pos != -1:
        start = max(0, pos - window_chars)
        end   = min(len(cv_text), pos + len(crit) + window_chars)
        return cv_text[start:end].strip()
    tokens = [t for t in re.findall(r"[a-z0-9+.#]{3,}", crit.lower()) if t not in {"and","the","for","with","to","of"}]
    if not tokens:
        return ""
    for sent in _SENT_SPLIT.split(cv_text):
        lo = sent.lower()
        if any(t in lo for t in tokens):
            return sent.strip()[: 2*window_chars]
    return ""

def _coverage_status_from_scores(exact_hit: bool, token_cov: float, semantic_hit: bool) -> tuple[str, str]:
    if exact_hit or semantic_hit or token_cov >= 0.85:
        return ("Meets", "✅")
    if token_cov >= 0.35:
        return ("Partial", "⚠️")
    return ("Missing", "⛔")

def _token_coverage(criterion: str, cand_tokens: set) -> float:
    toks = set(re.findall(r"[a-z0-9+.#]+", criterion.lower()))
    toks = {t for t in toks if t not in {"and","the","for","with","to","of"}}
    return 0.0 if not toks else len(toks & cand_tokens) / len(toks)

# Embedding cache in session_state (persists across reruns)
def _get_emb_cache():
    if "emb_cache" not in st.session_state:
        st.session_state["emb_cache"] = {}
    return st.session_state["emb_cache"]

def get_embedding_cached(text: str, model: str = "text-embedding-3-small") -> list[float]:
    cache = _get_emb_cache()
    key = (model, text)
    if key in cache:
        return cache[key]
    vec = _embed_texts([text], model=model)[0]
    cache[key] = vec
    return vec

def semantic_hit(criterion: str, cand_skill_list: list[str], threshold: float) -> tuple[bool, float, str]:
    if not cand_skill_list:
        return (False, 0.0, "")
    try:
        jv = get_embedding_cached(criterion)
        best_s, best_skill = 0.0, ""
        for s in cand_skill_list:
            cv = get_embedding_cached(s)
            sim = _cos_sim(jv, cv)
            if sim > best_s:
                best_s, best_skill = sim, s
        return (best_s >= threshold, best_s, best_skill)
    except Exception:
        return (False, 0.0, "")

# -------------------------------
# Candidate insights (GPT)
# -------------------------------
def generate_candidate_insight(cand: dict, jd: dict, scores: dict, model: str = "gpt-4o-mini") -> str:
    """
    Produce a short, human-readable summary of strengths, gaps, and fit.
    """
    sys = (
        "You are an expert recruiter. Given structured candidate, job description, and scores, "
        "write a concise, bullet-point insight summary (4-6 bullets) using neutral tone. "
        "First line: Overall fit (High/Medium/Low) and a one-line rationale."
    )
    payload = {
        "candidate": cand or {},
        "job": jd or {},
        "scores": scores or {},
    }
    user = (
        "Summarise the candidate against the JD. Use: strengths, concerns, likely gaps, notable evidence.\n"
        "Return plain text bullets (no JSON). Keep to ~120-180 words."
        f"\n\nDATA:\n{json.dumps(payload, indent=2)[:6000]}"
    )
    return call_gpt([{"role": "system", "content": sys}, {"role": "user", "content": user}], model=model, temperature=0.2)

# -------------------------------
# App: UI
# -------------------------------
st.set_page_config(page_title="Candidate Pack Summariser", layout="wide")
st.title("📄 Candidate Pack Summariser")

with st.sidebar:
    with st.expander("⚙️ Settings", expanded=False):
        if os.getenv("OPENAI_API_KEY"):
            st.success("OPENAI_API_KEY: found")
        else:
            st.error("OPENAI_API_KEY: missing. Add to .env and restart.")

        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
        st.caption("Use gpt-4o for higher accuracy; gpt-4o-mini for speed.")

        st.divider()
        st.subheader("Matching options")
        USE_SEMANTIC = st.checkbox("Use semantic matching (embeddings)", value=True)
        SEM_THRESHOLD = st.slider("Semantic match threshold", 0.70, 0.95, 0.82, 0.01)
        st.caption("Higher threshold = stricter semantic matches.")

        st.markdown("**Weights (exact/token/semantic)**")
        W_EXACT = st.slider("Weight: exact", 0.0, 1.0, 0.60, 0.05)
        W_TOKEN = st.slider("Weight: token", 0.0, 1.0, 0.15, 0.05)
        W_SEM   = st.slider("Weight: semantic", 0.0, 1.0, 0.25, 0.05)
        st.caption("Weights are normalized to sum to 1 for the final score.")

        st.divider()
        st.subheader("Long documents")
        USE_CHUNKING = st.checkbox("Use chunking for long CV/JD", value=True)
        CHUNK_TARGET = st.slider("Chunk size (chars)", 6000, 12000, 9000, 500)

        st.divider()
        st.subheader("Insights")
        MAKE_INSIGHTS = st.checkbox("Generate candidate insight panels (uses GPT)", value=True)

        st.divider()
        st.subheader("Cost meter (optional)")
        st.caption("We estimate input tokens by ~4 chars/token. Enter your own prices (per 1K tokens).")
        PRICE_CHAT_IN  = st.number_input("Price: chat input ($/1K tok)", 0.0, 10.0, 0.0, 0.01)
        PRICE_EMB_IN   = st.number_input("Price: embedding input ($/1K tok)", 0.0, 10.0, 0.0, 0.01)

tabs = st.tabs(["Analysis", "Developer"])

# Shared helpers for cost meter
def _approx_tokens(chars: int) -> int:
    return max(1, int(round(chars / 4)))

def _add_cost(costs: dict, key: str, tokens: int, price_per_1k: float):
    if price_per_1k <= 0:
        return
    costs[key] = costs.get(key, 0.0) + (tokens / 1000.0) * price_per_1k

# ===========================
# ANALYSIS TAB
# ===========================
with tabs[0]:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload Job Description (single file)")
        jd_file = st.file_uploader("JD file (.pdf/.docx/.txt)", type=["pdf", "docx", "txt"], key="jd_single")

    with col2:
        st.subheader("Upload Candidate CVs (multiple)")
        cv_files: List = st.file_uploader(
            "CV files (.pdf/.docx/.txt)", type=["pdf", "docx", "txt"], key="cv_multi", accept_multiple_files=True
        )

    run_bulk = st.button("Analyse All Candidates")

    analysis_rows = []
    candidate_packets = []
    estimated_costs = {}
    jd_data = {}

    if run_bulk:
        if jd_file is None:
            st.error("Please upload a Job Description first.")
        elif not cv_files:
            st.error("Please upload at least one Candidate CV.")
        else:
            jd_bytes = jd_file.read()
            jd_text = extract_text(jd_file.name, jd_bytes)
            if len(jd_text.strip()) < 50:
                st.error("No readable JD text found. Try another file or enable OCR dependencies.")
            else:
                # --- Parse JD (with chunking) ---
                if USE_CHUNKING and len(jd_text) > MAX_CHARS:
                    chunks = chunk_text(jd_text, target=CHUNK_TARGET)
                    per = []
                    with st.spinner(f"Analysing JD in {len(chunks)} chunk(s)…"):
                        for ch in chunks:
                            _add_cost(estimated_costs, "chat_input_$", _approx_tokens(len(ch)), PRICE_CHAT_IN)
                            r = parse_jd(ch, model=model)
                            if r.get("data"): per.append(r["data"])
                    jd_data = merge_jd_chunks(per) if per else {}
                    jd_res = {"data": jd_data, "raw": "(chunked)", "error": None}
                else:
                    with st.spinner("Analysing Job Description with GPT…"):
                        _add_cost(estimated_costs, "chat_input_$", _approx_tokens(len(jd_text)), PRICE_CHAT_IN)
                        jd_res = parse_jd(jd_text, model=model)
                    jd_data = jd_res["data"] or {}

                # --- Dashboard summary (header stats) ---
                st.subheader("Dashboard")
                if jd_data:
                    crits_tmp = build_jd_criteria(jd_data, max_items_per_section=999)
                    st.markdown(f"- **JD criteria detected:** {len(crits_tmp)}")
                else:
                    st.markdown("- **JD criteria detected:** 0")

                # --- Analyse candidates ---
                with st.spinner(f"Analysing {len(cv_files)} candidate(s)…"):
                    for f in cv_files:
                        fname = f.name
                        cbytes = f.read()
                        cv_text = extract_text(fname, cbytes)
                        if len(cv_text.strip()) < 50:
                            candidate_packets.append({
                                "file_name": fname,
                                "cv_text": cv_text,
                                "cand_res": {"data": None, "raw": "", "error": "No readable text"},
                            })
                            continue

                        if USE_CHUNKING and len(cv_text) > MAX_CHARS:
                            chunks = chunk_text(cv_text, target=CHUNK_TARGET)
                            per = []
                            for ch in chunks:
                                _add_cost(estimated_costs, "chat_input_$", _approx_tokens(len(ch)), PRICE_CHAT_IN)
                                r = extract_candidate(ch, model=model)
                                if r.get("data"): per.append(r["data"])
                            cand_data = merge_candidate_chunks(per) if per else {}
                            cand_res = {"data": cand_data, "raw": "(chunked)", "error": None}
                        else:
                            _add_cost(estimated_costs, "chat_input_$", _approx_tokens(len(cv_text)), PRICE_CHAT_IN)
                            cand_res = extract_candidate(cv_text, model=model)
                            cand_data = cand_res["data"] or {}

                        display_name = cand_data.get("name") or os.path.splitext(fname)[0]

                        # Heuristic score
                        try:
                            s_heur = score_candidate_heuristic(cand_data, jd_data)
                        except Exception as e:
                            s_heur = {"score": 0, "error": str(e), "exact_overlap": [], "gaps": []}

                        # Semantic score (optional)
                        s_sem = None
                        if USE_SEMANTIC:
                            cand_sk = sorted(norm_skill_set(cand_data.get("skills", [])))
                            jd_sk   = sorted(norm_skill_set(jd_data.get("key_skills", [])))
                            emb_chars = sum(len(x) for x in cand_sk + jd_sk)
                            _add_cost(estimated_costs, "embedding_input_$", _approx_tokens(emb_chars), PRICE_EMB_IN)
                            try:
                                s_sem = score_candidate_semantic(
                                    cand_data, jd_data, threshold=SEM_THRESHOLD,
                                    w_exact=W_EXACT, w_token=W_TOKEN, w_sem=W_SEM
                                )
                            except Exception as e:
                                s_sem = {"score": 0, "error": str(e), "semantic_matches": []}

                        # Candidate completeness
                        key_fields = ["name","email","phone","summary","education","experience","skills"]
                        nonempty = 0
                        for k in key_fields:
                            v = cand_data.get(k)
                            if isinstance(v, str):
                                nonempty += 1 if v.strip() else 0
                            elif isinstance(v, list):
                                nonempty += 1 if any(isinstance(x, str) and x.strip() for x in v) else 0
                        completeness = round(nonempty / len(key_fields), 2)

                        row = {
                            "Candidate": display_name,
                            "File": fname,
                            "Score (heuristic)": s_heur.get("score", 0),
                            "Exact Overlaps": len(s_heur.get("exact_overlap", [])),
                            "Token Overlap": s_heur.get("token_overlap_count", 0),
                            "Gaps (count)": len(s_heur.get("gaps", [])),
                            "Candidate completeness": completeness,
                        }
                        if s_sem is not None:
                            row["Score (semantic)"] = s_sem.get("score", 0)
                            row["Semantic matches"] = len(s_sem.get("semantic_matches", []))
                        analysis_rows.append(row)

                        candidate_packets.append({
                            "file_name": fname,
                            "cv_text": cv_text,
                            "cand_res": cand_res,
                            "score_heur": s_heur,
                            "score_sem": s_sem,
                        })

                # Rankings table
                if analysis_rows:
                    df = pd.DataFrame(analysis_rows)
                    sort_cols = ["Score (semantic)"] if ("Score (semantic)" in df.columns) else ["Score (heuristic)"]
                    extras = [c for c in ["Exact Overlaps", "Token Overlap"] if c in df.columns]
                    df = df.sort_values(by=sort_cols + extras, ascending=False).reset_index(drop=True)
                    df.index = df.index + 1

                    # Dashboard summary extras
                    top_name = df.iloc[0]["Candidate"] if not df.empty else "—"
                    st.markdown(f"- **Top candidate:** {top_name}")
                    if "Score (semantic)" in df.columns:
                        st.markdown(f"- **Average semantic score:** {round(float(df['Score (semantic)'].mean()),1)}")
                    st.markdown(f"- **Average heuristic score:** {round(float(df['Score (heuristic)'].mean()),1)}")

                    # -----------------------
                    # Scoring legend / explanation
                    # -----------------------
                    with st.expander("ℹ️ Scoring legend", expanded=False):
                        st.markdown(f"""
                    **Heuristic score** → based on **exact phrase** and **token overlap** between the JD and candidate CV.  
                    - Transparent and rule-based (no AI meaning model).  
                    - Fast and auditable.  
                    - Good for exact keywords (e.g., “CPA”, “Power BI”).

                    **Semantic score** → adds **meaning-based** matching using OpenAI embeddings (cosine similarity).  
                    - Finds similar or related phrases even when worded differently.  
                    - Controlled by **semantic threshold** = `{SEM_THRESHOLD}`.  
                    - Weighted combination of:
                    - Exact: **{W_EXACT:.2f}**
                    - Token: **{W_TOKEN:.2f}**
                    - Semantic: **{W_SEM:.2f}**
                    (Weights are normalised to sum to 1.)

                    💡 **Tip:**  
                    Use *heuristic only* when you want strict keyword matching.  
                    Use *semantic ON* for broader understanding and flexible phrasing matches.
                    """)


                    st.subheader("Ranked Candidates")
                    st.dataframe(df, width="stretch")

                    st.download_button(
                        "Download results (CSV)",
                        data=df.to_csv(index=True).encode("utf-8"),
                        file_name="candidate_ranking.csv",
                        mime="text/csv",
                    )

                    # Save run → ZIP (JD + table + candidates)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr(f"run_{ts}/jd.json", json.dumps(jd_data, indent=2))
                        zf.writestr(f"run_{ts}/candidate_ranking.csv", df.to_csv(index=True))
                        for pkt in candidate_packets:
                            base = os.path.splitext(os.path.basename(pkt["file_name"]))[0]
                            cand_json = pkt["cand_res"]["data"] or {}
                            zf.writestr(f"run_{ts}/candidates/{base}.json", json.dumps(cand_json, indent=2))
                    st.download_button(
                        "Save run (ZIP: JD + table + candidates)",
                        data=zip_buf.getvalue(),
                        file_name=f"run_{ts}.zip",
                        mime="application/zip",
                    )

                    if PRICE_CHAT_IN > 0 or PRICE_EMB_IN > 0:
                        st.subheader("Estimated cost (inputs only)")
                        pretty = {k: f"${v:0.4f}" for k, v in estimated_costs.items()}
                        st.json(pretty)

                # ----- Field completeness + coverage matrix -----
                if jd_data and analysis_rows:
                    st.subheader("JD field completeness")
                    comp = jd_field_completeness(jd_data)
                    st.markdown(f"**Completeness:** {comp['_completeness_pct']}% "
                                f"({comp['_populated_fields']}/{comp['_total_fields']} populated)")

                    criteria = build_jd_criteria(jd_data, max_items_per_section=50)
                    if not criteria:
                        st.info("No JD criteria found to compare.")
                    else:
                        st.subheader("Criteria coverage matrix")

                        # Prepare normalized skills & tokens per candidate
                        cand_norm_skills = {}
                        cand_tokens = {}
                        for pkt in candidate_packets:
                            cand_data_i = pkt["cand_res"]["data"] or {}
                            nskills = sorted(norm_skill_set(cand_data_i.get("skills", [])))
                            cand_norm_skills[pkt["file_name"]] = nskills
                            cand_tokens[pkt["file_name"]] = set().union(*[_tokset(s) for s in nskills]) if nskills else set()

                        cand_files = [row["File"] for row in analysis_rows]
                        matrix_rows = []
                        detailed_records = []

                        # Live status + progress
                        try:
                            status = st.status("Building coverage matrix…", expanded=True)
                            progress = st.progress(0, text="Starting…")
                            use_status = True
                        except Exception:
                            status = st.spinner("Building coverage matrix…")
                            use_status = False

                        total = len(criteria)
                        with status:
                            for i, c in enumerate(criteria):
                                row_display = {"Criterion": f"{c['section']}: {c['text']}"}
                                for fname in cand_files:
                                    nskills = cand_norm_skills.get(fname, [])
                                    tokens  = cand_tokens.get(fname, set())
                                    exact = (c["text"].strip().lower().replace("&"," and ") in nskills)
                                    tok_cov = _token_coverage(c["text"], tokens)
                                    sem_ok, best_sim, best_skill = (False, 0.0, "")
                                    if USE_SEMANTIC:
                                        sem_ok, best_sim, best_skill = semantic_hit(c["text"], nskills, SEM_THRESHOLD)
                                    status_txt, icon = _coverage_status_from_scores(exact, tok_cov, sem_ok)
                                    pkt_i = next((p for p in candidate_packets if p["file_name"] == fname), None)
                                    snippet = find_evidence_snippet(pkt_i["cv_text"] if pkt_i else "", c["text"])
                                    row_display[fname] = icon
                                    detailed_records.append({
                                        "criterion_section": c["section"],
                                        "criterion_text": c["text"],
                                        "candidate_file": fname,
                                        "status": status_txt,
                                        "icon": icon,
                                        "exact": exact,
                                        "token_coverage": round(tok_cov, 2),
                                        "semantic_hit": sem_ok,
                                        "semantic_best_sim": round(best_sim, 3),
                                        "semantic_best_skill": best_skill,
                                        "evidence": snippet
                                    })
                                matrix_rows.append(row_display)
                                if use_status:
                                    pct = int(((i + 1) / total) * 100)
                                    progress.progress(pct, text=f"Processed {i+1}/{total} criteria")
                            if use_status:
                                status.update(label="Coverage matrix built ✅", state="complete")

                        # Build DataFrame and color style
                        mat_df = pd.DataFrame(matrix_rows)
                        display_map = {r["File"]: r["Candidate"] for r in analysis_rows}
                        nice_cols = ["Criterion"] + [display_map.get(c, c) for c in cand_files]
                        mat_df.columns = ["Criterion"] + cand_files

                        # map icons to colors
                        color_map = {"✅": "#e6ffe6", "⚠️": "#fff8e6", "⛔": "#ffe6e6"}
                        def _colorize(v):
                            return f"background-color: {color_map.get(v,'')};"

                        styled = mat_df.copy()
                        styler = styled.style
                        for c in cand_files:
                            styler = styler.applymap(_colorize, subset=[c])
                        # Rename columns for display after styling
                        styled.columns = ["Criterion"] + [display_map.get(c, c) for c in cand_files]
                        st.dataframe(styled, width="stretch")

                        # Evidence explorer
                        st.subheader("Evidence explorer")
                        crit_opts = [f"{r['section']}: {r['text']}" for r in criteria]
                        cand_opts = [display_map.get(c, c) for c in cand_files]
                        colA, colB = st.columns(2)
                        with colA:
                            pick_crit = st.selectbox("Pick a JD criterion", crit_opts, index=0)
                        with colB:
                            pick_cand_disp = st.selectbox("Pick a candidate", cand_opts, index=0)
                        inv_map = {v:k for k,v in display_map.items()}
                        pick_cand_file = inv_map.get(pick_cand_disp, pick_cand_disp)
                        sect, text = pick_crit.split(": ", 1) if ": " in pick_crit else ("key_skills", pick_crit)
                        rec = next((d for d in detailed_records
                                    if d["candidate_file"] == pick_cand_file
                                    and d["criterion_section"] == sect
                                    and d["criterion_text"] == text), None)
                        if rec:
                            st.markdown(f"**Status:** {rec['icon']} {rec['status']}")
                            st.markdown(f"**Exact phrase match:** {rec['exact']}")
                            st.markdown(f"**Token coverage:** {rec['token_coverage']}")
                            if USE_SEMANTIC:
                                st.markdown(f"**Semantic match:** {rec['semantic_hit']}  "
                                            f"(best_sim={rec['semantic_best_sim']}, vs **{rec['semantic_best_skill']}**)")
                            with st.expander("Evidence snippet from CV"):
                                st.write(rec["evidence"] or "(no snippet found)")
                        else:
                            st.info("No evidence record found for the selected combination.")

                # ----- Candidate Insight Panels -----
                if MAKE_INSIGHTS and candidate_packets:
                    st.subheader("Candidate insights")
                    for pkt in candidate_packets:
                        cand = pkt["cand_res"]["data"] or {}
                        name = cand.get("name") or os.path.splitext(pkt["file_name"])[0]
                        scores_for_llm = {"heuristic": pkt.get("score_heur") or {}}
                        if pkt.get("score_sem") is not None:
                            scores_for_llm["semantic"] = pkt.get("score_sem")
                        # estimate cost for insight prompt
                        insight_chars = len(json.dumps(cand)) + len(json.dumps(jd_data)) + 1500
                        _add_cost(estimated_costs, "chat_input_$", _approx_tokens(insight_chars), PRICE_CHAT_IN)
                        with st.expander(f"🧭 {name}", expanded=False):
                            try:
                                summary = generate_candidate_insight(cand, jd_data, scores_for_llm, model=model)
                            except Exception as e:
                                summary = f"(Could not generate insight: {e})"
                            st.write(summary)

                # Save for Developer tab
                st.session_state["jd_packet"] = {"file_name": jd_file.name, "jd_text": jd_text, "jd_res": jd_res}
                st.session_state["candidate_packets"] = candidate_packets

# ===========================
# DEVELOPER TAB
# ===========================
with tabs[1]:
    st.caption("Developer/testing tools. Inspect prompts, extracted text, parsed JSON, raw model output, and scores.")

    if "jd_packet" in st.session_state:
        jd_len = len(st.session_state["jd_packet"]["jd_text"])
        st.markdown(f"**JD text length:** {jd_len} chars")
    if "candidate_packets" in st.session_state:
        totals = sum(1 for _ in st.session_state["candidate_packets"])
        st.markdown(f"**Candidates loaded:** {totals}")

    st.divider()
    DEBUG_SHOW_PROMPTS = st.checkbox("Show prompts sent to GPT", value=False)

    if "jd_packet" in st.session_state:
        jp = st.session_state["jd_packet"]
        with st.expander("JD — extracted text (first 2,000 chars)"):
            st.text(jp["jd_text"][:2000] or "(no text)")
        if DEBUG_SHOW_PROMPTS:
            with st.expander("Prompt to GPT — JD (first 1,500 chars)"):
                st.text(build_user_prompt("JD", jp["jd_text"])[:1500])
        with st.expander("JD — parsed JSON"):
            st.json(jp["jd_res"]["data"] or {})
        with st.expander("JD — raw model output"):
            st.code(jp["jd_res"]["raw"] or "(empty)", language="json")

    pkt = None
    if "candidate_packets" in st.session_state and st.session_state["candidate_packets"]:
        names = [p["file_name"] for p in st.session_state["candidate_packets"]]
        pick = st.selectbox("Select candidate to inspect", names)
        pkt = next((p for p in st.session_state["candidate_packets"] if p["file_name"] == pick), None)

        if pkt:
            with st.expander("CV — extracted text (first 2,000 chars)"):
                st.text(pkt["cv_text"][:2000] or "(no text)")
            if DEBUG_SHOW_PROMPTS:
                with st.expander("Prompt to GPT — CV (first 1,500 chars)"):
                    st.text(build_user_prompt("CV", pkt["cv_text"])[:1500])
            with st.expander("Candidate — parsed JSON"):
                st.json(pkt["cand_res"]["data"] or {})
            with st.expander("Candidate — raw model output"):
                st.code(pkt["cand_res"]["raw"] or "(empty)", language="json")
            with st.expander("Scoring — heuristic"):
                st.json(pkt.get("score_heur") or {})
            with st.expander("Scoring — semantic"):
                st.json(pkt.get("score_sem") or {})

st.caption(
    "Tip: If PDFs return little or no text, install Tesseract and pdf2image+pytesseract for OCR fallback. "
    "For .docx, install python-docx."
)
