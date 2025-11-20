import os
import re
import json
import tempfile
from typing import Dict, Any, Optional, List
import pandas as pd
import streamlit as st
from math import sqrt

# --- Load .env if available ---
try:
    from dotenv import load_dotenv  # pip install python-dotenv
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

def call_gpt(messages, model: str = "gpt-4o-mini", temperature: float = 0.1) -> str:
    """
    Call OpenAI using the modern client (openai>=1.0.0).
    Requires OPENAI_API_KEY to be set (via .env or environment).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env or environment variables.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content

# -------------------------------
# Text extraction helpers
# -------------------------------
def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n?|\u00a0", "\n", text)  # normalize line endings / nbsp
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse huge gaps
    return text.strip()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    extracted = ""

    # Primary: PyMuPDF
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

    # Fallback: OCR (pdf2image + pytesseract)
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
    """Try to pull the first top-level JSON object from a string and parse it."""
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
    """Send text with BEGIN/END tags; return {'data': <dict or None>, 'raw': <str>, 'error': <str or None>}."""
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
# Generic normaliser + richer (non-embedding) scorer
# -------------------------------
def norm_skill_set(values) -> set:
    """
    Generic normaliser: lowercase, split on common separators, strip punctuation
    (keeps + . # for C++, C#, Power BI). No role-specific synonyms.
    """
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

def score_candidate(candidate: dict, jd: dict) -> dict:
    """
    Richer skill match (no embeddings):
      - Normalize skills (split & clean)
      - Exact overlap on normalized phrases
      - Token-level overlap to capture phrasing differences
    Final score = 70% exact phrase coverage + 30% token coverage.
    """
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

def _get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key)

def _embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    if not texts:
        return []
    client = _get_openai_client()
    resp = client.embeddings.create(model=model, input=texts)
    # each item -> .embedding (list[float])
    return [d.embedding for d in resp.data]

def _cos_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a)); nb = sqrt(sum(y*y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot/(na*nb)

def score_candidate_semantic(candidate: dict, jd: dict, threshold: float = 0.82) -> dict:
    """
    Generic semantic scorer:
      - Exact phrase overlap on normalized skills
      - Token-level overlap (phrasing differences)
      - Semantic coverage via embeddings (soft matches)
    Final = 60% exact + 15% token + 25% semantic.
    """
    cand_sk = sorted(norm_skill_set(candidate.get("skills", [])))
    jd_sk   = sorted(norm_skill_set(jd.get("key_skills", [])))

    # Exact + token overlap (same as heuristic)
    exact = sorted(set(cand_sk) & set(jd_sk))
    cand_tokens = set().union(*[_tokset(s) for s in cand_sk]) if cand_sk else set()
    jd_tokens   = set().union(*[_tokset(s) for s in jd_sk]) if jd_sk else set()
    token_overlap = sorted(jd_tokens & cand_tokens)

    # Semantic: best match per JD skill over candidate skills
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
                semantic_matches.append({
                    "jd": jd_sk[j_idx],
                    "candidate": best_cand,
                    "similarity": round(best_s, 3)
                })

    # Scores
    denom_exact = max(1, len(jd_sk))
    exact_score = len(exact) / denom_exact
    denom_tok   = max(1, len(jd_tokens))
    token_score = len(token_overlap) / denom_tok
    denom_sem   = max(1, len(jd_sk))
    sem_score   = len(semantic_matches) / denom_sem

    final = round(100 * (0.60 * exact_score + 0.15 * token_score + 0.25 * sem_score), 1)

    gaps_exact = sorted([s for s in jd_sk if s not in set(exact)])

    return {
        "score": final,
        "exact_overlap": exact,
        "token_overlap_count": len(token_overlap),
        "semantic_matches": semantic_matches,   # [{jd, candidate, similarity}]
        "jd_skill_count": len(jd_sk),
        "candidate_skill_count": len(cand_sk),
        "gaps_exact": gaps_exact,
    }


# -------------------------------
# Streamlit App (clean UI + multi-CV + rankings)
# -------------------------------
st.set_page_config(page_title="Candidate Pack Summariser", layout="wide")
st.title("📄 Candidate Pack Summariser")

with st.sidebar:
    st.header("Settings")
    if os.getenv("OPENAI_API_KEY"):
        st.success("OPENAI_API_KEY: found")
    else:
        st.error("OPENAI_API_KEY: missing. Create a .env with OPENAI_API_KEY=... and restart.")

    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.caption("Use gpt-4o for higher accuracy; gpt-4o-mini for speed.")

    # 🟩 ADD THIS NEW SECTION HERE
    st.divider()
    st.subheader("Matching options")
    USE_SEMANTIC = st.checkbox("Use semantic matching (embeddings)", value=True)
    SEM_THRESHOLD = st.slider(
        "Semantic match threshold",
        0.70, 0.95, 0.82, 0.01
    )
    st.caption("Higher threshold = stricter semantic matches. 0.82 is a sensible default.")


tabs = st.tabs(["Analysis", "Developer"])

# ===========================
# ANALYSIS TAB (clean UI)
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

    # Storage for results
    analysis_rows = []
    candidate_packets = []  # keep per-candidate dicts for Developer tab

    if run_bulk:
        if jd_file is None:
            st.error("Please upload a Job Description first.")
        elif not cv_files:
            st.error("Please upload at least one Candidate CV.")
        else:
            # Extract JD text & structure once
            jd_bytes = jd_file.read()
            jd_text = extract_text(jd_file.name, jd_bytes)
            if len(jd_text.strip()) < 50:
                st.error("No readable JD text found. Try another file or enable OCR dependencies.")
            else:
                with st.spinner("Analysing Job Description with GPT…"):
                    jd_res = parse_jd(jd_text, model=model)  # {'data','raw','error'}
                jd_data = jd_res["data"] or {}

                # Process each CV
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

                        cand_res = extract_candidate(cv_text, model=model)  # {'data','raw','error'}
                        cand_data = cand_res["data"] or {}
                        display_name = cand_data.get("name") or os.path.splitext(fname)[0]

                        # Heuristic score (always)
                        try:
                            s_heur = score_candidate(cand_data, jd_data)
                        except Exception as e:
                            s_heur = {"score": 0, "error": str(e), "exact_overlap": [], "gaps": []}

                        # Semantic score (optional)
                        s_sem = None
                        if USE_SEMANTIC:
                            try:
                                s_sem = score_candidate_semantic(cand_data, jd_data, threshold=SEM_THRESHOLD)
                            except Exception as e:
                                s_sem = {"score": 0, "error": str(e), "semantic_matches": []}

                        # Row for table
                        row = {
                            "Candidate": display_name,
                            "File": fname,
                            "Score (heuristic)": s_heur.get("score", 0),
                            "Exact Overlaps": len(s_heur.get("exact_overlap", [])),
                            "Token Overlap": s_heur.get("token_overlap_count", 0),
                            "Gaps (count)": len(s_heur.get("gaps", [])),
                        }
                        if s_sem is not None:
                            row["Score (semantic)"] = s_sem.get("score", 0)
                            row["Semantic matches"] = len(s_sem.get("semantic_matches", []))
                        analysis_rows.append(row)

                        # Keep full packet for Developer tab
                        candidate_packets.append({
                            "file_name": fname,
                            "cv_text": cv_text,
                            "cand_res": cand_res,
                            # store scores if you want to inspect later
                            "score_heur": s_heur,
                            "score_sem": s_sem,
                        })

                # Rankings table
                if analysis_rows:
                    df = pd.DataFrame(analysis_rows)
                    sort_cols = ["Score (semantic)"] if ("Score (semantic)" in df.columns) else ["Score (heuristic)"]
                    # Use secondary keys to stabilise tie-breaks
                    extras = [c for c in ["Exact Overlaps", "Token Overlap"] if c in df.columns]
                    df = df.sort_values(by=sort_cols + extras, ascending=False).reset_index(drop=True)
                    df.index = df.index + 1  # 1-based rank

                    st.subheader("Ranked Candidates")
                    st.dataframe(df, use_container_width=True)

                    st.download_button(
                        "Download results (CSV)",
                        data=df.to_csv(index=True).encode("utf-8"),
                        file_name="candidate_ranking.csv",
                        mime="text/csv",
                    )


                # Keep JD packet in session for Developer tab
                st.session_state["jd_packet"] = {"file_name": jd_file.name, "jd_text": jd_text, "jd_res": jd_res}
                st.session_state["candidate_packets"] = candidate_packets

# ===========================
# DEVELOPER TAB (all the previews & raw JSON)
# ===========================
with tabs[1]:
    st.caption("Developer/testing tools. Use this for debugging prompts, previewing extractions, and inspecting raw model output.")

    # diagnostics – show lengths if available
    if "jd_packet" in st.session_state:
        jd_len = len(st.session_state["jd_packet"]["jd_text"])
        st.markdown(f"**JD text length:** {jd_len} chars")
    if "candidate_packets" in st.session_state:
        totals = sum(1 for _ in st.session_state["candidate_packets"])
        st.markdown(f"**Candidates loaded:** {totals}")

    st.divider()
    DEBUG_SHOW_PROMPTS = st.checkbox("Show prompts sent to GPT", value=False)

    # JD preview & raw
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


    # Candidate chooser
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

