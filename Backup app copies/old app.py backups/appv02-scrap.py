import os
import io
import re
import json
import time
import tempfile
from typing import Dict, Any, Optional

import streamlit as st

# --- Load environment variables from .env (OPENAI_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # reads .env in the project root if present
except Exception:
    pass  # if python-dotenv isn't installed, app will still run if env vars are set

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
# Utility: OpenAI call (modern SDK)
# -------------------------------
from openai import OpenAI

def _get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env file with "
            "OPENAI_API_KEY=sk-... or set it in your environment."
        )
    return OpenAI(api_key=key)

def call_gpt(messages, model: str = "gpt-4o-mini", temperature: float = 0.1) -> str:
    """
    Call OpenAI using the modern client (openai>=1.0.0).
    """
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

# -------------------------------
# Text extraction helpers
# -------------------------------
def _clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r"\r\n?|\u00a0", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Try PyMuPDF first; if empty, fall back to OCR if available."""
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
            ocr_parts = []
            for img in images:
                ocr_parts.append(pytesseract.image_to_string(img))
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
        # python-docx needs a file-like object
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
        # Try PDF as a best-effort default
        return extract_text_from_pdf(file_bytes)

# -------------------------------
# Prompted analysis (GPT-powered)
# -------------------------------
def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Try to pull the first top-level JSON object from a string and parse it."""
    try:
        # Quick win: direct JSON
        return json.loads(text)
    except Exception:
        pass

    # Fallback: greedy brace matching
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

    # Last resort: quote normalization
    try:
        normalized = text.replace("'", '"')
        return json.loads(normalized)
    except Exception:
        return None

def extract_candidate(cv_text: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    if not cv_text or len(cv_text.strip()) < 50:
        return {"error": "No readable text found in CV."}

    system_prompt = (
        "You are an expert HR analyst. Extract key information from the candidate's CV and return JSON. "
        "Use fields: name, email, phone, summary, education, experience, skills, certifications, linkedin, github. "
        "For lists (education, experience, skills, certifications), return arrays. Do not invent details."
    )

    user_prompt = f"""
Here is the full text of the candidate's CV. Read it carefully and output ONLY a JSON object. No commentary.

"""

    try:
        content = call_gpt(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.1,
        )
        parsed = _extract_json_block(content)
        return parsed if parsed is not None else {"raw_output": content}
    except Exception as e:
        return {"error": str(e)}

def parse_jd(jd_text: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    if not jd_text or len(jd_text.strip()) < 50:
        return {"error": "No readable text found in JD."}

    system_prompt = (
        "You are an HR assistant that analyses job descriptions and returns structured JSON. "
        "Fields: role_title, department, location, employment_type, key_skills, responsibilities, "
        "qualifications, experience_required. Lists should be arrays. No extra commentary."
    )

    user_prompt = f"""
Here is the full Job Description text. Read it carefully and output ONLY a JSON object. No commentary.


"""

    try:
        content = call_gpt(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.1,
        )
        parsed = _extract_json_block(content)
        return parsed if parsed is not None else {"raw_output": content}
    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Scoring (simple baseline; replace with smarter prompt later)
# -------------------------------
def score_candidate(candidate: Dict[str, Any], jd: Dict[str, Any]) -> Dict[str, Any]:
    """Very simple overlap score on skills; placeholder until semantic scoring prompt is added."""
    cand_skills = set([s.strip().lower() for s in candidate.get("skills", []) if isinstance(s, str)])
    jd_skills = set([s.strip().lower() for s in jd.get("key_skills", []) if isinstance(s, str)])
    overlap = len(cand_skills & jd_skills)
    total = max(1, len(jd_skills))
    score = round(100 * overlap / total, 1)
    return {"score": score, "skills_overlap": sorted(cand_skills & jd_skills), "jd_skills": sorted(jd_skills)}

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Candidate Pack Summariser", layout="wide")
st.title("📄 Candidate Pack Summariser")

# Sidebar (no API key prompt — we rely on environment/.env)
with st.sidebar:
    st.header("Settings")
    # Show status of API key for clarity
    if os.getenv("OPENAI_API_KEY"):
        st.success("OPENAI_API_KEY loaded from environment/.env")
    else:
        st.error(
            "OPENAI_API_KEY not found. Create a .env file with:\n\n"
            "OPENAI_API_KEY=sk-...\n\n"
            "Then restart the app."
        )
        st.stop()

    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.caption("Use gpt-4o for higher accuracy; gpt-4o-mini for speed.")

    st.divider()
    st.subheader("Diagnostics")
    diag_cv_len = st.empty()
    diag_jd_len = st.empty()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Candidate CV")
    cv_file = st.file_uploader("CV file (.pdf/.docx/.txt)", type=["pdf", "docx", "txt"], key="cv")

with col2:
    st.subheader("Upload Job Description")
    jd_file = st.file_uploader("JD file (.pdf/.docx/.txt)", type=["pdf", "docx", "txt"], key="jd")

run = st.button("Analyse")

cv_text = ""
jd_text = ""

if cv_file is not None:
    cv_bytes = cv_file.read()
    cv_text = extract_text(cv_file.name, cv_bytes)
    diag_cv_len.markdown(f"**CV text length:** {len(cv_text)} chars")
else:
    diag_cv_len.markdown("**CV text length:** —")

if jd_file is not None:
    jd_bytes = jd_file.read()
    jd_text = extract_text(jd_file.name, jd_bytes)
    diag_jd_len.markdown(f"**JD text length:** {len(jd_text)} chars")
else:
    diag_jd_len.markdown("**JD text length:** —")

# Preview text extracts (collapsed)
with st.expander("Preview extracted CV text (first 2,000 chars)"):
    st.text(cv_text[:2000] or "(no text extracted)")
with st.expander("Preview extracted JD text (first 2,000 chars)"):
    st.text(jd_text[:2000] or "(no text extracted)")

if run:
    if not cv_text or len(cv_text.strip()) < 50:
        st.error("No readable CV text found. Try another file or enable OCR dependencies.")
    if not jd_text or len(jd_text.strip()) < 50:
        st.error("No readable JD text found. Try another file or enable OCR dependencies.")

    if cv_text and jd_text and len(cv_text) >= 50 and len(jd_text) >= 50:
        with st.spinner("Analysing with GPT…"):
            cand = extract_candidate(cv_text, model=model)
            jd_struct = parse_jd(jd_text, model=model)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Candidate (extracted)")
            st.json(cand)
            st.download_button("Download candidate.json", data=json.dumps(cand, indent=2), file_name="candidate.json")
        with c2:
            st.subheader("Job Description (extracted)")
            st.json(jd_struct)
            st.download_button("Download job_description.json", data=json.dumps(jd_struct, indent=2), file_name="job_description.json")

        if isinstance(cand, dict) and isinstance(jd_struct, dict):
            st.subheader("Match Scoring (baseline)")
            score = score_candidate(cand if cand else {}, jd_struct if jd_struct else {})
            st.json(score)
            st.download_button("Download score.json", data=json.dumps(score, indent=2), file_name="score.json")

st.caption(
    "Tip: If PDFs return little or no text, install Tesseract and pdf2image+pytesseract for OCR fallback.\n"
    "For .docx, install python-docx."
)

