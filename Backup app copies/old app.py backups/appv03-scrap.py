import os
import re
import json
import tempfile
from typing import Dict, Any, Optional

import streamlit as st
import datetime as _dt

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


_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], 1)}
# also 3-letter aliases
_MONTHS.update({m[:3].lower(): _MONTHS[m.lower()] for m in list(_MONTHS) if len(m) > 3})

def norm_skill_set(values) -> set:
    out = set()
    if isinstance(values, list):
        for v in values:
            if not isinstance(v, str): continue
            # split on commas & slashes
            parts = re.split(r"[,/;•|]", v)
            for p in parts:
                s = re.sub(r"[^a-z0-9+.# ]+", " ", p.lower()).strip()
                s = re.sub(r"\s+", " ", s)
                if s: out.add(s)
    return out

_DATE_RE = re.compile(
    r"(?P<mon>[A-Za-z]{3,9})\s+(?P<year>\d{4})|(?P<year2>\d{4})", re.I)

def parse_date_phrase(s: str) -> Optional[_dt.date]:
    if not isinstance(s, str): return None
    s = s.strip()
    m = _DATE_RE.search(s)
    if not m: return None
    if m.group("year"):
        mon = _MONTHS.get(m.group("mon").lower(), 1)
        year = int(m.group("year"))
        return _dt.date(year, mon, 1)
    # year only
    if m.group("year2"):
        return _dt.date(int(m.group("year2")), 1, 1)
    return None

def estimate_duration_months(duration_field: str) -> Optional[int]:
    """Very rough parse e.g. 'July 2008 to April 2014'."""
    if not isinstance(duration_field, str): return None
    parts = re.split(r"\bto\b|\-|–|—|>", duration_field, flags=re.I)
    if len(parts) < 2: return None
    start = parse_date_phrase(parts[0])
    end = parse_date_phrase(parts[1]) or _dt.date.today()
    if start and end:
        return max(0, (end.year - start.year) * 12 + (end.month - start.month))
    return None



# -------------------------------
# Prompt builders + safe JSON extraction
# -------------------------------
def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Try to pull the first top-level JSON object from a string and parse it."""
    # direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # greedy brace match
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

    # quote-normalization fallback
    try:
        normalized = text.replace("'", '"')
        return json.loads(normalized)
    except Exception:
        return None

MAX_CHARS = 12000  # we’ll send at most this much of the document to the model

def build_user_prompt(label: str, text: str) -> str:
    """
    Wrap the document text with explicit BEGIN/END tags and length marker.
    This helps both debugging and makes it unambiguous for the model.
    """
    text = text or ""
    trimmed = text[:MAX_CHARS]
    return (
        f"BEGIN_{label}_TEXT len={len(text)} trimmed_to={len(trimmed)}\n"
        f"{trimmed}\nEND_{label}_TEXT"
    )

# -------------------------------
# GPT-powered extractors
# -------------------------------
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
# Scoring (simple baseline)
# -------------------------------

def norm_skill_set(values) -> set:
    """
    Normalize a list of skill strings into a deduped, lowercase set.
    Splits on common separators and removes punctuation noise.
    """
    out = set()
    if isinstance(values, list):
        for v in values:
            if not isinstance(v, str):
                continue
            # split on commas, slashes, semicolons, bullets, pipes
            parts = re.split(r"[,/;•|]", v)
            for p in parts:
                s = p.lower()
                # common canonicalizations
                s = s.replace("cash flow", "cashflow")
                s = s.replace("&", "and")
                # strip non-word-ish chars but keep + . # (useful for c# etc)
                s = re.sub(r"[^a-z0-9+.# ]+", " ", s)
                s = re.sub(r"\s+", " ", s).strip()
                if s:
                    out.add(s)
    return out

def _tokset(s: str) -> set:
    """Tokenize a normalized skill into word-ish tokens (captures 'budgeting', 'forecasting', 'c#', 'power', 'bi')."""
    return set(re.findall(r"[a-z0-9+.#]+", s.lower()))


def score_candidate(candidate: dict, jd: dict) -> dict:
    """
    Richer skill match:
      - Normalize skills (split & clean)
      - Exact overlap on normalized phrases
      - Token-level overlap to capture phrasing differences
    Final score = 70% exact phrase coverage + 30% token coverage.
    """
    cand_sk = norm_skill_set(candidate.get("skills", []))
    jd_sk   = norm_skill_set(jd.get("key_skills", []))

    # Exact phrase overlap
    exact_overlap = sorted(cand_sk & jd_sk)

    # Token-level overlap
    cand_tokens = set().union(*[_tokset(s) for s in cand_sk]) if cand_sk else set()
    jd_tokens   = set().union(*[_tokset(s) for s in jd_sk]) if jd_sk else set()
    token_overlap = sorted(jd_tokens & cand_tokens)

    # Scores
    denom_exact = max(1, len(jd_sk))
    exact_score = len(exact_overlap) / denom_exact

    denom_tok   = max(1, len(jd_tokens))
    token_score = len(token_overlap) / denom_tok

    final = round(100 * (0.7 * exact_score + 0.3 * token_score), 1)

    # Gaps = JD skills not exactly covered by normalized candidate phrases
    gaps = sorted([s for s in jd_sk if s not in cand_sk])

    return {
        "score": final,
        "exact_overlap": exact_overlap,
        "token_overlap_count": len(token_overlap),
        "jd_skill_count": len(jd_sk),
        "candidate_skill_count": len(cand_sk),
        "gaps": gaps,
        # optional raw fields if you want to inspect
        # "cand_skills_norm": sorted(cand_sk),
        # "jd_skills_norm": sorted(jd_sk),
        # "token_overlap": token_overlap,
    }


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Candidate Pack Summariser", layout="wide")
st.title("📄 Candidate Pack Summariser")

with st.sidebar:
    st.header("Settings")
    api_status = "loaded from environment/.env" if os.getenv("OPENAI_API_KEY") else "missing"
    if os.getenv("OPENAI_API_KEY"):
        st.success(f"OPENAI_API_KEY {api_status}")
    else:
        st.error("OPENAI_API_KEY is missing. Create .env with OPENAI_API_KEY=... and restart.")

    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.caption("Use gpt-4o for higher accuracy; gpt-4o-mini for speed.")

    st.divider()
    st.subheader("Diagnostics")
    diag_cv_len = st.empty()
    diag_jd_len = st.empty()

    st.divider()
    DEBUG_SHOW_PROMPTS = st.checkbox("Debug: show prompts sent to GPT", value=False)

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

# Prompt preview (to verify we actually send text to GPT)
if DEBUG_SHOW_PROMPTS:
    with st.expander("Prompt to GPT — CV (first 1,500 chars)"):
        st.text(build_user_prompt("CV", cv_text)[:1500])
    with st.expander("Prompt to GPT — JD (first 1,500 chars)"):
        st.text(build_user_prompt("JD", jd_text)[:1500])

if run:
    errs = False
    if not cv_text or len(cv_text.strip()) < 50:
        st.error("No readable CV text found. Try another file or enable OCR dependencies."); errs = True
    if not jd_text or len(jd_text.strip()) < 50:
        st.error("No readable JD text found. Try another file or enable OCR dependencies."); errs = True

    if not errs:
        with st.spinner("Analysing with GPT…"):
            cand_res = extract_candidate(cv_text, model=model)   # {'data','raw','error'}
            jd_res   = parse_jd(jd_text, model=model)

        c1, c2 = st.columns(2)

        # ---------- Candidate ----------
        with c1:
            st.subheader("Candidate (extracted)")
            if cand_res["error"]:
                st.error(cand_res["error"])
            if cand_res["data"] is not None:
                st.json(cand_res["data"])
                st.download_button("Download candidate.json",
                                   data=json.dumps(cand_res["data"], indent=2),
                                   file_name="candidate.json")
            else:
                st.warning("Model did not return valid JSON. See raw output below.")
            with st.expander("Raw model output — Candidate"):
                st.code(cand_res["raw"] or "(empty)", language="json")

        # ---------- JD ----------
        with c2:
            st.subheader("Job Description (extracted)")
            if jd_res["error"]:
                st.error(jd_res["error"])
            if jd_res["data"] is not None:
                st.json(jd_res["data"])
                st.download_button("Download job_description.json",
                                   data=json.dumps(jd_res["data"], indent=2),
                                   file_name="job_description.json")
            else:
                st.warning("Model did not return valid JSON. See raw output below.")
            with st.expander("Raw model output — JD"):
                st.code(jd_res["raw"] or "(empty)", language="json")

        # ---------- Friendly summaries ----------
        cand_data = cand_res["data"] or {}
        jd_data   = jd_res["data"] or {}

        # A concise, human-friendly card
        with st.expander("Candidate — summary (card)"):
            name  = cand_data.get("name", "")
            email = cand_data.get("email", "")
            phone = cand_data.get("phone", "")
            st.markdown(f"**{name or '—'}**  \n{email or '—'} | {phone or '—'}")

            summary = cand_data.get("summary", "")
            if isinstance(summary, str) and summary.strip():
                st.write(summary.strip())

            skills = cand_data.get("skills", [])
            if isinstance(skills, list) and skills:
                top_skills = [s for s in skills if isinstance(s, str)][:10]
                if top_skills:
                    st.markdown("**Top skills:** " + ", ".join(top_skills))

            exp = cand_data.get("experience", [])
            if isinstance(exp, list) and exp:
                st.markdown("**Experience (first 2):**")
                for role in exp[:2]:
                    if isinstance(role, dict):
                        pos = role.get("position", "")
                        comp = role.get("company", "")
                        dur = role.get("duration", "")
                        line = " - " + " — ".join([x for x in [pos, comp] if x]) + (f" ({dur})" if dur else "")
                        st.write(line)

        # ---------- Match Scoring ----------
        st.subheader("Match Scoring (baseline)")
        try:
            score = score_candidate(cand_data, jd_data)
            st.json(score)
            st.download_button("Download score.json",
                               data=json.dumps(score, indent=2),
                               file_name="score.json")
        except Exception as e:
            st.error(f"Scoring error: {e}")
            score = None

        # A compact view of overlaps and gaps using existing score fields
        if isinstance(score, dict):
            with st.expander("Match — quick view"):
                st.write(f"**Score:** {score.get('score', 0)} / 100")
                if score.get("exact_overlap"):
                    st.markdown("**Exact overlaps:** " + ", ".join(score["exact_overlap"]))
                st.markdown(f"**Token overlap (unique tokens):** {score.get('token_overlap_count', 0)}")
                gaps = score.get("gaps", [])
                st.markdown("**Gaps vs JD:** " + (", ".join(gaps) if gaps else "—"))


        # Original quick tables (kept for reference/inspection)
        with st.expander("Candidate — quick view"):
            name = cand_data.get("name","")
            email = cand_data.get("email","")
            phone = cand_data.get("phone","")
            st.markdown(f"**Name:** {name}  \n**Email:** {email}  \n**Phone:** {phone}")
            if isinstance(cand_data.get("skills"), list):
                st.markdown("**Skills:** " + ", ".join(cand_data.get("skills")[:25]))
            if isinstance(cand_data.get("education"), list) and cand_data["education"]:
                st.markdown("**Education:**")
                for edu in cand_data["education"][:5]:
                    st.write("- ", edu)
            if isinstance(cand_data.get("experience"), list) and cand_data["experience"]:
                st.markdown("**Experience (first 3):**")
                for exp in cand_data["experience"][:3]:
                    st.write("- ", exp)

        with st.expander("Job Description — quick view"):
            st.markdown(f"**Role:** {jd_data.get('role_title','')}  \n"
                        f"**Location:** {jd_data.get('location','')}  \n"
                        f"**Type:** {jd_data.get('employment_type','')}")
            if isinstance(jd_data.get("key_skills"), list):
                st.markdown("**Key skills:** " + ", ".join(jd_data.get("key_skills")[:25]))

st.caption(
    "Tip: If PDFs return little or no text, install Tesseract and pdf2image+pytesseract for OCR fallback.\n"
    "For .docx, install python-docx."
)
