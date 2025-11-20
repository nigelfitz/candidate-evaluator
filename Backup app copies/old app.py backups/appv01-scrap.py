# app.py
# Candidate Pack Summariser (Local) — Private-by-default
# Runs locally; files processed in memory; nothing stored unless user downloads.

import os, io, zipfile, json, re, time, threading
from typing import Dict, Any, List

import streamlit as st
import pandas as pd

# -------- OCR / parsing deps --------
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import fitz  # PyMuPDF
import docx

# Optional PDF parser (not required when using PyMuPDF)
try:
    from pypdf import PdfReader  # noqa
except Exception:
    PdfReader = None

# Point pytesseract to your install if needed
# Adjust this path if Tesseract is installed elsewhere on your machine
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# OpenAI SDK (v1+)
from openai import OpenAI

# -------------------- Config --------------------
MODEL_SCORE = "gpt-4o-mini"     # cheaper model used for scoring / JD parsing
MODEL_EXTRACT = "gpt-4o-mini"   # used for extraction
MAX_CVS = 10

# -------------------- Helpers --------------------
def read_file_to_text(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT — uses OCR fallback for scanned PDFs."""
    if uploaded_file is None:
        return ""

    name = (uploaded_file.name or "").lower()
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    text = ""

    # ---- TXT ----
    if name.endswith(".txt"):
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = file_bytes.decode("latin-1", errors="ignore")

    # ---- DOCX ----
    elif name.endswith(".docx"):
        d = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in d.paragraphs).strip()

    # ---- PDF ----
    elif name.endswith(".pdf"):
        extracted = ""
        try:
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            for page in pdf:
                extracted += page.get_text("text")
        except Exception:
            extracted = ""

        # If not enough text found, run OCR
        if len(extracted.strip()) < 100:
            try:
                images = convert_from_bytes(file_bytes)
                ocr_text = []
                for img in images:
                    ocr_text.append(pytesseract.image_to_string(img))
                extracted = "\n".join(ocr_text)
                print(f"OCR used for {uploaded_file.name}")
            except Exception as e:
                print(f"OCR failed: {e}")

        text = extracted

    return (text or "").strip()


def client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=key)


def call_json(model: str, system: str, user: str, schema_hint: Dict[str, Any]) -> Dict[str, Any]:
    """Ask model to return JSON; try to coerce if it wraps it in text."""
    resp = client().chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": user + "\n\nReturn ONLY valid JSON. Match this shape loosely:\n"
                           + json.dumps(schema_hint, indent=2),
            },
        ],
    )
    try:
        content = (resp.choices[0].message.content or "").strip()
    except Exception:
        # backward-safety for different SDK object shapes
        content = (resp.choices[0].message["content"] or "").strip()

    m = re.search(r"\{.*\}", content, re.S)
    if m:
        content = m.group(0)

    try:
        return json.loads(content)
    except Exception:
        return {"_raw": content}

# -------------------- Schemas --------------------
CAND_SCHEMA = {
    "name": "",
    "contact": {"email": "", "phone": "", "linkedin": ""},
    "years_experience": 0,
    "current_title": "",
    "current_company": "",
    "recent_roles": [{"title": "", "company": "", "from": "", "to": ""}],
    "skills_core": [],
    "skills_tools": [],
    "education_highest": {"degree": "", "field": "", "institution": "", "year": ""},
    "certifications": [],
    "location": "",
    "work_rights": "",
    "notice_period": "",
    "compensation": {"current": "", "expected": ""},
    "achievements": [],
    "red_flags": [],
    "missing_info": [],
}

JD_SCHEMA = {
    "role_title": "",
    "must_have_skills": [],
    "nice_to_have_skills": [],
    "years_required": {"min": 0, "preferred": 0},
    "domain": "",
    "location": "",
    "work_rights_required": "",
    "salary_range": "",
    "key_outcomes": [],
}

SCORE_SCHEMA = {
    "fit_score": 0,
    "reasons": [],
    "interview_questions": [],
    "followups": [],
}

# -------------------- Rule-based Extraction --------------------
def extract_candidate_rule_based(cv_text: str) -> Dict[str, Any]:
    """
    Deterministic extraction using regex/heuristics (no OpenAI).
    """
    text = (cv_text or "").strip()
    out = json.loads(json.dumps(CAND_SCHEMA))  # deep copy

    # helpers
    def find_first(pattern, flags=re.I | re.M):
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else ""

    def find_block(start_rx, stop_rx=r"^\s*[A-Z][A-Za-z ]{2,}\s*:?\s*$", max_lines=50):
        """Capture lines after a heading until next ALL-CAPS-ish heading or blank gap."""
        lines = text.splitlines()
        block, capture = [], False
        start = re.compile(start_rx, re.I)
        stop = re.compile(stop_rx, re.M)
        for ln in lines:
            if not capture and start.search(ln):
                capture = True
                continue
            if capture:
                if stop.match(ln) or len(block) >= max_lines:
                    break
                block.append(ln)
        return "\n".join(block).strip()

    # contact
    out["contact"]["email"] = find_first(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
    out["contact"]["phone"] = find_first(r"(\+?\d[\d\s\-()]{7,})")
    out["contact"]["linkedin"] = find_first(r"(https?://(?:www\.)?linkedin\.com/[^\s)]+)")

    # name
    name = find_first(r"(?:^|\n)\s*Name\s*:\s*(.+)")
    if not name:
        for ln in text.splitlines():
            t = ln.strip()
            if 4 <= len(t) <= 60 and ":" not in t and re.match(r"^[A-Za-z][A-Za-z .'-]+$", t):
                name = t
                break
    out["name"] = name

    # labelled fields
    out["location"] = find_first(r"(?:^|\n)\s*Location\s*:\s*(.+)")
    out["work_rights"] = find_first(r"(?:^|\n)\s*Work\s*Rights?\s*:\s*(.+)")
    out["notice_period"] = find_first(r"(?:^|\n)\s*Notice\s*Period\s*:\s*(.+)")

    # years of experience
    yrs = find_first(r"(\d+)\s*\+?\s*years?\b")
    out["years_experience"] = int(yrs) if yrs.isdigit() else 0

    # current title/company
    out["current_title"] = find_first(r"(?:^|\n)\s*Current\s+Title\s*:\s*(.+)")
    out["current_company"] = find_first(r"(?:^|\n)\s*Current\s+Company\s*:\s*(.+)")

    if not out["current_title"] or not out["current_company"]:
        top_chunk = "\n".join(text.splitlines()[:30])
        m = re.search(
            r"^\s*([A-Z][A-Za-z0-9 &/\-]+)\s+at\s+([A-Z][A-Za-z0-9 &/\-.,]+)\s*$",
            top_chunk,
            re.I | re.M,
        )
        if m:
            out["current_title"] = out["current_title"] or m.group(1).strip()
            out["current_company"] = out["current_company"] or m.group(2).strip()
        else:
            exp = find_block(r"^(Experience|Employment|Work History)\s*:?\s*$")
            m2 = re.search(
                r"^\s*([A-Z][A-Za-z0-9 &/\-]+)\s+[,@-]\s+([A-Z][A-Za-z0-9 &/\-.,]+)\s*$",
                exp,
                re.I | re.M,
            )
            if m2:
                out["current_title"] = out["current_title"] or m2.group(1).strip()
                out["current_company"] = out["current_company"] or m2.group(2).strip()

    # skills
    skills_line = find_first(r"(?:Experience|Years).*?in\s+([A-Za-z0-9 ,\-/+&\.]+)")
    skills = []
    if skills_line:
        skills += [t.strip() for t in re.split(r"[,\u2022;•/|]+", skills_line) if t.strip()]

    skills_block = find_block(r"^Skills\s*:?\s*$")
    if skills_block:
        for ln in skills_block.splitlines():
            for tok in re.split(r"[,\u2022;•/|]+", ln):
                if tok.strip():
                    skills.append(tok.strip())

    out["skills_core"] = sorted(set([s for s in skills if 1 < len(s) <= 40]))[:24]

    # education (light)
    deg = find_first(r"(Bachelor|Master|MSc|BSc|PhD|Diploma)[^,\n]*", flags=re.I)
    inst = find_first(r"(University|Institute|College)[^\n]*", flags=re.I)
    if not deg:
        deg = find_first(r"(Bachelor|Master|PhD)[^\n]*", flags=re.I)
    out["education_highest"]["degree"] = deg
    out["education_highest"]["institution"] = inst

    # location enrichment if still empty
    if not out["location"]:
        au_cities = r"(Brisbane|Sydney|Melbourne|Perth|Adelaide|Canberra|Hobart|Darwin)"
        au_states = r"(QLD|NSW|VIC|WA|SA|ACT|TAS|NT)"
        nz_cities = r"(Auckland|Wellington|Christchurch|Hamilton|Tauranga|Dunedin)"
        uk_cities = r"(London|Manchester|Birmingham|Leeds|Glasgow|Edinburgh|Bristol)"
        us_states = r"(AL|AK|AZ|AR|CA|CO|CT|DC|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VA|VT|WA|WI|WV|WY)"
        mloc = re.search(fr"{au_cities}\s*,?\s*{au_states}\b", text, re.I)
        if not mloc:
            mloc = re.search(fr"{nz_cities}\b", text, re.I)
        if not mloc:
            mloc = re.search(fr"{uk_cities}\b", text, re.I)
        if not mloc:
            mloc = re.search(fr"\b([A-Z][A-Za-z ]+),\s*{us_states}\b", text, re.I)
        if mloc:
            out["location"] = (mloc.group(0)).strip()

    # red flags / missing info
    if not (out["name"] or out["current_title"] or out["current_company"] or out["skills_core"]):
        out["red_flags"] = ["Very little structured content found"]

    if not out.get("contact", {}).get("email"):
        out["missing_info"].append("Email not found")
    if not out.get("contact", {}).get("phone"):
        out["missing_info"].append("Phone not found")

    return out


def extract_candidate(cv_text: str) -> Dict[str, Any]:
    """
    AI extractor (gpt-4o-mini) with JSON mode.
    Falls back to deterministic rule-based extraction if the AI returns
    empty/invalid content.
    """
    text = (cv_text or "").strip()

    # Short texts -> fallback immediately
    if len(text) < 40:
        st.info("CV text is very short; using rule-based extractor.")
        return extract_candidate_rule_based(text)

    system = (
        "You are an experienced recruitment analyst.\n"
        "Return ONLY valid JSON. Do NOT fabricate or infer missing fields.\n"
        "Strings must come from the CV text (exact or trivially normalized).\n"
        "If a field is not explicitly present, leave it empty and add a short note to 'missing_info'."
    )
    user = f"""
CV TEXT (triple backticks):


Extract ONLY what is explicitly present into this structure (do not add new keys):
{json.dumps(CAND_SCHEMA, indent=2)}
Rules:
- strings: use substrings from the CV (you may trim whitespace/punctuation)
- numbers: only if clearly stated (e.g., '10+ years'); otherwise 0
- lists: items must be explicitly mentioned in the CV
- If unknown, use "" or [] and add a note in 'missing_info'
Return valid JSON ONLY.
"""

    ai_ok = False
    ai_data: Dict[str, Any] = {}

    try:
        resp = client().chat.completions.create(
            model=MODEL_EXTRACT,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        try:
            content = resp.choices[0].message.content.strip()
        except Exception:
            content = resp.choices[0].message["content"].strip()

        parsed = json.loads(content) if content else {}
        ai_data = {**CAND_SCHEMA, **parsed}

        key_filled = any(
            [
                bool(ai_data.get("name")),
                bool(ai_data.get("current_title")),
                bool(ai_data.get("current_company")),
                bool(ai_data.get("skills_core")),
            ]
        )
        ai_ok = key_filled

    except Exception as e:
        st.warning(f"AI extraction error: {e}. Falling back to rule-based.")

    if ai_ok:
        return ai_data

    st.info("AI returned little/no structured content; using rule-based extractor.")
    return extract_candidate_rule_based(text)


def enforce_source_truth(candidate: Dict[str, Any], source_text: str) -> Dict[str, Any]:
    """Remove any value that doesn't literally appear in the CV text."""
    if not isinstance(candidate, dict):
        return candidate
    txt = (source_text or "").lower()

    def present(s: str) -> bool:
        return isinstance(s, str) and s.strip() and s.lower() in txt

    for k in ["name", "current_title", "current_company", "location", "work_rights", "notice_period"]:
        v = candidate.get(k, "")
        if isinstance(v, str) and v and not present(v):
            candidate.setdefault("missing_info", []).append(f"{k} not explicitly found in CV text")
            candidate[k] = ""

    for k in ["skills_core", "skills_tools", "achievements", "certifications"]:
        if isinstance(candidate.get(k), list):
            candidate[k] = [x for x in candidate[k] if present(x)]

    email = candidate.get("contact", {}).get("email") or ""
    phone = candidate.get("contact", {}).get("phone") or ""
    linkedin = candidate.get("contact", {}).get("linkedin") or ""

    if not email:
        m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", source_text or "")
        if m:
            candidate.setdefault("contact", {}).update({"email": m.group(0)})
    if not phone:
        m = re.search(r"(\+?\d[\d\s\-()]{7,})", source_text or "")
        if m:
            candidate.setdefault("contact", {}).update({"phone": m.group(0).strip()})
    if not linkedin and "linkedin.com" in (source_text or "").lower():
        m = re.search(r"(https?://)?([\w]+\.)?linkedin\.com/[^\s)]+", source_text or "", re.I)
        if m:
            candidate.setdefault("contact", {}).update({"linkedin": m.group(0)})

    return candidate

# -------------------- JD Parse & Scoring --------------------
def parse_jd(jd_text: str) -> Dict[str, Any]:
    system = "You are an expert tech recruiter. Parse the job description into structured fields."
    user = """JOB DESCRIPTION:

Extract role title, core skills, nice-to-haves, years, domain, location, work rights, salary, key outcomes."""
    return call_json(MODEL_SCORE, system, user, JD_SCHEMA)


def score_candidate(c: Dict[str, Any], jd: Dict[str, Any]) -> Dict[str, Any]:
    system = "You are a fair, explainable scoring engine for recruiting."
    user = f"""Score the candidate against the JD using this rubric:
- Skills match 40
- Experience (years, domain, seniority) 25
- Achievements/impact signals 20
- Logistics (location, work rights, notice, comp) 10
- Red flags penalty up to -15

Return JSON with: fit_score (0-100), reasons (3–6 bullets), interview_questions (3–5), followups (2–4).
CANDIDATE:
{json.dumps(c, ensure_ascii=False)}
JD:
{json.dumps(jd, ensure_ascii=False)}"""
    out = call_json(MODEL_SCORE, system, user, SCORE_SCHEMA)
    try:
        s = float(out.get("fit_score", 0))
        out["fit_score"] = max(0, min(100, s))
    except Exception:
        out["fit_score"] = 0
    return out


def row_from(c: Dict[str, Any], s: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": c.get("name", ""),
        "years_experience": c.get("years_experience", ""),
        "current_title": c.get("current_title", ""),
        "current_company": c.get("current_company", ""),
        "top_skills": ", ".join([*c.get("skills_core", [])][:6]),
        "education": (c.get("education_highest") or {}).get("degree", ""),
        "location": c.get("location", ""),
        "work_rights": c.get("work_rights", ""),
        "notice_period": c.get("notice_period", ""),
        "expected_comp": (c.get("compensation") or {}).get("expected", ""),
        "red_flags": "; ".join(c.get("red_flags", []))[:150],
        "fit_score": s.get("fit_score", 0),
    }


def one_pager_md(c: Dict[str, Any], s: Dict[str, Any], jd: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines += [f"# {c.get('name','(Unknown)')}", ""]
    lines += [
        f"**Current:** {c.get('current_title','')} @ {c.get('current_company','')}",
        f"**Experience:** {c.get('years_experience','?')} yrs | **Location:** {c.get('location','')}",
        f"**Fit score:** {s.get('fit_score',0)} / 100",
        "",
    ]
    if c.get("skills_core"):
        lines.append("**Core skills:** " + ", ".join(c["skills_core"][:12]))
    if c.get("achievements"):
        lines += ["**Notable achievements:**"] + [f"- {a}" for a in c["achievements"][:5]]
    lines.append("")
    lines.append("**Why this candidate fits**")
    lines += [f"- {r}" for r in s.get("reasons", [])[:6]]
    if s.get("interview_questions"):
        lines += ["", "**Suggested interview questions**"] + [f"- {q}" for q in s["interview_questions"]]
    if s.get("followups"):
        lines += ["", "**Follow-ups / Missing info**"] + [f"- {f}" for f in s["followups"]]
    if c.get("missing_info"):
        lines += [f"- {m}" for m in c["missing_info"][:5]]
    return "\n".join(lines)

# -------------------- UI --------------------
st.set_page_config(page_title="Candidate Pack Summariser (Local)", page_icon="📝", layout="wide")
st.title("📝 Candidate Pack Summariser (Local)")
st.caption("Runs locally. Files aren’t stored; processed in memory. Downloads are explicit.")

with st.sidebar:
    st.subheader("Upload")
    jd_file = st.file_uploader("Job description (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="jd")
    cv_files = st.file_uploader(
        "Candidate CVs (1–10)", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="cvs"
    )

    jd_manual = st.text_area(
        "Or paste JD text here (fallback)",
        height=200,
        help="Use this if your JD PDF/DOCX doesn't extract text.",
    )

    show_debug = st.checkbox("Show model debug (JSON)", value=False)
    enforce_truth = st.checkbox("Enforce 'from-source only' filter", value=False)

    st.divider()

    with st.expander("🩺 Diagnostics", expanded=False):
        has_key = bool(os.environ.get("OPENAI_API_KEY"))
        st.write(f"API key detected: {'YES' if has_key else 'NO'}")
        try:
            jd_len = len(read_file_to_text(jd_file)) if jd_file else 0
            st.write(f"JD text length (file): {jd_len}")
            if jd_manual:
                st.write(f"JD text length (manual): {len(jd_manual)}")
            cv_lens = []
            if cv_files:
                for f in cv_files:
                    try:
                        cv_lens.append((f.name, len(read_file_to_text(f))))
                    except Exception as e:
                        cv_lens.append((f.name, f"ERROR: {e}"))
            st.write({"cv_text_lengths": cv_lens})
        except Exception as e:
            st.write(f"Read error: {e}")

        def _test_openai():
            try:
                r = client().chat.completions.create(
                    model=MODEL_SCORE, messages=[{"role": "user", "content": "Reply with OK"}], temperature=0
                )
                return r.choices[0].message.content
            except Exception as e:
                return f"ERROR: {e}"

        if st.button("Run OpenAI test"):
            st.write(_test_openai())

    st.divider()
    with st.expander("🔒 Privacy & Data Security", expanded=False):
        st.markdown(
            """
**Candidate data is stored only on your device or your connected cloud account (e.g., Google Drive).**  
**No data is ever made public or visible to anyone else.**  
**We use OpenAI’s API for analysis — your data is _not used to train AI models_ and is automatically deleted after processing.**
            """
        )
    st.button("Connect Google Drive (optional) — coming soon", disabled=True)

col1, col2 = st.columns([1, 1])
run = col1.button("Analyse Candidates", type="primary")
clear = col2.button("Clear Session")

if clear:
    st.session_state.clear()
    st.rerun()

# -------- Robust run block (won't kill the server on errors) --------
if run:
    try:
        if not cv_files:
            st.error("Please upload at least one CV.")
        else:
            files = list(cv_files)[:MAX_CVS]

            # JD text: prefer file, fallback to manual
            jd_text = ""
            try:
                if jd_file:
                    jd_text = read_file_to_text(jd_file)
            except Exception as e:
                st.error(f"Could not read JD: {e}")

            if (not jd_text) or (len(jd_text.strip()) < 50):
                if jd_manual and len(jd_manual.strip()) >= 50:
                    jd_text = jd_manual.strip()
                else:
                    st.error(
                        "No usable Job Description text found. Please paste the JD text into the sidebar fallback box."
                    )
                    st.stop()

            with st.spinner("Parsing and scoring…"):
                jd = parse_jd(jd_text)

                rows, pages = [], {}
                for f in files:
                    # Read file text
                    try:
                        cv_text = read_file_to_text(f)
                        if show_debug:
                            st.write(f"**{f.name} — first 800 chars of extracted text:**")
                            st.code((cv_text or "")[:800] or "[no text extracted]")
                    except Exception as e:
                        st.error(f"Failed to read {f.name}: {e}")
                        continue

                    # Extract + score with per-file safeguards
                    try:
                        cand = extract_candidate(cv_text)

                        if enforce_truth:
                            cand = enforce_source_truth(cand, cv_text)

                        score = score_candidate(cand, jd)

                        if show_debug:
                            st.write(f"**{f.name} — extracted candidate data:**")
                            st.json(cand)
                            st.write("**Scoring output:**")
                            st.json(score)

                        rows.append({**row_from(cand, score), "source_file": f.name})
                        pages[f.name] = one_pager_md(cand, score, jd)
                        time.sleep(0.15)

                    except Exception as e:
                        st.error(f"Error while processing {f.name}: {e}")
                        st.exception(e)

            if not rows:
                st.warning("No candidates processed.")
            else:
                df = pd.DataFrame(rows).sort_values("fit_score", ascending=False)
                st.subheader("📊 Shortlist & Comparison")
                st.dataframe(df, height=360)

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv_bytes, file_name="shortlist.csv", mime="text/csv")

                bio = io.BytesIO()
                with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
                    for name, md in pages.items():
                        base = os.path.splitext(os.path.basename(name))[0]
                        z.writestr(f"{base}.md", md)
                st.download_button(
                    "Download candidate one-pagers (ZIP)", data=bio.getvalue(), file_name="candidate_one_pagers.zip", mime="application/zip"
                )

                st.success("Done. Files processed in memory. Nothing stored unless you download.")

    except Exception as e:
        st.error("Unexpected error during analysis. The server is still running.")
        st.exception(e)

st.markdown("---")
st.markdown(
    "**Privacy note:** Files are processed transiently in memory on your machine. "
    "No server database or disk saves. Only extracted text is analysed by the OpenAI API; "
    "API data is not used to train models."
)

