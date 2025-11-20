# Candidate Pack Summariser — pt3 (Legacy JD Extractor Stable Build, quoting sanitized)
from __future__ import annotations

import io, os, re, json, hashlib, textwrap
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

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

def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

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

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

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

def compute_max_similarity_to_chunks(query_texts, chunk_texts, chunk_embs, info, vec_or_meta):
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

@st.cache_data(show_spinner=False)
def analyze_candidates(
    candidates: List[Candidate],
    criteria: List[str],
    weights: Optional[List[float]] = None,
    chunk_chars: int = 1200,
    overlap: int = 150
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

    progress = st.progress(0.0, text="Scoring…")
    total = max(1, len(candidates))

    for i, cand in enumerate(candidates):
        rows = cand_to_rows[i]
        cand_chunks = [all_chunk_texts[r] for r in rows]
        if len(cand_chunks) == 0:
            crit_scores = np.zeros(len(criteria)); argmax = np.zeros(len(criteria), dtype=int)
        else:
            cand_chunk_embs = chunk_embs[rows] if not isinstance(chunk_embs, np.ndarray) else chunk_embs[rows, :]
            crit_scores, argmax = compute_max_similarity_to_chunks(criteria, cand_chunks, cand_chunk_embs, info, vec_or_meta)

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

st.set_page_config(page_title="Candidate Pack Summariser", layout="wide")
st.title("📦 Candidate Pack Summariser")
st.caption("Legacy JD extractor • Criteria rows × Candidates columns • Local scoring with progress bar.")

with st.sidebar:
    st.header("Uploads")
    st.markdown("**Job Description**")
    jd_file = st.file_uploader("Upload JD (PDF/DOCX/TXT)", type=["pdf","docx","txt"], key="jd_upl", accept_multiple_files=False)
    st.markdown("**Candidate Resumes**")
    uploaded = st.file_uploader("Upload candidate resumes", accept_multiple_files=True, type=["pdf","docx","txt"], key="cand_upl")
    with st.expander("Or paste JD text manually"):
        jd_manual = st.text_area("JD text", value=st.session_state.get("jd_manual_text",""), height=180)
        st.session_state.jd_manual_text = jd_manual

    if jd_file is not None:
        b = jd_file.read(); jd_text = read_file_bytes(b, jd_file.name)
        st.session_state.jd = JD(file_name=jd_file.name, text=jd_text, hash=hash_bytes(b))
    elif jd_manual.strip():
        st.session_state.jd = JD(file_name="Manual JD", text=jd_manual, hash=hash_text(jd_manual))

    st.markdown("---")
    colb1, colb2, colb3 = st.columns(3)
    if colb1.button("Analyze / Update", type="primary"):
        st.session_state._trigger_analyze = True
        st.rerun()
    if colb2.button("Reset session"):
        for k in ["cached_candidates","last_coverage","last_insights","last_snippets","jd","jd_manual_text","criteria_text","weights_csv","evidence_map","cat_map"]:
            v = st.session_state.get(k)
            st.session_state[k] = [] if isinstance(v, list) else (pd.DataFrame() if isinstance(v, pd.DataFrame) else ({} if isinstance(v, dict) else ""))
        st.rerun()
    if colb3.button("Clear global caches"):
        st.cache_data.clear(); st.cache_resource.clear(); st.success("Cleared Streamlit caches.")

new_candidates: List[Candidate] = []
if uploaded:
    st.info("Scroll down in the main panel to label candidates if needed.")
    for uf in uploaded:
        b = uf.read(); text = read_file_bytes(b, uf.name)
        name_guess = infer_candidate_name(uf.name, text)
        new_candidates.append(Candidate(name=name_guess, file_name=uf.name, text=text, hash=hash_bytes(b)))
if new_candidates:
    existing = {c.hash: c for c in st.session_state.get("cached_candidates", [])}
    for c in new_candidates: existing[c.hash] = c
    st.session_state.cached_candidates = list(existing.values())

overview_tab, jd_tab, coverage_tab, insights_tab, compare_tab, export_tab, settings_tab = st.tabs(["Overview","Job Description","Coverage","Insights","Compare","Export","Settings"])

with overview_tab:
    st.subheader("Overview")
    left, right = st.columns([1,2])
    with left:
        st.metric("Candidates loaded", len(st.session_state.get("cached_candidates",[])))
        covdf = st.session_state.get("last_coverage", pd.DataFrame())
        st.metric("Criteria", len([c for c in covdf.columns if c not in ('Candidate','Overall')]) if isinstance(covdf, pd.DataFrame) and not covdf.empty else 0)
        if isinstance(covdf, pd.DataFrame) and not covdf.empty:
            st.metric("Top overall score", f"{covdf['Overall'].max():.2f}")
    with right:
        st.info("**How to read scores**\n\n• Scores are cosine similarities (0–1).\n• 'Overall' is the weighted average across your criteria.\n• Increase chunk size for speed; decrease for finer matching.")

    if st.session_state.get("cached_candidates"):
        st.markdown("### Candidate Labels")
        for c in st.session_state.cached_candidates:
            c.name = st.text_input(f"Label for {c.file_name}", value=c.name, key=f"nm_{c.hash[:8]}")

with jd_tab:
    st.subheader("Job Description")
    if st.session_state.get("jd"):
        jd_obj = st.session_state.jd
        st.write(f"**{jd_obj.file_name}** — {len(jd_obj.text):,} characters")
        with st.expander("Preview JD text"):
            st.text_area("JD content", value=jd_obj.text, height=300)
    else:
        st.warning("Upload a JD file or paste JD text in the sidebar.")

    if st.session_state.get("jd"):
        st.markdown("### Extract criteria from JD (Legacy)")
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
            with st.expander(f"{sec.replace("_"," ").title()} ({len(bullets)})"):
                opts = [clean_for_display_legacy(b) for b in bullets]
                sel = st.multiselect("Select items", options=opts, default=opts[:min(6, len(opts))], key=f"legacy_{sec}")
                picks[sec] = sel

        colj1, colj2 = st.columns(2)
        with colj1:
            per_sec = st.number_input("Max per section", 1, 20, value=6)
        with colj2:
            cap_total = st.number_input("Total cap", 5, 80, value=30)

        if st.button("Build Criteria"):
            any_sel = any(picks.get(sec) for sec in picks)
            if any_sel:
                crits, cat_map = [], {}
                for sec, lst in picks.items():
                    for x in lst[:per_sec]:
                        if x and x not in crits:
                            crits.append(x); cat_map[x] = sec
                        if len(crits) >= cap_total: break
                    if len(crits) >= cap_total: break
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
    st.subheader("Candidate Insights (local heuristics)")
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

with settings_tab:
    st.subheader("Settings")
    st.markdown("### Weights")
    st.session_state["weights_mode"] = st.radio("Weights", ["Uniform","Custom"],
        index=(0 if st.session_state.get("weights_mode","Uniform") == "Uniform" else 1),
        help="Use uniform weights or specify custom weights per criterion."
    )
    if st.session_state["weights_mode"] == "Custom":
        st.caption("Provide either 'criterion, weight' per line or one weight per line (aligned with criteria order).")
        default_weights = "Python, 2\nData analysis, 1\nStakeholder communication, 1"
        st.session_state["weights_csv"] = st.text_area("Custom weights", value=st.session_state.get("weights_csv", default_weights), height=120)

    st.markdown("---")
    st.markdown("### Performance")
    st.session_state["chunk_chars"] = st.slider("Chunk size (characters)", 600, 2000, st.session_state.get("chunk_chars",1200), step=100)
    st.session_state["overlap"] = st.slider("Chunk overlap", 50, 400, st.session_state.get("overlap",150), step=25)

if st.session_state.get("_trigger_analyze", False):
    crit_text = st.session_state.get("criteria_text", "").strip()
    if not crit_text and st.session_state.get("jd") is not None:
        secs = parse_jd_legacy(st.session_state.jd.text)
        crits, cat_map = build_criteria_legacy(secs, per_section=6, cap_total=30)
        st.session_state.criteria_text = "\n".join(crits)
        st.session_state.cat_map = cat_map
        crit_text = st.session_state.criteria_text

    criteria = parse_criteria_text(crit_text)
    weights = get_weights(criteria, st.session_state.get("weights_mode","Uniform"), st.session_state.get("weights_csv",""))

    if not criteria:
        st.warning("No criteria to analyze. Build criteria (Job Description tab) then click Analyze.")
    elif not st.session_state.get("cached_candidates"):
        st.warning("Please upload candidate resumes before running Analyze.")
    else:
        with st.spinner("Running analysis…"):
            cov, ins_local, snips, ev_map = analyze_candidates(
                st.session_state.cached_candidates, criteria, weights,
                chunk_chars=st.session_state.get("chunk_chars",1200),
                overlap=st.session_state.get("overlap",150)
            )
            st.session_state.last_coverage = cov
            st.session_state.last_insights = ins_local
            st.session_state.last_snippets = snips
            st.session_state.evidence_map = ev_map
        st.session_state._trigger_analyze = False
        if isinstance(st.session_state.last_coverage, pd.DataFrame) and not st.session_state.last_coverage.empty:
            try: st.toast("Analysis complete.", icon="✅")
            except Exception: pass
        else:
            st.warning("Analysis ran but produced no rows. Check criteria and document text.")

st.caption("Tip: Global cache clear is in the sidebar. Session reset clears only this run's data.")
