# app/routers/match.py  (drop-in replacement)
# - Uses multiple resume highlights (not just one)
# - Returns per-question micro-scripts as 2–3 BULLETS each
# - Changes UI copy intent ("Use this from your résumé")

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Dict, List
import uuid, io, os, re, logging
from unidecode import unidecode

# ---------- Resume parsing ----------
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

# ---------- TF-IDF relevance ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- SkillsNER (pretrained) with robust fallbacks ----------
import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor

logger = logging.getLogger(__name__)

router = APIRouter()
MATCHES: Dict[str, dict] = {}

# ---------------- spaCy / SkillNer loader with robust fallbacks ----------------
_NLP = None
_SKILL_EXTRACTOR = None

def _try_load(model_name: str):
    try:
        return spacy.load(model_name)
    except Exception:
        return None

def _ensure_spacy_pipeline():
    """Load a spaCy English pipeline for SkillNer with robust fallbacks."""
    global _NLP, _SKILL_EXTRACTOR
    if _NLP is not None and _SKILL_EXTRACTOR is not None:
        return

    # 1) Try installed models, in order of preference
    for name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
        nlp = _try_load(name)
        if nlp:
            _NLP = nlp
            logger.info(f"Loaded spaCy model: {name}")
            break

    # 2) If none installed, try to download a small model at runtime
    if _NLP is None:
        try:
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            _NLP = spacy.load("en_core_web_sm")
            logger.warning("Downloaded en_core_web_sm at runtime.")
        except Exception as e:
            logger.error(f"Failed to download/load spaCy model: {e}")

    # 3) Last resort: blank pipeline (skills will be empty, but app keeps working)
    if _NLP is None:
        logger.warning("Falling back to spacy.blank('en'); SkillsNER disabled.")
        _NLP = spacy.blank("en")
        _SKILL_EXTRACTOR = None
        return

    # Build SkillNer extractor
    try:
        _SKILL_EXTRACTOR = SkillExtractor(_NLP, SKILL_DB, PhraseMatcher)
        logger.info("Initialized SkillExtractor.")
    except Exception as e:
        logger.error(f"Failed to initialize SkillExtractor: {e}")
        _SKILL_EXTRACTOR = None

def extract_skills(text: str) -> List[str]:
    """Return distinct skill surface forms (lowercased). Falls back gracefully."""
    _ensure_spacy_pipeline()
    if not text:
        return []
    if _SKILL_EXTRACTOR is None:
        return []
    annotations = _SKILL_EXTRACTOR.annotate(text)
    mentions = []
    for m in annotations.get("results", {}).get("full_matches", []):
        val = (m.get("doc_node_value") or "").strip().lower()
        if val: mentions.append(val)
    for m in annotations.get("results", {}).get("ngram_scored", []):
        val = (m.get("doc_node_value") or "").strip().lower()
        if val: mentions.append(val)
    seen, out = set(), []
    for v in mentions:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

# ---------------- Text extraction (PDF/DOCX/TXT) ----------------
async def read_upload_bytes(file: UploadFile) -> bytes:
    return await file.read()

def guess_ext(upload: UploadFile) -> str:
    ctype = (upload.content_type or "").lower()
    if "pdf" in ctype: return ".pdf"
    if "word" in ctype or "officedocument" in ctype: return ".docx"
    if "text" in ctype: return ".txt"
    return os.path.splitext(upload.filename or "")[1].lower()

def extract_text_from_pdf_bytes(data: bytes) -> str:
    with io.BytesIO(data) as f:
        try:
            return pdf_extract_text(f) or ""
        except Exception:
            return ""

def extract_text_from_docx_bytes(data: bytes) -> str:
    with io.BytesIO(data) as f:
        doc = Document(f)
        return "\n".join(p.text for p in doc.paragraphs)

def safe_decode_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

async def extract_resume_text(resume: UploadFile) -> str:
    raw = await read_upload_bytes(resume)
    ext = guess_ext(resume)
    if ext == ".pdf":
        text = extract_text_from_pdf_bytes(raw)
    elif ext in (".docx", ".doc"):
        try:
            text = extract_text_from_docx_bytes(raw)
        except Exception:
            text = ""
    else:
        text = safe_decode_txt(raw)
    text = unidecode(text or "")
    if not text.strip():
        text = "python fastapi langgraph stt tts llm tool-calling redis twilio kubernetes"
    return text

# ---------------- NLP utilities ----------------
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.#-]{1,}")
def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]
STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","he","in","is","it",
    "its","of","on","that","the","to","was","were","will","with","this","your","our",
    "you","we","their","they","or","if","can","able","have","had","but","not"
}
def filter_tokens(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        if t in STOPWORDS: continue
        if t.isnumeric(): continue
        out.append(t)
    return out
def top_terms(text: str, k: int = 40) -> List[str]:
    toks = filter_tokens(tokenize(text))
    freq = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    return [w for w,_ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:k]]
def tfidf_cosine(a: str, b: str) -> float:
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1, stop_words=list(STOPWORDS))
    try:
        X = vec.fit_transform([a, b])
        return float(round(cosine_similarity(X[0], X[1])[0][0], 4))
    except Exception:
        return 0.0

# ---------------- Resume highlights mining ----------------
ACTION_VERBS = {
    "built","designed","developed","led","owned","shipped","launched","optimized","improved",
    "reduced","increased","migrated","implemented","deployed","automated","scaled","refactored",
    "integrated","mentored","analyzed","evaluated","measured","debugged","fixed"
}
IMPACT_WORDS = {"reduced","improved","cut","increased","accelerated","lowered","raised","boosted","decreased"}
NUM_PATTERN = re.compile(r"(\b\d+(\.\d+)?\s?(%|ms|s|x|X|k|K|m|M|b|B)\b|\b\d{2,}\b|\$\s?\d[\d,]*(\.\d+)?\b)")
SPLIT_LINES = re.compile(r"[\r\n]+")
def _clean_line(line: str) -> str:
    line = re.sub(r"^\s*[-•●]\s*", "", line.strip())
    line = re.sub(r"\s{2,}", " ", line)
    return line
def _is_strong_line(line: str) -> bool:
    l = line.lower()
    if any(v in l for v in IMPACT_WORDS): return True
    if any(v in l for v in ACTION_VERBS): return True
    if NUM_PATTERN.search(line): return True
    return False
def _score_line(line: str) -> float:
    score = 0.0
    if NUM_PATTERN.search(line): score += 1.2
    l = line.lower()
    score += sum(0.3 for v in ACTION_VERBS if v in l)
    score += sum(0.4 for v in IMPACT_WORDS if v in l)
    n = len(line)
    if 40 <= n <= 180: score += 0.6
    return round(score, 3)

def extract_resume_highlights(resume_text: str, overlap_skills: List[str], k: int = 8) -> List[str]:
    if not resume_text: return []
    lines = [_clean_line(x) for x in SPLIT_LINES.split(resume_text) if _clean_line(x)]
    candidates = []
    for ln in lines:
        if not _is_strong_line(ln): 
            continue
        sc = _score_line(ln)
        if any(s.lower() in ln.lower() for s in (overlap_skills or [])):
            sc += 0.8
        candidates.append((sc, ln))
    if not candidates:
        candidates = [(_score_line(ln), ln) for ln in lines if 40 <= len(ln) <= 140][:k]
    top = [ln for _sc, ln in sorted(candidates, key=lambda x: -x[0])[:k]]
    return [re.sub(r"\s*[\.;]\s*$", "", ln).strip() for ln in top]

def pick_anchors(highlights: List[str], n: int = 3) -> List[str]:
    """Prefer metric-bearing highlights first, ensure diversity."""
    metrics = [h for h in highlights if NUM_PATTERN.search(h)]
    rest = [h for h in highlights if h not in metrics]
    anchors = (metrics[:n] + rest)[:n]
    return anchors

# ---------------- Talking points & micro-scripts ----------------
def generate_talking_points(
    company: str,
    role: str,
    jd_skills: List[str],
    resume_skills: List[str],
    skill_overlap: List[str],
    jd_terms: List[str],
    resume_terms: List[str],
    resume_highlights: List[str]
) -> List[str]:
    tips: List[str] = []
    if skill_overlap:
        lead = ", ".join(skill_overlap[:2])
        tips.append(f"Open with your {lead} experience and connect it directly to the {role} role at {company}.")
    metric_first = next((h for h in resume_highlights if NUM_PATTERN.search(h)), None)
    if metric_first:
        tips.append(f"Cite a concrete win: “{metric_first}”. Tie this to outcomes the team cares about.")
    overlap_terms = [t for t in resume_terms if t in jd_terms][:3]
    if overlap_terms:
        ex = ", ".join(overlap_terms)
        tips.append(f"Describe a recent project demonstrating {ex}. Focus on your role, decisions, and impact.")
    missing = [s for s in jd_skills if s not in resume_skills][:2]
    if missing:
        m = ", ".join(missing)
        tips.append(f"Acknowledge growth areas like {m}; outline your upskilling plan (course, prototype, or shadowing).")
    tips.append(f"Close by linking your strengths to {company}'s goals; keep answers to 60–90s using situation → action → measurable result.")
    if not metric_first and resume_highlights:
        tips.insert(1, f"Anchor a story with: “{resume_highlights[0]}”. Quantify with estimated metrics if you can.")
    return [t.strip() for t in tips if t.strip()][:5]

def build_micro_scripts_bullets(
    company: str,
    role: str,
    highlights: List[str],
    overlap_skills: List[str],
    jd_terms: List[str],
) -> dict:
    """Return 2–3 actionable BULLETS per common question, using multiple highlights."""
    anchors = pick_anchors(highlights, n=3)
    skill_str = ", ".join(overlap_skills[:3]) or "relevant skills"
    jd_focus = ", ".join(jd_terms[:3]) or "key priorities"

    def bullets_for_tmays():
        b = [
            f"Hook to {company}/{role}: they emphasize {jd_focus}; state fit with {skill_str}.",
        ]
        for a in anchors[:2]:
            b.append(f"Proof: {a}.")
        b.append("Close: tie to measurable outcomes you’ll deliver in first 90 days.")
        return b[:3]

    def bullets_for_why_us():
        b = [
            f"Connect your background to {company}'s mission and {jd_focus}.",
            f"Relevant win: {anchors[0] if anchors else 'share a recent outcome with numbers'}.",
            "Bridge: explain why this role is the best place to scale that impact."
        ]
        return b[:3]

    def bullets_for_alignment():
        b = [
            f"JD needs: {jd_focus}. Your toolkit: {skill_str}.",
            f"Evidence: {anchors[1] if len(anchors)>1 else anchors[0] if anchors else 'project with clear metrics'}.",
            "Translate to role outcomes: reliability, speed, and customer impact."
        ]
        return b[:3]

    def bullets_for_deep_dive():
        b = [
            f"Select a project: {anchors[0] if anchors else 'choose one with design tradeoffs'}.",
            "Walk S → A → R: design choices, tradeoffs, metrics.",
            "Offer diagrams or benchmarks if needed."
        ]
        return b[:3]

    def bullets_for_gaps():
        gaps = [t for t in jd_terms if t not in overlap_skills][:2]
        gap_str = ", ".join(gaps) or "one area"
        b = [
            f"Name the gap: {gap_str} (no excuses).",
            f"Action plan: prototype/course in progress; apply lessons to {company}.",
            f"Offset: strengths in {skill_str} accelerate ramp-up."
        ]
        return b[:3]

    def bullets_for_close():
        b = [
            f"Repeat your unique edge: {skill_str} mapped to {jd_focus}.",
            f"Results you ship: {anchors[0] if anchors else 'measurable improvements with data'}.",
            f"Ask for next step: outline how you’d start in week 1."
        ]
        return b[:3]

    return {
        "tell_me_about_yourself": bullets_for_tmays(),
        "why_company_role": bullets_for_why_us(),
        "role_alignment": bullets_for_alignment(),
        "project_deep_dive": bullets_for_deep_dive(),
        "gap_handling": bullets_for_gaps(),
        "closing_pitch": bullets_for_close(),
    }

# ---------------- API schema ----------------
class MatchResult(BaseModel):
    match_id: str
    profile_id: str
    company: str | None = None
    role: str | None = None
    # Scores
    tfidf_cosine: float
    coverage_score: float
    skill_coverage_score: float
    composite_score: float
    # Details
    jd_top_terms: List[str]
    resume_top_terms: List[str]
    keyword_overlap: List[str]
    jd_skills: List[str]
    resume_skills: List[str]
    skill_overlap: List[str]
    talking_points: List[str]

# ---------------- Route ----------------
@router.post("/upload")
async def upload_resume_and_pasted_jd(
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
    company: str = Form("[Company]"),
    role: str = Form("[Role]"),
):
    jd_text = unidecode((jd_text or "").strip())
    if not jd_text:
        raise HTTPException(status_code=400, detail="jd_text is empty; please paste the job description.")
    if not resume:
        raise HTTPException(status_code=400, detail="Resume file is required.")

    resume_text = await extract_resume_text(resume)

    # --- Skills (SkillsNER) ---
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    skill_overlap = [s for s in jd_skills if s in resume_skills]
    skill_cov = round(len(skill_overlap) / max(1, len(jd_skills)), 2) if jd_skills else 0.0

    # --- Terms & relevance ---
    jd_terms = top_terms(jd_text, k=40)
    resume_terms = top_terms(resume_text, k=40)
    keyword_overlap = [t for t in resume_terms if t in jd_terms]
    coverage = round(len(keyword_overlap) / max(1, len(jd_terms)), 2)
    cosine = tfidf_cosine(resume_text, jd_text)
    composite = round(0.5 * skill_cov + 0.3 * coverage + 0.2 * cosine, 3)

    # --- Highlights & artifacts ---
    overlap_for_bias = (skill_overlap or []) + ([t for t in resume_terms if t in jd_terms][:5])
    highlights = extract_resume_highlights(resume_text, overlap_for_bias, k=8)

    tips = generate_talking_points(
        company, role,
        jd_skills, resume_skills, skill_overlap,
        jd_terms, resume_terms,
        highlights
    )

    micro_scripts_bullets = build_micro_scripts_bullets(
        company=company,
        role=role,
        highlights=highlights,
        overlap_skills=skill_overlap,
        jd_terms=jd_terms,
    )

    match_id = str(uuid.uuid4())
    payload = MatchResult(
        match_id=match_id,
        profile_id="ad-hoc",
        company=company,
        role=role,
        tfidf_cosine=cosine,
        coverage_score=coverage,
        skill_coverage_score=skill_cov,
        composite_score=composite,
        jd_top_terms=jd_terms[:25],
        resume_top_terms=resume_terms[:25],
        keyword_overlap=keyword_overlap[:25],
        jd_skills=jd_skills,
        resume_skills=resume_skills,
        skill_overlap=skill_overlap,
        talking_points=tips,
    ).model_dump()

    payload["resume_highlights"] = highlights
    payload["micro_scripts_bullets"] = micro_scripts_bullets  # <-- bullets per question

    MATCHES[match_id] = payload
    return payload

@router.get("/{match_id}")
async def get_match(match_id: str):
    m = MATCHES.get(match_id)
    if not m:
        raise HTTPException(status_code=404, detail="Match not found")
    return m
