# app/routers/live.py (DROP-IN REPLACEMENT)
# Live Prep v3.1: clearer API fields (feedback_speech separated), rubric scoring, resume-aware nudges

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import time, uuid, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.routers.match import MATCHES  # match_id context (jd terms, bullets, skills, etc.)

router = APIRouter()
SESSIONS: Dict[str, dict] = {}

# ---------------- Utilities ----------------
FILLERS = {
    "um","uh","like","you know","sort of","kind of","basically","actually","literally",
    "right","so yeah","i mean"
}
HEDGES = {"maybe","i think","i guess","probably","might","perhaps","i feel","somewhat"}
VAGUE = {"synergy","leverage","innovative","cutting-edge","state-of-the-art","disrupt","holistic","optimize"}
ACTION_VERBS = {
    "built","designed","developed","led","owned","shipped","launched","optimized","improved",
    "reduced","increased","migrated","implemented","deployed","automated","scaled","refactored",
    "integrated","mentored","analyzed","evaluated","measured","debugged","fixed","architected"
}
SAR_CUES = {
    "situation","context","problem","challenge","goal",
    "action","approach","decision","designed","implemented","built",
    "result","impact","outcome","metric","learned"
}
NUM_RE = re.compile(r"\b\d+(\.\d+)?\s?(%|ms|s|x|k|m|b)?\b|\$\s?\d[\d,]*", re.I)

def _cosine(a: str, b: str) -> float:
    if not a or not b: return 0.0
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    X = vec.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0][0])

def _count_set(text: str, vocab: set) -> int:
    if not text: return 0
    low = " " + text.lower() + " "
    return sum(1 for w in vocab if f" {w} " in low)

def _has_metric(text: str) -> bool:
    return bool(NUM_RE.search(text or ""))

def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.!\?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]

def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+\-#\.]*", text or "")

def _type_token_ratio(tokens: List[str]) -> float:
    if not tokens: return 0.0
    return len(set(t.lower() for t in tokens)) / max(1, len(tokens))

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _to10(x: float) -> float:
    return round(10.0 * _clip01(x), 1)

def _coverage_score(answer: str, ctx: dict) -> float:
    ref_terms = list(ctx.get("jd_top_terms", []))
    skills = list(ctx.get("skill_overlap", []) or ctx.get("resume_skills", []))
    bullets = []
    msb = ctx.get("micro_scripts_bullets", {})
    if ctx.get("current_key") and ctx["current_key"] in msb:
        bullets = msb[ctx["current_key"]]
    ref_blob = " ".join(ref_terms + skills + bullets)
    if not ref_blob: return 0.0
    return _clip01(_cosine(answer, ref_blob))

def _structure_score(answer: str) -> float:
    sents = _sentences(answer)
    words = len(_tokens(answer))
    cues = _count_set(answer, SAR_CUES)
    sent_ok = 3 <= len(sents) <= 8
    word_ok = 70 <= words <= 180
    cue_norm = min(1.0, cues/3.0)
    base = 0.45*cue_norm + 0.3*(1.0 if sent_ok else 0.0) + 0.25*(1.0 if word_ok else 0.0)
    return _clip01(base)

def _specificity_score(answer: str, ctx: dict) -> float:
    tokens = _tokens(answer)
    words = max(1, len(tokens))
    metrics = len(NUM_RE.findall(answer))
    m_per_100 = (metrics / words) * 100.0
    acts = _count_set(answer, ACTION_VERBS)
    skills = _count_set(answer, set(ctx.get("skill_overlap", []) or ctx.get("resume_skills", [])))
    m_comp = min(1.0, m_per_100/2.0)
    act_comp = min(1.0, acts/3.0)
    skill_comp = min(1.0, skills/3.0)
    return _clip01(0.5*m_comp + 0.25*act_comp + 0.25*skill_comp)

def _clarity_score(answer: str) -> float:
    words = max(1, len(_tokens(answer)))
    sents = _sentences(answer)
    avg_len = words / max(1, len(sents))
    fillers = _count_set(answer, FILLERS)
    hedges = _count_set(answer, HEDGES)
    vague = _count_set(answer, VAGUE)
    ttr = _type_token_ratio(_tokens(answer))
    ttr_score = 1.0 - (abs(ttr - 0.5) / 0.5)  # 1.0 at 0.5
    ttr_score = _clip01(ttr_score)
    sent_score = 1.0 - _clip01(abs(avg_len - 18) / 18)  # ideal 12–24
    penalty = 0.12*fillers + 0.08*hedges + 0.06*vague
    raw = (0.5*sent_score + 0.5*ttr_score) - penalty
    return _clip01(raw)

def _alignment_score(answer: str, ctx: dict) -> float:
    q = ctx.get("question","")
    if not q: return 0.0
    return _clip01(_cosine(answer, q))

def _tip_catalog(answer: str, subs: dict, ctx: dict) -> List[str]:
    tips: List[str] = []
    def add(t):
        if t not in tips:
            tips.append(t)

    has_metric = _has_metric(answer)
    fillers = _count_set(answer, FILLERS) + _count_set(answer, HEDGES)
    sents = _sentences(answer)
    words = len(_tokens(answer))
    acts = _count_set(answer, ACTION_VERBS)
    skills = _count_set(answer, set(ctx.get("skill_overlap", []) or ctx.get("resume_skills", [])))

    bullets = []
    msb = ctx.get("micro_scripts_bullets", {})
    if ctx.get("current_key") in msb:
        bullets = msb[ctx["current_key"]]
    resume_hint = bullets[0] if bullets else None

    ranked = sorted(subs.items(), key=lambda kv: kv[1])
    for name, val in ranked:
        if name == "coverage" and val < 7.0:
            add("Tie your story to the JD priorities and the question explicitly.")
            if resume_hint: add(f"Drop in a concrete example: {resume_hint}")
        if name == "structure" and val < 7.0:
            add("Use S→A→R: 1 line of context, 2–3 actions, 1–2 results.")
        if name == "specificity" and val < 7.0:
            if not has_metric: add("Add one metric (%, ms, $, x) to prove impact.")
            if skills < 1: add("Name 1–2 tools/skills you used.")
            if acts < 2: add("Use strong action verbs (built, optimized, reduced).")
        if name == "clarity" and val < 7.0:
            if fillers > 1: add("Cut fillers/hedges; pause instead of saying 'um/like'.")
            if len(sents) > 6 or words > 180: add("Shorten sentences; aim for 12–24 words each.")
        if name == "alignment" and val < 6.5:
            add("Answer the question head-on in the first line, then give your example.")

    if not tips:
        add("Solid answer—consider previewing tradeoffs and risks to deepen it.")
    return tips[:4]

def score_answer(answer: str, question: str, ctx: dict) -> dict:
    answer = (answer or "").strip()
    if not answer:
        return {
            "coverage":0.0,"alignment":0.0,"structure":0.0,"specificity":0.0,"clarity":0.0,
            "overall":0.0,"tips":["Try a full S→A→R example with one metric."]
        }

    ctx = dict(ctx)
    ctx["question"] = question

    cov = _to10(_coverage_score(answer, ctx))
    ali = _to10(_alignment_score(answer, ctx))
    struc = _to10(_structure_score(answer))
    spec = _to10(_specificity_score(answer, ctx))
    clar = _to10(_clarity_score(answer))

    overall = round((0.28*cov + 0.26*struc + 0.22*spec + 0.14*clar + 0.10*ali), 1)

    subs = {"coverage":cov, "alignment":ali, "structure":struc, "specificity":spec, "clarity":clar}
    tips = _tip_catalog(answer, subs, ctx)

    return {**subs, "overall": overall, "tips": tips}

# ---------------- Question generation ----------------
BEHAVIORAL_TEMPLATES = [
    "Tell me about a time you shipped impact with {skills}. What was the result?",
    "Describe a challenging project involving {skills}. How did you make decisions and measure success?",
    "Walk me through a time you improved {focus}. What changed and how do you know?"
]
TECH_TEMPLATES = [
    "How would you design a service to improve {focus}? Talk tradeoffs and measurement.",
    "You need to integrate {skills} into an existing stack. What pitfalls and how to test?",
    "How do you ensure reliability and latency when scaling {focus}?"
]
COMPANY_TEMPLATES = [
    "Why {company}? Map your experience to {role} and our priorities around {focus}.",
    "How would you make impact in your first 90 days at {company} as a {role}?"
]

def _pick(lst, n): return lst[:n] if lst else []

def build_question_bank(ctx: dict) -> List[dict]:
    company = ctx.get("company", "[Company]")
    role = ctx.get("role", "[Role]")
    jd_terms = ctx.get("jd_top_terms", []) or ctx.get("jd_terms", [])
    skills = ctx.get("skill_overlap", []) or ctx.get("resume_skills", []) or ctx.get("jd_skills", [])
    focus = ", ".join(_pick(jd_terms, 3)) or "key priorities"
    sk = ", ".join(_pick(skills, 3)) or "relevant skills"

    qs = []
    for t in BEHAVIORAL_TEMPLATES:
        qs.append({"q": t.format(skills=sk, focus=focus), "kind":"behavioral", "key":"tmays"})
    for t in TECH_TEMPLATES:
        qs.append({"q": t.format(skills=sk, focus=focus), "kind":"technical", "key":"tech"})
    for t in COMPANY_TEMPLATES:
        qs.append({"q": t.format(company=company, role=role, focus=focus), "kind":"company", "key":"why"})
    return qs

# ---------------- API ----------------
class StartReq(BaseModel):
    match_id: Optional[str] = None
    mode: Optional[str] = "auto"  # auto|behavioral|technical|company

@router.post("/start")
async def start_session(req: StartReq):
    ctx = {
        "company": "[Company]",
        "role": "[Role]",
        "jd_top_terms": [],
        "skill_overlap": [],
        "resume_skills": [],
        "jd_skills": [],
        "micro_scripts_bullets": {}
    }
    if req.match_id:
        m = MATCHES.get(req.match_id)
        if not m:
            raise HTTPException(status_code=404, detail="match_id not found")
        for k in ("company","role","jd_top_terms","skill_overlap","resume_skills","jd_skills","micro_scripts_bullets"):
            if k in m: ctx[k] = m[k]

    bank = build_question_bank({**ctx})
    if req.mode in {"behavioral","technical","company"}:
        bank = [q for q in bank if q["kind"] == req.mode] + [q for q in bank if q["kind"] != req.mode]
    if not bank:
        bank = [{"q":"Tell me about a recent project you’re proud of. What changed because of your work?","kind":"behavioral","key":"tmays"}]

    sid = str(uuid.uuid4())
    first = bank[0]
    SESSIONS[sid] = {
        "created_at": time.time(),
        "company": ctx.get("company"),
        "role": ctx.get("role"),
        "jd_top_terms": ctx.get("jd_top_terms", []),
        "skill_overlap": ctx.get("skill_overlap", []),
        "resume_skills": ctx.get("resume_skills", []),
        "jd_skills": ctx.get("jd_skills", []),
        "micro_scripts_bullets": ctx.get("micro_scripts_bullets", {}),
        "queue": bank,
        "idx": 0,
        "scores": [],
        "current_key": first["key"],
    }
    return {"session_id": sid, "question": first["q"]}

class AnswerReq(BaseModel):
    session_id: str
    answer_text: str

@router.post("/answer")
async def submit_answer(req: AnswerReq):
    s = SESSIONS.get(req.session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")

    idx = s["idx"]
    qobj = s["queue"][idx]
    s["current_key"] = qobj["key"]

    ctx = {
        "jd_top_terms": s.get("jd_top_terms", []),
        "skill_overlap": s.get("skill_overlap", []),
        "resume_skills": s.get("resume_skills", []),
        "micro_scripts_bullets": s.get("micro_scripts_bullets", {}),
        "current_key": s.get("current_key")
    }
    result = score_answer(req.answer_text, qobj["q"], ctx)
    s["scores"].append({
        "question": qobj["q"],
        "kind": qobj["kind"],
        **result
    })

    # Resume-aware hint (for UI if needed)
    bullets = []
    msb = s.get("micro_scripts_bullets", {})
    if qobj["key"] in msb:
        bullets = msb[qobj["key"]]
    follow_hint = bullets[0] if bullets else "Add a concrete project with a metric to anchor your story."

    # Next question (cycle)
    idx = (idx + 1) % len(s["queue"])
    s["idx"] = idx
    next_q = s["queue"][idx]
    s["current_key"] = next_q["key"]

    # Return tips-only speech, not announcing the next question (UI will announce on click)
    feedback_text = " ".join([t.strip().rstrip(".") + "." for t in result.get("tips", []) if t and t.strip()])

    return {
        "scores": result,
        "next_question": next_q["q"],
        "feedback_speech": feedback_text,
        "resume_hint": follow_hint
    }
