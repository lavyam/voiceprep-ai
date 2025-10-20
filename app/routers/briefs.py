# app/routers/briefs.py (DROP-IN REPLACEMENT)
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
)
from reportlab.lib import colors

# Sessions are required; MATCHES is optional (be robust even if not wired)
from app.routers.live import SESSIONS
try:
    from app.routers.match import MATCHES
except Exception:
    MATCHES = {}

router = APIRouter()

# ---------------- Helpers ----------------
def _truncate(s: str, n: int) -> str:
    if not s: return ""
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"

def _average(vals):
    vals = [float(v or 0.0) for v in vals if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else 0.0

def _average_overall(items):
    return _average([i.get("overall", 0.0) for i in (items or [])])

def _avg_subscores(items):
    if not items: 
        return {"coverage":0,"structure":0,"specificity":0,"clarity":0,"alignment":0,"overall":0}
    keys = ["coverage","structure","specificity","clarity","alignment","overall"]
    return {k: _average([i.get(k,0.0) for i in items]) for k in keys}

def _collect_top_tips(items, limit=8):
    seen = set()
    out = []
    for it in items or []:
        for t in (it.get("tips") or []):
            if t and t not in seen:
                seen.add(t)
                out.append(t)
                if len(out) >= limit: break
        if len(out) >= limit: break
    return out

def _build_per_q_rows(items):
    rows = [["#", "Question", "Overall", "Cov", "Struc", "Spec", "Clar"]]
    for idx, it in enumerate(items or [], 1):
        q = _truncate(it.get("question",""), 92)
        rows.append([
            str(idx),
            q,
            str(it.get("overall","-")),
            str(it.get("coverage","-")),
            str(it.get("structure","-")),
            str(it.get("specificity","-")),
            str(it.get("clarity","-")),
        ])
    return rows

def _closing_message(avg: float, answered_count: int) -> str:
    LOW, MID = 6.0, 8.0
    if answered_count == 0:
        return "Nice effort! Try answering at least one question to get a personalized summary. You’ve got this!"
    if avg < LOW:
        return f"Thanks for practicing! Your average score was {avg}. Focus on stating the question upfront, using S→A→R, and adding one metric. You’re improving every rep — you’ll do great!"
    if avg < MID:
        return f"Good session — average {avg}. What worked: relevance and clarity. Next time, add one measurable result and name 1–2 tools to ace it. You’ve got this!"
    return f"Awesome session — average {avg}! Strong structure and specificity; you seem ready. Keep this momentum and good luck!"

def _focus_areas(avg_subs: dict) -> list[str]:
    """
    Turn weakest subscores into concrete focus bullets.
    Inputs are already 0–10 averages.
    """
    tips = []
    # Sort subs (ignore 'overall') by ascending score
    order = sorted([(k, v) for k, v in avg_subs.items() if k != "overall"], key=lambda x: x[1])
    for k, v in order[:3]:  # top 2–3 weaknesses
        if k == "coverage":
            tips.append("Tie your intro line to the exact question and name the JD priority you’re addressing.")
        elif k == "structure":
            tips.append("Use S→A→R: 1 line of context, 2–3 concrete actions, 1–2 measurable results.")
        elif k == "specificity":
            tips.append("Add one metric (%, ms, $, x) and name 1–2 tools to prove impact.")
        elif k == "clarity":
            tips.append("Trim filler words and keep sentences ~12–24 words; pause instead of ‘um/like’.")
        elif k == "alignment":
            tips.append("Answer the ask in your first sentence, then support with a tight example.")
    # De-dup safety (just in case)
    out, seen = [], set()
    for t in tips:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _fallback_session_summary_text(session_id: str, company: str, role: str) -> str:
    s = SESSIONS.get(session_id, {})
    items = s.get("scores", [])
    count = len(items)
    avg = _average_overall(items)
    top_tips = _collect_top_tips(items, limit=6)

    lines = []
    lines.append("Interview Practice Brief")
    lines.append(f"Session ID: {session_id}")
    lines.append(f"Company / Role: {company} / {role}")
    lines.append(f"Questions Answered: {count}")
    lines.append(f"Average Score: {avg}")
    lines.append("")
    if top_tips:
        lines.append("Top Tips:")
        for t in top_tips: lines.append(f" • {t}")
        lines.append("")
    if items:
        lines.append("Per-Question Snapshot:")
        for i, it in enumerate(items, 1):
            lines.append(f"{i}. {it.get('question','')}")
            lines.append(f"   Overall {it.get('overall','-')} | Coverage {it.get('coverage','-')} | Structure {it.get('structure','-')} | Specificity {it.get('specificity','-')} | Clarity {it.get('clarity','-')}")
        lines.append("")
    lines.append("Closing:")
    lines.append(_closing_message(avg, count))
    return "\n".join(lines)

# ---------------- PDF Builder ----------------
def build_pdf(data: dict) -> BytesIO:
    """
    data keys expected (safe to omit):
      - company, role
      - company_overview
      - resume_keywords, jd_keywords, overlap, coverage_score  (from MATCHES)
      - talking_points (list[str]), resume_highlights (list[str])  (from MATCHES)
      - tmays, tech_fit  (talking points legacy)
      - scores: list[dict] with question, overall, coverage, structure, specificity, clarity, tips
      - session_summary_text: optional long-form free text (UI param or backend-built)
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, leftMargin=36, rightMargin=36, topMargin=40, bottomMargin=36)
    styles = getSampleStyleSheet()

    H1 = styles['Heading1']
    H2 = styles['Heading2']
    H3 = styles['Heading3']
    Body = styles['BodyText']

    story = []
    story.append(Paragraph(f"Interview Prep Brief – {data.get('company','[Company]')} | {data.get('role','[Role]')}", H1))
    story.append(Paragraph("Generated by: VoicePrep Agent", Body))
    story.append(Spacer(1, 12))

    # Company Snapshot
    story.append(Paragraph("Company Snapshot", H2))
    story.append(Paragraph(data.get('company_overview','(Use /api/qa/company in UI and pass here if desired.)'), Body))
    story.append(Spacer(1, 12))

    # Resume ↔ JD Match
    ov = data.get("overlap", []) or []
    score = float(data.get("coverage_score", 0.0) or 0.0)
    story.append(Paragraph("Resume ↔ Job Description Match", H2))
    story.append(Paragraph(f"Coverage Score: {int(score*100)}%", Body))
    rows = [["Overlap Keywords", ", ".join(ov[:12]) or "—"]]
    match_table = Table(rows, colWidths=[520])
    match_table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    story.append(match_table)
    story.append(Spacer(1, 12))

    # Talking Points (from MATCHES + legacy tmays/tech_fit)
    tps = data.get("talking_points") or []
    highlights = data.get("resume_highlights") or []
    story.append(Paragraph("Talking Points", H2))
    tp_items = []
    # Prefer provided talking points; fall back to tmays/tech_fit lines
    if tps:
        tp_items = [ListItem(Paragraph(tp, Body)) for tp in tps[:8]]
    else:
        tp_items = [
            ListItem(Paragraph(data.get('tmays', 'Elevator intro tailored to mission.'), Body)),
            ListItem(Paragraph(data.get('tech_fit',"Map your projects to the role's core skills."), Body))
        ]
    story.append(ListFlowable(tp_items, bulletType='bullet'))
    story.append(Spacer(1, 8))

    if highlights:
        story.append(Paragraph("Resume Highlights to Mention", H3))
        story.append(ListFlowable([ListItem(Paragraph(h, Body)) for h in highlights[:8]], bulletType='bullet'))
        story.append(Spacer(1, 12))

    # Live Prep Debrief (Overview + Tips + Focus Areas)
    items = data.get('scores', []) or []
    avg = _average_overall(items)
    avgs = _avg_subscores(items)
    story.append(Paragraph("Live Prep Debrief (Last Session)", H2))
    story.append(Paragraph(f"Questions Answered: {len(items)}", Body))
    story.append(Paragraph(f"Averages — Overall {avgs.get('overall',0)} | Coverage {avgs.get('coverage',0)} | Structure {avgs.get('structure',0)} | Specificity {avgs.get('specificity',0)} | Clarity {avgs.get('clarity',0)} | Alignment {avgs.get('alignment',0)}", Body))
    story.append(Spacer(1, 8))

    # Top Tips (dedup across session)
    top_tips = _collect_top_tips(items, limit=6)
    if top_tips:
        story.append(Paragraph("Top Tips", H3))
        story.append(ListFlowable([ListItem(Paragraph(t, Body)) for t in top_tips], bulletType='bullet'))
        story.append(Spacer(1, 8))

    # Overall Focus Areas (what to focus on) — generated from weakest subscores
    focus_bullets = _focus_areas(avgs)
    if focus_bullets:
        story.append(Paragraph("Overall Focus Areas", H3))
        story.append(ListFlowable([ListItem(Paragraph(t, Body)) for t in focus_bullets], bulletType='bullet'))
        story.append(Spacer(1, 12))

    # Per-Question Snapshot table
    if items:
        pq_rows = _build_per_q_rows(items)
        pq_table = Table(pq_rows, colWidths=[24, 300, 48, 36, 44, 38, 38])
        pq_table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
            ('GRID',(0,0),(-1,-1),0.5,colors.grey),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
        ]))
        story.append(pq_table)
        story.append(Spacer(1, 12))

    # Closing
    story.append(Paragraph("Closing", H3))
    story.append(Paragraph(_closing_message(avg, len(items)), Body))
    story.append(Spacer(1, 16))

    # Optional freeform session summary (from UI or backend)
    summary_text = data.get("session_summary_text")
    if summary_text:
        story.append(Paragraph("Session Summary (freeform)", H3))
        story.append(Paragraph(_truncate(summary_text, 3000).replace("\n", "<br/>"), Body))
        story.append(Spacer(1, 12))

    doc.build(story)
    buf.seek(0)
    return buf

# ---------------- Route ----------------
@router.get("/{session_id}.pdf")
async def download_brief(
    session_id: str,
    company: str = Query(default="[Company]"),
    role: str = Query(default="[Role]"),
    match_id: str | None = Query(default=None, description="Optional ID from /api/match/upload"),
    session_summary: str | None = Query(default=None, description="Optional freeform summary text (URL-safe, e.g., from UI)")
):
    # Pull session / match data
    s = SESSIONS.get(session_id, {})
    # MATCHES is optional; handle missing keys gracefully
    m = MATCHES.get(match_id) if match_id else {}

    # Compose payload (preserving your existing fields)
    payload = {
        "company": company,
        "role": role,
        "company_overview": "Fill from /api/qa/company in UI and pass here if desired.",
        "tmays": "Elevator intro tailored to mission.",
        "tech_fit": "Map your projects to the role's core skills.",
        "scores": s.get("scores", []),
    }

    # Enrich with MATCH data if available
    if m:
        payload.update({
            "resume_keywords": m.get("resume_keywords", []),
            "jd_keywords": m.get("jd_keywords", []),
            "overlap": m.get("overlap", []) or m.get("skill_overlap", []),
            "coverage_score": m.get("coverage_score", m.get("coverage", 0.0)),
            "talking_points": m.get("talking_points", []),
            "resume_highlights": m.get("resume_highlights", []),
        })

    # Prefer UI-provided summary; otherwise auto-build from session
    if session_summary and session_summary.strip():
        payload["session_summary_text"] = session_summary.strip()
    else:
        payload["session_summary_text"] = _fallback_session_summary_text(session_id, company, role)

    try:
        pdf = build_pdf(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    headers = {"Content-Disposition": f'attachment; filename="Interview_Prep_Brief_{session_id}.pdf"'}
    return StreamingResponse(pdf, media_type="application/pdf", headers=headers)
