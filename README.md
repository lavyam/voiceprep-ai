# New: Resume + JD Upload Flow

- `POST /api/match/upload` (multipart form): fields `resume` (file), `jd` (file), `company` (form), `role` (form)
  - Returns `match_id`, keyword overlap, and a `coverage_score` (0..1).
- `GET /api/briefs/{session_id}.pdf?...&match_id=...` will embed overlap keywords and score in the PDF brief.

Note: For demo speed, PDF/DOCX parsing is not included; text files work best. Plug in real parsers later.
