from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.tools.company import get_company_overview
from app.tools.role import get_role_insights
from app.tools.news import get_recent_news

router = APIRouter()

class QAResponse(BaseModel):
    company_overview: dict | None = None
    role_insights: dict | None = None
    recent_news: list[dict] | None = None

@router.get("/company")
async def company(company: str, role: Optional[str] = None, days: int = 60):
    try:
        overview = get_company_overview(company)
        role_info = get_role_insights(company, role) if role else None
        news = get_recent_news(company, days)
        return QAResponse(company_overview=overview, role_insights=role_info, recent_news=news)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
