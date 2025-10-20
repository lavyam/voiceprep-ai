from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict
import uuid

PROFILES: Dict[str, dict] = {}
router = APIRouter()

class Profile(BaseModel):
    name: str | None = None
    headline: str | None = None
    skills: list[str] = []
    highlights: list[str] = []

@router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    if file.content_type not in {"application/pdf","text/plain","application/msword","application/vnd.openxmlformats-officedocument.wordprocessingml.document"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    profile_id = str(uuid.uuid4())
    # Minimal fake parse for demo
    PROFILES[profile_id] = Profile(
        name="Candidate",
        headline="AI/ML Engineer",
        skills=["Python","FastAPI","LangGraph","STT/TTS","LLM Tool-Calling"],
        highlights=[
            "Built multi-agent legal research pipeline",
            "Deployed FastAPI service with CI/CD",
            "Evaluated tool-calling reliability at scale"
        ]
    ).model_dump()
    return {"profile_id": profile_id, "profile": PROFILES[profile_id]}

@router.get("/{profile_id}")
async def get_profile(profile_id: str):
    p = PROFILES.get(profile_id)
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found")
    return p
