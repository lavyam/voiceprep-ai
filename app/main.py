from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routers import profile, qa, live, briefs, match

app = FastAPI(title="VoicePrep API", version="0.2.0")

app.include_router(profile.router, prefix="/api/profile", tags=["profile"])
app.include_router(match.router, prefix="/api/match", tags=["match"])
app.include_router(qa.router, prefix="/api/qa", tags=["qa"])
app.include_router(live.router, prefix="/api/live", tags=["live"])
app.include_router(briefs.router, prefix="/api/briefs", tags=["briefs"])

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
