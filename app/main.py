from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.routers import profile, qa, live, briefs, match

app = FastAPI(title="VoicePrep API", version="0.2.0")

# Routers (keep your API on /api/*)
app.include_router(profile.router, prefix="/api/profile", tags=["profile"])
app.include_router(match.router,   prefix="/api/match",   tags=["match"])
app.include_router(qa.router,      prefix="/api/qa",      tags=["qa"])
app.include_router(live.router,    prefix="/api/live",    tags=["live"])
app.include_router(briefs.router,  prefix="/api/briefs",  tags=["briefs"])

# CORS (open for demo; tighten allow_origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check (useful on Render/Railway)
@app.get("/healthz", tags=["meta"])
def healthz():
    return {"ok": True, "version": "0.2.0"}

# Static files and SPA index
# Serve assets at /static (e.g., /static/js/app.js)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Serve index.html at /
@app.get("/", include_in_schema=False)
def index():
    return FileResponse("app/static/index.html")
