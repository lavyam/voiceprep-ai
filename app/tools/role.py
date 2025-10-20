def get_role_insights(company: str, role: str | None) -> dict:
    if not role:
        return {}
    return {
        "role": role,
        "responsibilities": [
            "Design voice-enabled agents (STT/TTS, tool-calling)",
            "Evaluate generative code suggestions pragmatically",
            "Integrate with telephony and external APIs"
        ],
        "skills": ["Python","FastAPI","LLMs","LangGraph","Twilio","Observability"]
    }
