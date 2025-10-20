from datetime import datetime
def get_recent_news(company: str, days: int = 60):
    today = datetime.utcnow().date().isoformat()
    return [
        {"date": today, "title": f"{company} launches feature for voice security", "url": "#"},
        {"date": today, "title": f"{company} partners with carrier to block spam calls", "url": "#"}
    ]
