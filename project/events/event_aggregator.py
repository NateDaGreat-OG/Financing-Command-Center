from .ipo_scraper import fetch_upcoming_ipos
from .nasdaq_additions_scraper import fetch_nasdaq_additions

def aggregate_events():
    ipos = fetch_upcoming_ipos()
    additions = fetch_nasdaq_additions()
    return ipos + additions
