

from urllib.parse import urlparse
import time
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT = 20




def web_search(query: str, max_results: int = 10) -> list[dict]:
    with DDGS() as ddgs:
        try:
            results = list(ddgs.text(query, max_results=max_results))
        except:
            return []
    return [
        {"url": r["href"], "title": r["title"], "snippet": r.get("body", "")}
        for r in results
    ]

def fetch_page(url, delay=None):

    session = requests.Session()
    try:
        resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    except requests.RequestException as e:
        return None
    finally:
        if delay:
            time.sleep(delay)

    if resp.status_code != 200:
        return None

    return resp