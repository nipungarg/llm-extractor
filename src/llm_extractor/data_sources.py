from __future__ import annotations

import httpx

WIKI_API = "https://en.wikipedia.org/w/api.php"


def fetch_wikipedia(title: str) -> str:
    """Scrape the plain text of a Wikipedia article — a real, free data source (no API key).

    We use Wikipedia's public MediaWiki API instead of pasting dummy text, so every
    later demo runs on real content.
    """
    params = {
        "action": "query",
        "prop": "extracts",      # we want the article body
        "explaintext": 1,        # give us plain text, not HTML
        "redirects": 1,          # follow redirects, e.g. "KYC" -> "Know your customer"
        "format": "json",
        "formatversion": 2,      # cleaner response: pages is a simple list
        "titles": title,
    }
    headers = {"User-Agent": "ai-bootcamp/1.0 (learning project)"}  # Wikipedia asks for a UA
    resp = httpx.get(WIKI_API, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    return pages[0].get("extract", "") if pages else ""