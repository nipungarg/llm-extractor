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
        "prop": "extracts",  # we want the article body
        "explaintext": 1,  # give us plain text, not HTML
        "redirects": 1,  # follow redirects, e.g. "KYC" -> "Know your customer"
        "format": "json",
        "formatversion": 2,  # cleaner response: pages is a simple list
        "titles": title,
    }
    headers = {
        "User-Agent": "ai-bootcamp/1.0 (learning project)"
    }  # Wikipedia asks for a UA
    resp = httpx.get(WIKI_API, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    return pages[0].get("extract", "") if pages else ""


def load_cord_receipts(n: int = 5, split: str = "test"):
    """Stream real receipt images + ground-truth labels from the public CORD-v2 dataset.

    No images to create yourself — these are real Indonesian retail receipts with
    structured ground truth, ideal for testing extraction accuracy.
    Yields (image_bytes, ground_truth_dict).
    """
    import io
    import json

    from datasets import load_dataset

    # streaming=True avoids downloading the whole dataset
    ds = load_dataset("naver-clova-ix/cord-v2", split=split, streaming=True)
    for i, example in enumerate(ds):
        if i >= n:
            break
        buf = io.BytesIO()
        example["image"].save(buf, format="PNG")  # PIL image -> PNG bytes
        ground_truth = json.loads(example["ground_truth"]).get("gt_parse", {})
        yield buf.getvalue(), ground_truth
