import os, json, time, hashlib
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_RAW_DIR = os.path.join(ROOT, "data", "raw")
CONFIG = os.path.join(ROOT, "config", "feeds.json")

os.makedirs(DATA_RAW_DIR, exist_ok=True)

def load_feeds():
    with open(CONFIG, "r") as f:
        return json.load(f)["feeds"]

def clean_html(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        # remove scripts/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception:
        return html

def entry_to_obj(entry):
    link = entry.get("link", "")
    title = entry.get("title", "")
    summary = entry.get("summary", "") or entry.get("description", "")
    published = entry.get("published", "") or entry.get("updated", "")
    published_ts = None
    if published:
        try:
            published_ts = datetime(*entry.published_parsed[:6]).isoformat()
        except Exception:
            published_ts = None

    content_text = clean_html(summary)
    # some feeds include content in content[0]['value']
    for c in entry.get("content", []):
        if "value" in c and len(c["value"]) > len(summary):
            content_text = clean_html(c["value"])

    src = urlparse(link).netloc
    uid = hashlib.md5((link or title).encode("utf-8")).hexdigest()
    return {
        "id": uid,
        "source": src,
        "title": title,
        "link": link,
        "published": published_ts,
        "content": content_text
    }

def scrape():
    feeds = load_feeds()
    out_path = os.path.join(DATA_RAW_DIR, f"raw_{int(time.time())}.jsonl")
    seen = set()
    with open(out_path, "w", encoding="utf-8") as w:
        for feed in feeds:
            print(f"Fetching {feed}")
            d = feedparser.parse(feed)
            for e in tqdm(d.entries):
                obj = entry_to_obj(e)
                if obj["id"] in seen:
                    continue
                seen.add(obj["id"])
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    scrape()