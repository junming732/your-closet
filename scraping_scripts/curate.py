import os, json, glob, re, hashlib

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
CURATED_DIR = os.path.join(ROOT, "data", "curated")
os.makedirs(CURATED_DIR, exist_ok=True)

BRAND_HINTS = [
    "Zara","H&M","COS","Arket","Uniqlo","Levi's","Gucci","Prada","Saint Laurent","Balenciaga",
    "Nike","Adidas","New Balance","Dr. Martens","Onitsuka","Coach","Michael Kors","Reformation",
    "Aritzia","Everlane","AllSaints","Mango","Massimo Dutti","Sézane","Lululemon"
]
COLORS = ["black","white","beige","cream","navy","blue","red","green","khaki","tan","brown","grey","gray","pastel","pink","yellow","olive"]
STYLES = ["casual","business","business casual","smart casual","streetwear","evening","formal","semi-formal","athleisure","minimalist","boho","preppy","y2k"]

def heuristic_extract(text):
    text_l = text.lower()
    brands = sorted({b for b in BRAND_HINTS if b.lower() in text_l})
    colors = sorted({c for c in COLORS if re.search(r"\b"+re.escape(c)+r"\b", text_l)})
    styles = sorted({s for s in STYLES if s in text_l})
    # price mentions (very rough): "$120" or "120 USD" or "€90"
    prices = re.findall(r"(?:(?:USD|\$|€|£)\s?\d{2,5}|\d{2,5}\s?(?:USD|EUR|GBP))", text)
    return brands, colors, styles, prices

def dedupe_and_extract():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "raw_*.jsonl")))
    seen = set()
    out_path = os.path.join(CURATED_DIR, "curated.jsonl")
    n_in, n_out = 0, 0
    with open(out_path, "w", encoding="utf-8") as w:
        for f in files:
            with open(f, "r", encoding="utf-8") as r:
                for line in r:
                    n_in += 1
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    # simple dedupe by id or hash of title+link
                    uid = obj.get("id") or hashlib.md5((obj.get("title","")+obj.get("link","")).encode("utf-8")).hexdigest()
                    if uid in seen:
                        continue
                    seen.add(uid)
                    content = obj.get("content","")
                    brands, colors, styles, prices = heuristic_extract(content + " " + obj.get("title",""))
                    out = {
                        "id": uid,
                        "title": obj.get("title"),
                        "link": obj.get("link"),
                        "source": obj.get("source"),
                        "published": obj.get("published"),
                        "content": content,
                        "attributes": {
                            "brands": brands,
                            "colors": colors,
                            "styles": styles,
                            "prices": prices
                        }
                    }
                    w.write(json.dumps(out, ensure_ascii=False) + "\n")
                    n_out += 1
    print(f"Read {n_in} raw entries, wrote {n_out} curated entries to {out_path}")

if __name__ == "__main__":
    dedupe_and_extract()