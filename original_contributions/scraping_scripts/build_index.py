import os, json, faiss, sys

ROOT = os.path.dirname(os.path.dirname(__file__))
CURATED = os.path.join(ROOT, "data", "curated", "curated.jsonl")
INDEX_DIR = os.path.join(ROOT, "data", "index")
META_PATH = os.path.join(INDEX_DIR, "meta.jsonl")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")

os.makedirs(INDEX_DIR, exist_ok=True)

# Choose embedding model via sentence-transformers
def load_embedder(model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def load_curated():
    items = []
    with open(CURATED, "r", encoding="utf-8") as r:
        for line in r:
            items.append(json.loads(line))
    return items

def build_index(model_name="all-MiniLM-L6-v2", normalize=True):
    items = load_curated()
    texts = [f"{it.get('title','')}\n{it.get('content','')}"[:2048] for it in items]  # clip for speed
    embedder = load_embedder(model_name)
    print("Encoding texts...")
    X = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=normalize)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d) if normalize else faiss.IndexFlatL2(d)
    index.add(X)

    # save
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as w:
        for it in items:
            w.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Built index with {len(items)} vectors. Saved to {INDEX_PATH}")

if __name__ == "__main__":
    model = "all-MiniLM-L6-v2"
    norm = True
    if len(sys.argv) > 1:
        model = sys.argv[1]
    if len(sys.argv) > 2:
        norm = sys.argv[2].lower() == "true"
    build_index(model, norm)