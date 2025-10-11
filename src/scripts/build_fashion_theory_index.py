from src.retrieval.fashion_theory_rag import make_client, GeminiEmbeddings, load_papers, chunk_docs, INDEX_DIR
from langchain_community.vectorstores import FAISS
import sys

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("BUILDING FASHION THEORY FAISS INDEX", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/5] Creating Gemini client...", flush=True)
    client = make_client()
    embeddings = GeminiEmbeddings(client)
    print("✓ Client created", flush=True)

    from pathlib import Path
    from src.retrieval.fashion_theory_rag import load_papers, chunk_docs

    print("\n[2/5] Loading papers from open_fashion_texts/...", flush=True)
    docs = load_papers()
    print(f"✓ Loaded {len(docs)} documents", flush=True)

    print("\n[3/5] Chunking documents...", flush=True)
    chunks = chunk_docs(docs)
    print(f"✓ Created {len(chunks)} chunks", flush=True)

    print("\n[4/5] Creating embeddings and building FAISS index...", flush=True)
    print(f"    This will take several minutes for {len(chunks)} chunks.", flush=True)
    print("    Creating embeddings via Gemini API...", flush=True)
    db = FAISS.from_documents(chunks, embeddings)
    print("✓ FAISS index built", flush=True)

    print("\n[5/5] Saving index to disk...", flush=True)
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    db.save_local(str(INDEX_DIR))
    print(f"✓ Saved to {INDEX_DIR}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("✅ SUCCESS! Fashion Theory FAISS index built at", INDEX_DIR, flush=True)
    print("=" * 70, flush=True)
