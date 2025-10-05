from src.retrieval.fashion_theory_rag import make_client, GeminiEmbeddings, load_papers, chunk_docs, INDEX_DIR
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    client = make_client()
    embeddings = GeminiEmbeddings(client)
    from pathlib import Path

    from src.retrieval.fashion_theory_rag import load_papers, chunk_docs
    docs = load_papers()
    chunks = chunk_docs(docs)
    db = FAISS.from_documents(chunks, embeddings)
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    db.save_local(str(INDEX_DIR))
    print("Fashion Theory FAISS index built successfully at", INDEX_DIR)
