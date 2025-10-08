"""
Fashion Theory RAG module.
Ingests public-domain or open-access fashion texts from academic_papers/,
builds a FAISS index, and answers questions in the 'Chat with Stylist' tab.
"""

import os
from pathlib import Path
from typing import List

from google import genai
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text
from langchain_core.embeddings import Embeddings

# Constants
INDEX_DIR = Path("faiss_index/fashion_theory")
GEMINI_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "text-embedding-004"  # Gemini embed model
PAPER_FOLDER = Path("open_fashion_texts")

# ---------------------------
# Gemini Client & Embeddings
# ---------------------------

def make_client(api_key: str = None):
    """Create Gemini API client."""
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY env var is not set")
    return genai.Client(api_key=api_key)

class GeminiEmbeddings(Embeddings):
    def __init__(self, client, model: str = EMBED_MODEL):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [
            self.client.models.embed_content(model=self.model, contents=t).embeddings[0].values
            for t in texts
        ]

    def embed_query(self, text: str) -> List[float]:
        return self.client.models.embed_content(model=self.model, contents=text).embeddings[0].values


# ---------------------------
# Load public papers
# ---------------------------

def load_papers(folder: Path = PAPER_FOLDER) -> List[Document]:
    """Load .pdf and .txt files as Documents."""
    docs = []
    for file in folder.glob("*"):
        if file.suffix.lower() == ".pdf":
            raw = extract_text(str(file)) or ""
            pages = [p.strip() for p in raw.split("\f") if p.strip()]
            for i, page in enumerate(pages, start=1):
                docs.append(Document(page_content=page, metadata={"source": str(file), "page": i}))
        elif file.suffix.lower() == ".txt":
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            docs.append(Document(page_content=text, metadata={"source": str(file), "page": 1}))
    return docs

def chunk_docs(docs: List[Document]) -> List[Document]:
    """Chunk documents into smaller pieces."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)

# ---------------------------
# Build / Load Vector Store
# ---------------------------

def get_vectorstore(client) -> FAISS:
    embeddings = GeminiEmbeddings(client)
    faiss_file = INDEX_DIR / "index.faiss"
    pkl_file = INDEX_DIR / "index.pkl"

    if faiss_file.exists() and pkl_file.exists():
        return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    else:
        raise RuntimeError("Fashion Theory index not found. Run scripts/build_fashion_theory_index.py first.")



def retrieve_docs(db: FAISS, query: str, k: int = 5) -> List[Document]:
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5}
    )
    return retriever.invoke(query)

def format_context_and_citations(docs: List[Document], max_chars: int = 900):
    """
    Return a string of chunks with [source/page] tags
    and a citations string you can append to the answer.
    """
    blocks = []
    citations = []
    for d in docs:
        src = Path(d.metadata.get("source", "unknown")).name
        page = d.metadata.get("page", "?")
        text = d.page_content.strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        blocks.append(f"[{src} p.{page}] {text}")
        citations.append(f"{src} p.{page}")
    context_text = "\n\n".join(blocks)
    citations_text = "Sources: " + ", ".join(sorted(set(citations)))
    return context_text, citations_text


# ---------------------------
# Generate Answer
# ---------------------------

def generate_fashion_theory_advice(client, question: str, temperature: float = 0.7) -> str:
    """
    Answer a question using the Fashion Theory RAG.
    """
    db = get_vectorstore(client)
    docs = retrieve_docs(db, question)
    # DEBUG: see what was retrieved
    print("Retrieved docs:")
    for d in docs:
        print(d.metadata.get("source"), d.metadata.get("page"))

    context, citations = format_context_and_citations(docs)

    system_prompt = """You are a knowledgeable fashion assistant.
You also have access to a set of public-domain fashion texts covering historical fashion trends, garment construction methods, pattern cutting, and fashion illustration .
•⁠  ⁠When the user asks something clearly related to these topics, ground your answer in the provided documents and cite the source as given.
•⁠  ⁠If you cannot find relevant information in the provided documents, still do your best to answer from your broader knowledge.
•⁠  ⁠Never refuse a question just because it is outside the scope; always answer as best you can, clearly indicating whether the answer comes from the documents or from general knowledge.
•⁠  ⁠If you’re unsure, say so rather than making things up.

"""

    full_prompt = f"""{system_prompt}

Question: {question}

Context:
{context}

{citations}
"""

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[{"text": full_prompt}],
        config={"temperature": temperature}
    )
    return resp.text
