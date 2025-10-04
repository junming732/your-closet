# src/retrieval/gemini_rag.py

import os
from pathlib import Path
from typing import List

from google import genai

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text

# ---------------------------
# Gemini Client Setup
# ---------------------------

def make_client(api_key: str = None):
    """Create Gemini API client."""
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY env var is not set")
    return genai.Client(api_key=api_key)

# default model names from her notebook
GEMINI_MODEL = "gemini-2.5-flash"  # can be changed in app
INDEX_DIR = "faiss_index"          # folder to save FAISS index

# ---------------------------
# Embeddings (Gemini)
# ---------------------------

class GeminiEmbeddings(Embeddings):
    def __init__(self, client, model="text-embedding-004"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return result.embeddings[0].values

# ---------------------------
# PDF -> Documents
# ---------------------------

def load_pdf_as_documents(path: str) -> List[Document]:
    """Extract text from PDF and create documents with page metadata."""
    raw = extract_text(path) or ""
    pages = [p.strip() for p in raw.split("\f") if p.strip()]
    docs = []
    for i, page in enumerate(pages, start=1):
        docs.append(Document(page_content=page, metadata={"source": path, "page": i}))
    return docs

# ---------------------------
# Chunking
# ---------------------------

def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)

# ---------------------------
# Build / Load Vector Store
# ---------------------------

def get_vectorstore(chunks: List[Document], embeddings: Embeddings) -> FAISS:
    if Path(INDEX_DIR).exists():
        db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(INDEX_DIR)
    return db

# ---------------------------
# Retrieval
# ---------------------------

def retrieve_docs(db: FAISS, query: str, k: int = 5) -> List[Document]:
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5}
    )
    return retriever.get_relevant_documents(query)

def format_context(docs: List[Document], max_chars_per_chunk: int = 900) -> str:
    blocks = []
    for d in docs:
        page = d.metadata.get("page", "?")
        text = d.page_content.strip().replace("\n", " ")
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "…"
        blocks.append(f"[p.{page}] {text}")
    return "\n\n".join(blocks)

def format_citations(docs: List[Document]) -> str:
    pages = []
    for d in docs:
        p = d.metadata.get("page", None)
        if p is not None:
            pages.append(int(p))
    pages = sorted(set(pages))
    if not pages:
        return "Sources: (no page markers)"
    return "Sources: " + ", ".join([f"p.{p}" for p in pages])

# ---------------------------
# Generate Answer
# ---------------------------

def generate_outfit_advice(client, base_prompt: str, docs: list, temperature: float = 0.7) -> str:
    """
    Take the stylist prompt built in wardrobe_app.generate_outfit,
    append retrieved style tips from docs,
    and call Gemini model once.
    """
    context = format_context(docs)
    citations = format_citations(docs)

    full_prompt = f"""{base_prompt}

            Retrieved Style Tips:
            {context}

            {citations}
            """

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[{"text": full_prompt}],
        config={"temperature": temperature}
    )
    return resp.text

