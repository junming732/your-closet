# src/retrieval/gemini_rag.py

import os
from pathlib import Path
from typing import List

from google import genai
from google.genai import types

from langchain_community.vectorstores import FAISS
from src.app.logger_config import setup_logger, log_api_call, log_api_success, log_api_error

# Logger 
logger = setup_logger(__name__)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text


# Gemini Client Setup
def make_client(api_key: str = None):
    """Create Gemini API client."""
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY env var is not set")
    return genai.Client(api_key=api_key)

GEMINI_MODEL = "gemini-2.5-flash"  
BEGINNER_INDEX_DIR = "faiss_index/beginner_guide"  # Practical styling tips
THEORY_INDEX_DIR = "faiss_index/fashion_theory"     # Academic fashion knowledge

# Embeddings (Gemini)
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


# PDF -> Documents
def load_pdf_as_documents(path: str) -> List[Document]:
    """Extract text from PDF and create documents with page metadata."""
    raw = extract_text(path) or ""
    pages = [p.strip() for p in raw.split("\f") if p.strip()]
    docs = []
    for i, page in enumerate(pages, start=1):
        docs.append(Document(page_content=page, metadata={"source": path, "page": i}))
    return docs

# Chunking
def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)

# Build / Load Vector Store
def get_vectorstore(chunks: List[Document], embeddings: Embeddings, index_dir: str) -> FAISS:
    """
    Build or load a FAISS vector store.

    Args:
        chunks: Document chunks to index
        embeddings: Embeddings model
        index_dir: Directory path for the index

    Returns:
        FAISS vector store
    """
    if Path(index_dir).exists():
        db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        logger.info(f"Loaded existing index from {index_dir}")
    else:
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(index_dir)
        logger.info(f"Created new index at {index_dir} with {len(chunks)} chunks")
    return db


# Retrieval
def retrieve_docs(db: FAISS, query: str, k: int = 5) -> List[Document]:
    """
    Retrieve relevant documents from vector database with logging.

    Args:
        db: FAISS vector store
        query: Search query
        k: Number of documents to retrieve

    Returns:
        List of relevant documents
    """
    try:
        logger.debug(f"Retrieving {k} documents for query: '{query[:100]}...'")
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5}
        )
        docs = retriever.get_relevant_documents(query)
        return docs
    except Exception as e:
        logger.error(f"Document retrieval failed: {type(e).__name__} - {str(e)}")
        raise  # Re-raise to be caught by calling function

def format_context(docs: List[Document], max_chars_per_chunk: int = 900) -> str:
    """
    Format retrieved documents for the AI prompt (without page references).
    Page references are logged separately for debugging.
    """
    blocks = []
    for d in docs:
        page = d.metadata.get("page", "?")
        text = d.page_content.strip().replace("\n", " ")
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "…"

        # Log page reference for debugging
        logger.debug(f"Retrieved chunk from page {page}: {text[:100]}...")

        # Add only text (no page reference) for AI
        blocks.append(text)

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


# Generate Answer
def generate_outfit_advice(client, base_prompt: str, docs: list, temperature: float = 0.7, safety_settings: list = None):
    """
    Take the stylist prompt built in wardrobe_app.generate_outfit,
    append retrieved style tips from docs,
    and stream response from Gemini model.

    Args:
        client: Gemini client instance
        base_prompt: The stylist prompt with user requirements
        docs: Retrieved documents for context
        temperature: Generation temperature (default 0.7)
        safety_settings: Optional list of SafetySetting objects

    Yields:
        Chunks of text as they are generated
    """
    context = format_context(docs)
    citations = format_citations(docs)

    full_prompt = f"""{base_prompt}

            Retrieved Style Tips:
            {context}
            """

    # Build config
    config_dict = {"temperature": temperature}
    if safety_settings:
        config_dict["safety_settings"] = safety_settings

    try:
        log_api_call(logger, "Gemini API", "generate_outfit_advice", {"temperature": temperature})

        stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=[{"text": full_prompt}],
            config=types.GenerateContentConfig(**config_dict)
        )

        chunk_count = 0
        for chunk in stream:
            if chunk.candidates and chunk.candidates[0].content:
                if chunk.candidates[0].content.parts:
                    part = chunk.candidates[0].content.parts[0].text
                    if part:
                        chunk_count += 1
                        yield part

        log_api_success(logger, "Gemini API", f"Generated {chunk_count} chunks")

    except Exception as e:
        log_api_error(logger, "Gemini API", e)
        logger.error(f"Outfit advice generation failed: {str(e)}", exc_info=True)
        raise  # Re-raise to be caught by calling function

