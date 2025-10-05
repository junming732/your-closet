# 👗 Your Closet — Group Project

This repository contains our team’s work for the “Your Closet” assignment.

---

## Original Team Work (Unmodified)

To keep this repo clear for reviewers, **all teammates’ raw contributions are preserved unchanged** in
[`original_contributions/`](original_contributions/).

All integrated code now lives in the [`src/`](src/) directory.

---

## Installation & Setup

Clone the repo and run the following in your terminal:

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install dependencies
pip install -r requirements.txt
```

Make sure you create a `.env` file in the project root with your Google Gemini API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## Building & Updating the Fashion Theory RAG Index

The **Fashion Theory Q&A** feature uses a small set of public-domain fashion texts stored in the `open_fashion_texts/` folder.
Before running the app for the first time (or after adding/removing documents), you must (re)build the FAISS index:

```bash
# from the project root
python -m src.scripts.build_fashion_theory_index
```

This script will:

- Load all PDFs and TXT files from `open_fashion_texts/`
- Chunk and embed them with Gemini
- Save the FAISS index to `faiss_index/fashion_theory/`

YOnly need to run this script again when you add, remove, or update documents.
The Gradio app then automatically loads the updated index at startup:

```bash
python -m src.app.main
```

Gradio will start on **http://localhost:7861**.

- **Wardrobe Management** – Add, edit, delete, upload, or export your clothing items.
- **Build Outfit** – Combines your wardrobe items with style tips retrieved from our PDF knowledge base using Gemini.
- **Chat with Stylist** – Ask free-form styling questions, powered by Fashion Theory RAG index.

---

## File Structure

```
original_contributions/       # untouched team files
src/
  app/
    main.py                   # unified Gradio app
  retrieval/
    gemini_rag.py             # Gemini embeddings & outfit-advice RAG
    fashion_theory_rag.py      # Self-contained Fashion Theory RAG module
  scripts/
    build_fashion_theory_index.py  # Build FAISS index from open_fashion_texts/
open_fashion_texts/          # Place public-domain PDFs/TXT here
faiss_index/fashion_theory/   # Saved FAISS index
requirements.txt
.env.example                  # sample environment variables
```

---

## Contributions

- **Maria** – Wardrobe management & original Gradio UI
- **Dafne** – Gemini retrieval-augmented generation (RAG) pipeline for outfits
- **Junming** – Integration, refactor, scraping, fashion theory RAG
