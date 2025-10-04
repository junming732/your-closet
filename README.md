# 👗 Gemini Closet RAG — Group Project

This repository contains our team’s work for the “Fashion Stylist Assistant” assignment.

---

## Original Team Work (Unmodified)

To keep this repo clear for reviewers, **all teammates’ raw contributions are preserved unchanged** in
[`original_contributions/`](original_contributions/).
This folder includes Maria’s Gradio wardrobe app, Dafne’s Gemini RAG notebook, and all uploaded PDFs.

All production-ready, integrated code now lives in the [`src/`](src/) directory.
Our `src` code uses Maria’s wardrobe logic + Dafne’s retrieval logic, with a unified UI.

---

## Installation & Setup

Clone the repo and run the following in your terminal:

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

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

## Running the App

From the project root:

```bash
python -m src.app.main
```

Gradio will start on **http://localhost:7861** .

---

## How It Works

- **Wardrobe Management** – Add, edit, delete, upload, or export your clothing items.
- **Build Outfit** – Combines your wardrobe items with style tips retrieved from our PDF knowledge base using Gemini.
- **Chat with Stylist** – Ask free-form questions about what to wear.


---

## File Structure

```
original_contributions/    # untouched team files
src/
  app/
    main.py                # unified Gradio app
    wardrobe_app.py        # wardrobe logic
    ui_config.py           # theme, CSS, dropdown choices
  retrieval/
    gemini_rag.py          # Gemini embeddings, retrieval, and outfit-advice
requirements.txt
.env.example               # sample environment variables
```

---

## Contributions

- **Maria** – Wardrobe management & original Gradio UI
- **Dafne** – Gemini retrieval-augmented generation (RAG) pipeline
- **Junming** – Integration, refactor, scraping
