#!/usr/bin/env python3
"""
Convert Sustainable Fashion Q&A Dataset (CSV) to text format for RAG.

The dataset has columns: instruction (question), response (answer)
This script converts it to a formatted text file that RAG can process.

Run: python scripts/convert_qa_dataset.py
"""

import pandas as pd
from pathlib import Path
import sys

# Paths
DATASET_PATH = Path("sustainable_fashion.csv")  # User downloads this from Kaggle
OUTPUT_DIR = Path("open_fashion_texts")
OUTPUT_FILE = OUTPUT_DIR / "Sustainable_Fashion_QA_Dataset.txt"

def convert_qa_to_text():
    """Convert Q&A dataset to formatted text."""

    print("=" * 70)
    print("CONVERTING SUSTAINABLE FASHION Q&A DATASET")
    print("=" * 70)

    # Check if dataset exists
    if not DATASET_PATH.exists():
        print(f"\n✗ Error: Dataset not found at {DATASET_PATH}")
        print("\n📥 Please download the dataset first:")
        print("   1. Go to: https://www.kaggle.com/datasets/tiyabk/sustainable-fashion")
        print("   2. Click 'Download' button (requires Kaggle account)")
        print("   3. Save as: sustainable-fashion.csv")
        print(f"   4. Place in: {Path.cwd()}")
        print("\n   Then run this script again.")
        sys.exit(1)

    # Load dataset
    try:
        print(f"\n📂 Loading dataset from: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        print(f"✓ Loaded {len(df)} Q&A pairs")
    except Exception as e:
        print(f"\n✗ Error loading CSV: {e}")
        sys.exit(1)

    # Check required columns
    if 'instruction' not in df.columns or 'response' not in df.columns:
        print(f"\n✗ Error: Expected columns 'instruction' and 'response'")
        print(f"   Found columns: {list(df.columns)}")
        sys.exit(1)

    # Create formatted text
    print(f"\n📝 Converting to text format...")

    text_content = """SUSTAINABLE FASHION Q&A KNOWLEDGE BASE
========================================================================

This dataset contains questions and answers about sustainable fashion,
outfit composition, styling advice, and conscious consumption.

Source: ktiyab (2023), Sustainable Fashion Q&A Dataset, Kaggle
License: CC BY 4.0
Link: https://www.kaggle.com/datasets/tiyabk/sustainable-fashion

Topics covered:
- Sustainable fashion principles and practices
- Outfit composition and styling advice
- Neutral color palettes and coordination
- High-quality fabric selection
- Seasonal layering and appropriate clothing choices
- Accessory matching and coordination
- Occasion-appropriate attire
- Budget-conscious sustainable shopping

========================================================================

"""

    # Add each Q&A pair
    for idx, row in df.iterrows():
        question = str(row['instruction']).strip()
        answer = str(row['response']).strip()

        # Skip empty entries
        if not question or not answer or question == 'nan' or answer == 'nan':
            continue

        text_content += f"\nQ{idx + 1}: {question}\n\n"
        text_content += f"A: {answer}\n\n"
        text_content += "-" * 70 + "\n"

    # Write to file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(text_content)
        print(f"✓ Converted {len(df)} Q&A pairs to text format")
        print(f"✓ Saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"\n✗ Error writing file: {e}")
        sys.exit(1)

    # Stats
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"📊 Total Q&A pairs: {len(df)}")
    print(f"📄 Output file: {OUTPUT_FILE}")
    print(f"💾 File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

    print("\n✅ Next step: Run python scripts/build_fashion_theory_index.py")
    print("   This will rebuild the RAG index with the new Q&A content.")

if __name__ == "__main__":
    try:
        convert_qa_to_text()
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
