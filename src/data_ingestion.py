import fitz
import re
import json
import os
from tqdm import tqdm


RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chunks")
OUTPUT_FILE = os.path.join(CHUNKS_DIR, "chunks.json")

# Chunk size follows project spec: 200–500 words per chunk
MIN_WORDS = 200
MAX_WORDS = 500
OVERLAP_WORDS = 100  # ~20% overlap for context carry-over between chunks



def read_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def clean(text):
    text = re.sub(r'\b(page\s*\d+|\d+\s*/\s*\d+)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)   # 3+ newlines → paragraph break
    text = re.sub(r'[ \t]+', ' ', text)       # horizontal whitespace only
    return text.strip()


def split_recursive(text):
    """
    Tries to break text at natural boundaries, from coarsest to finest.
    """
    separators = ['\n\n', '\n', '. ', ' ']
    pieces = []

    def split_piece(t):
        t = t.strip()
        if not t:
            return

        if len(t.split()) <= MAX_WORDS:
            pieces.append(t)
            return

        for sep in separators:
            if sep in t:
                parts = t.split(sep)
                parts = [p.strip() for p in parts if p.strip()]
                for p in parts:
                    split_piece(p)
                return

        # absolute fallback
        words = t.split()
        for i in range(0, len(words), MAX_WORDS):
            pieces.append(" ".join(words[i: i + MAX_WORDS]))

    split_piece(text)
    return pieces


def group_into_chunks(pieces, page_num, source, starting_id):
    """
    Packs pieces into chunks bounded by MIN_WORDS - MAX_WORDS.
    """
    chunks = []
    cid = starting_id
    current_words = []
    overlap_tail = []

    for piece in pieces:
        piece_words = piece.split()

        # if adding this piece would bust the max, try to flush first
        if len(current_words) + len(piece_words) > MAX_WORDS and current_words:
            if len(current_words) >= MIN_WORDS:
                # valid chunk — flush it, carry overlap into next
                text = " ".join(current_words)
                chunks.append({
                    "chunk_id": cid,
                    "source": source,
                    "page": page_num,
                    "word_count": len(current_words),
                    "text": text,
                })
                cid += 1
                overlap_tail = current_words[-OVERLAP_WORDS:]
                current_words = overlap_tail + piece_words
            else:
                # too short to flush — keep accumulating instead of resetting
                current_words += piece_words
        else:
            current_words += piece_words

    if current_words:
        if len(current_words) >= MIN_WORDS:
            chunks.append({
                "chunk_id": cid,
                "source": source,
                "page": page_num,
                "word_count": len(current_words),
                "text": " ".join(current_words),
            })
            cid += 1
        elif chunks:
            chunks[-1]["text"] += " " + " ".join(current_words)
            chunks[-1]["word_count"] = len(chunks[-1]["text"].split())

    return chunks, cid


def process_pdf(path, starting_id=0):
    name = os.path.basename(path)
    print(f"\nreading {name}...")

    pages = read_pdf(path)
    print(f"  got {len(pages)} pages with actual text")

    all_chunks = []
    cid = starting_id

    for p in tqdm(pages, desc="  chunking", unit="page"):
        cleaned = clean(p["text"])
        if not cleaned:
            continue
        pieces = split_recursive(cleaned)
        new_chunks, cid = group_into_chunks(pieces, p["page"], name, cid)
        all_chunks.extend(new_chunks)

    print(f"  {name} -> {len(all_chunks)} chunks")
    return all_chunks, cid


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    pdfs = sorted([
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.lower().endswith(".pdf")
    ])

    if not pdfs:
        print("no pdfs in data/raw/ — put the ug and pg handbooks there first")
    else:
        print(f"found {len(pdfs)} pdf(s): {[os.path.basename(p) for p in pdfs]}")

        all_chunks = []
        next_id = 0

        for pdf in pdfs:
            chunks, next_id = process_pdf(pdf, starting_id=next_id)
            all_chunks.extend(chunks)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"\nall done — {len(all_chunks)} chunks saved to {OUTPUT_FILE}")
