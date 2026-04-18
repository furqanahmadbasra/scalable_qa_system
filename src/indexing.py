import json
import os
import pickle
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


CHUNKS_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "chunks", "chunks.json")
INDICES_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "indices")
VEC_FILE     = os.path.join(INDICES_DIR, "tfidf_vectorizer.pkl")
MATRIX_FILE  = os.path.join(INDICES_DIR, "tfidf_matrix.pkl")
CHUNKS_CACHE = os.path.join(INDICES_DIR, "chunks_cache.pkl")


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index():
    with open(VEC_FILE, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MATRIX_FILE, "rb") as f:
        matrix = pickle.load(f)
    with open(CHUNKS_CACHE, "rb") as f:
        chunks = pickle.load(f)
    return vectorizer, matrix, chunks


# ── run ───────────────────────────────────────────────────────────────────────

print("loading chunks...")
chunks = load_chunks()
texts = [preprocess(c["text"]) for c in chunks]

print(f"fitting tfidf on {len(texts)} chunks...")

vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2),
    stop_words="english",
)

matrix = vectorizer.fit_transform(texts)
print(f"tfidf matrix shape: {matrix.shape}")

os.makedirs(INDICES_DIR, exist_ok=True)

with open(VEC_FILE, "wb") as f:
    pickle.dump(vectorizer, f)
with open(MATRIX_FILE, "wb") as f:
    pickle.dump(matrix, f)
with open(CHUNKS_CACHE, "wb") as f:
    pickle.dump(chunks, f)

print(f"saved vectorizer, matrix, and chunk cache to {INDICES_DIR}")
