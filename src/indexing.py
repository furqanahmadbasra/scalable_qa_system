import json
import os
import pickle
import re

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

stemmer = PorterStemmer()


CHUNKS_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "chunks", "chunks.json")
INDICES_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "indices")
VEC_FILE     = os.path.join(INDICES_DIR, "tfidf_vectorizer.pkl")
MATRIX_FILE  = os.path.join(INDICES_DIR, "tfidf_matrix.pkl")
CHUNKS_CACHE = os.path.join(INDICES_DIR, "chunks_cache.pkl")


def preprocess(text):
    """Basic clean: lowercase + remove non-alphanumeric."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_and_stem(text):
    """
    Cleans, removes stop words, then stems each token.
    Done BEFORE TF-IDF to avoid pickling issues with custom analyzers.
    """
    tokens = preprocess(text).split()
    return " ".join([
        stemmer.stem(t)
        for t in tokens
        if t and t not in ENGLISH_STOP_WORDS
    ])



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

if __name__ == "__main__":
    print("loading chunks...")
    chunks = load_chunks()
    texts = [preprocess_and_stem(c["text"]) for c in chunks]

    print(f"fitting tfidf on {len(texts)} chunks...")

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
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
