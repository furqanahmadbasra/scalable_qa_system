import json
import os
import pickle
import re
import hashlib

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from nltk.stem import PorterStemmer


CHUNKS_FILE     = os.path.join(os.path.dirname(__file__), "..", "data", "chunks", "chunks.json")
INDICES_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "indices")
LSH_INDEX_FILE  = os.path.join(INDICES_DIR, "minhash_lsh_index.pkl")
MINHASH_FILE    = os.path.join(INDICES_DIR, "minhash_objects.pkl")
SIMHASH_FILE    = os.path.join(INDICES_DIR, "simhash_fingerprints.pkl")
SHINGLES_FILE   = os.path.join(INDICES_DIR, "chunk_shingles.pkl")

# minhash params — 128 hash functions, threshold 0.01 (extremely low because short queries have tiny Jaccard against long chunks)
NUM_PERM  = 128
THRESHOLD = 0.01

# simhash bit width
SIMHASH_BITS = 64

stemmer = PorterStemmer()


def clean_string(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    stemmed = [stemmer.stem(t) for t in tokens]
    return " ".join(stemmed)


def clean_tokens(text):
    return clean_string(text).split()


def make_shingles(text):
    """Word-level unigrams and bigrams to capture exact tokens and phrases."""
    words = text.split()
    shingles = set(words)  # Add all unigrams
    for i in range(len(words) - 1):
        shingles.add(f"{words[i]} {words[i+1]}")  # Add bigrams
    return shingles


def compute_minhash(shingles, num_perm=NUM_PERM):
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def compute_simhash(tokens, num_bits=SIMHASH_BITS):
    """
    manual simhash — for each token, hash it and vote +1/-1 per bit.
    final fingerprint: bit is 1 if vote sum > 0, else 0.
    stored as a plain python int.
    """
    v = [0] * num_bits
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(num_bits):
            if (h >> i) & 1:
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(num_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("loading chunks...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"building minhash + simhash for {len(chunks)} chunks...")

    lsh_index       = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    minhash_objects = {}   # chunk_id → MinHash object
    simhash_fps     = {}   # chunk_id → int fingerprint
    chunk_shingles  = {}   # chunk_id → set of shingles (for jaccard re-ranking later)

    for chunk in tqdm(chunks, desc="  indexing", unit="chunk"):
        cid       = chunk["chunk_id"]
        text_c    = clean_string(chunk["text"])
        tokens    = text_c.split()
        shingles  = make_shingles(text_c)

        mh = compute_minhash(shingles)
        sh = compute_simhash(tokens)

        lsh_index.insert(str(cid), mh)
        minhash_objects[cid] = mh
        simhash_fps[cid]     = sh
        chunk_shingles[cid]  = shingles

    os.makedirs(INDICES_DIR, exist_ok=True)

    with open(LSH_INDEX_FILE, "wb") as f:
        pickle.dump(lsh_index, f)
    with open(MINHASH_FILE, "wb") as f:
        pickle.dump(minhash_objects, f)
    with open(SIMHASH_FILE, "wb") as f:
        pickle.dump(simhash_fps, f)
    with open(SHINGLES_FILE, "wb") as f:
        pickle.dump(chunk_shingles, f)

    print(f"done — indices saved to {INDICES_DIR}")
    print(f"  lsh bands/rows: b={lsh_index.b}, r={lsh_index.r}")
