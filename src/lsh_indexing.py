import json
import os
import pickle
import re
import hashlib

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


CHUNKS_FILE     = os.path.join(os.path.dirname(__file__), "..", "data", "chunks", "chunks.json")
INDICES_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "indices")
LSH_INDEX_FILE  = os.path.join(INDICES_DIR, "minhash_lsh_index.pkl")
MINHASH_FILE    = os.path.join(INDICES_DIR, "minhash_objects.pkl")
SIMHASH_FILE    = os.path.join(INDICES_DIR, "simhash_fingerprints.pkl")
SHINGLES_FILE   = os.path.join(INDICES_DIR, "chunk_shingles.pkl")
TOKENS_FILE     = os.path.join(INDICES_DIR, "chunk_tokens.pkl")

# minhash params
NUM_PERM  = 128
THRESHOLD = 0.2

# simhash bit width
SIMHASH_BITS = 64

stemmer = PorterStemmer()
PRESERVED_POLICY_WORDS = {"not", "no", "nor", "must", "shall", "may"}
LSH_STOPWORDS = ENGLISH_STOP_WORDS - PRESERVED_POLICY_WORDS


def clean_string(text):
    return " ".join(clean_tokens(text))


def clean_tokens(text):
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)

    normalized = []
    for token in text.split():
        if "-" in token:
            parts = [p for p in token.split("-") if p]
            if not parts:
                continue
            # Preserve meaning for both split and joined forms: "credit-hour" ->
            # ["credit", "hour", "credithour"]
            normalized.extend(parts)
            normalized.append("".join(parts))
        else:
            normalized.append(token)

    filtered = []
    for token in normalized:
        if len(token) <= 1:
            continue
        if token in LSH_STOPWORDS:
            continue
        filtered.append(stemmer.stem(token))
    return filtered


def make_shingles(tokens):
    """Word-level unigrams and bigrams to capture exact tokens and phrases."""
    shingles = set(tokens)
    for i in range(len(tokens) - 1):
        shingles.add(f"{tokens[i]} {tokens[i+1]}")
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
    chunk_tokens    = {}   # chunk_id → list of normalized tokens

    for chunk in tqdm(chunks, desc="  indexing", unit="chunk"):
        cid       = chunk["chunk_id"]
        tokens    = clean_tokens(chunk["text"])
        shingles  = make_shingles(tokens)

        mh = compute_minhash(shingles)
        sh = compute_simhash(tokens)

        lsh_index.insert(str(cid), mh)
        minhash_objects[cid] = mh
        simhash_fps[cid]     = sh
        chunk_shingles[cid]  = shingles
        chunk_tokens[cid]    = tokens

    os.makedirs(INDICES_DIR, exist_ok=True)

    with open(LSH_INDEX_FILE, "wb") as f:
        pickle.dump(lsh_index, f)
    with open(MINHASH_FILE, "wb") as f:
        pickle.dump(minhash_objects, f)
    with open(SIMHASH_FILE, "wb") as f:
        pickle.dump(simhash_fps, f)
    with open(SHINGLES_FILE, "wb") as f:
        pickle.dump(chunk_shingles, f)
    with open(TOKENS_FILE, "wb") as f:
        pickle.dump(chunk_tokens, f)

    print(f"done — indices saved to {INDICES_DIR}")
    print(f"  lsh bands/rows: b={lsh_index.b}, r={lsh_index.r}")
