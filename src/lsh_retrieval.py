import os
import sys
import pickle

import numpy as np

sys.path.append(os.path.dirname(__file__))
from lsh_indexing import clean_string, clean_tokens, make_shingles, compute_minhash, compute_simhash, NUM_PERM, SIMHASH_BITS


INDICES_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "indices")
LSH_INDEX_FILE = os.path.join(INDICES_DIR, "minhash_lsh_index.pkl")
MINHASH_FILE   = os.path.join(INDICES_DIR, "minhash_objects.pkl")
SIMHASH_FILE   = os.path.join(INDICES_DIR, "simhash_fingerprints.pkl")
SHINGLES_FILE  = os.path.join(INDICES_DIR, "chunk_shingles.pkl")
CHUNKS_CACHE   = os.path.join(INDICES_DIR, "chunks_cache.pkl")


def hamming(a, b):
    """count differing bits between two ints"""
    return bin(a ^ b).count('1')


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def load_lsh_index():
    with open(LSH_INDEX_FILE, "rb") as f:
        lsh_index = pickle.load(f)
    with open(MINHASH_FILE, "rb") as f:
        minhash_objects = pickle.load(f)
    with open(SIMHASH_FILE, "rb") as f:
        simhash_fps = pickle.load(f)
    with open(SHINGLES_FILE, "rb") as f:
        chunk_shingles = pickle.load(f)
    with open(CHUNKS_CACHE, "rb") as f:
        chunks = pickle.load(f)
    return lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunks


def expand_query(query):
    """Simple synonym expansion for known NUST vocabulary mismatch"""
    query = query.lower()
    expansions = {
        "gpa": "cgpa sgpa",
        "fail": "f grade academic deficiency withdrawn",
        "attendance": "absent participation",
        "repeat": "clear grade",
    }
    for k, v in expansions.items():
        if k in query:
            query += " " + v
    return query


def search_minhash(query, lsh_index, minhash_objects, chunk_shingles, chunks, top_k=5):
    """query the lsh index → get candidates → re-rank by exact jaccard"""
    query = expand_query(query)
    text_c   = clean_string(query)
    shingles = make_shingles(text_c)
    qmh      = compute_minhash(shingles)

    candidate_ids = lsh_index.query(qmh)

    if not candidate_ids:
        # lsh returned nothing — fall back to top-5 by minhash estimated jaccard over all chunks
        scores = [
            (cid, minhash_objects[cid].jaccard(qmh))
            for cid in minhash_objects
        ]
        candidate_ids = [str(cid) for cid, _ in sorted(scores, key=lambda x: -x[1])[:top_k * 3]]

    # re-rank candidates by exact jaccard on shingle sets
    ranked = []
    for cid_str in candidate_ids:
        cid = int(cid_str)
        j   = jaccard(shingles, chunk_shingles[cid])
        ranked.append((cid, j))

    ranked.sort(key=lambda x: -x[1])
    top = ranked[:top_k]

    chunk_map = {c["chunk_id"]: c for c in chunks}
    results = []
    for cid, score in top:
        c = chunk_map[cid]
        results.append({
            "chunk_id": cid,
            "source":   c["source"],
            "page":     c["page"],
            "score":    round(score, 4),
            "text":     c["text"],
            "method":   "minhash_lsh",
        })
    return results


def search_simhash(query, simhash_fps, chunks, threshold=10, top_k=5):
    """linear scan — find chunks whose simhash fingerprint is within hamming threshold of query"""
    query = expand_query(query)
    tokens = clean_tokens(query)
    q_fp   = compute_simhash(tokens)

    distances = []
    for cid, fp in simhash_fps.items():
        d = hamming(q_fp, fp)
        distances.append((cid, d))

    # sort by hamming distance (lower = more similar)
    distances.sort(key=lambda x: x[1])
    top = [x for x in distances if x[1] <= threshold][:top_k]

    # if nothing is within threshold, just take the closest ones anyway
    if not top:
        top = distances[:top_k]

    chunk_map = {c["chunk_id"]: c for c in chunks}
    results = []
    for cid, dist in top:
        c = chunk_map[cid]
        results.append({
            "chunk_id": cid,
            "source":   c["source"],
            "page":     c["page"],
            "score":    dist,   # hamming distance — lower is better
            "text":     c["text"],
            "method":   "simhash",
        })
    return results


def hybrid_search(query, lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunks, top_k=5):
    """
    union of minhash + simhash candidates, then re-rank all of them by exact jaccard.
    gives the best of both worlds.
    """
    query = expand_query(query)
    text_c   = clean_string(query)
    tokens   = text_c.split()
    shingles = make_shingles(text_c)
    
    qmh      = compute_minhash(shingles)
    q_fp     = compute_simhash(tokens)

    # minhash candidates
    mh_candidates = set(int(x) for x in lsh_index.query(qmh))

    # simhash candidates — take closest 20 by hamming distance
    sh_distances  = sorted(simhash_fps.items(), key=lambda x: hamming(q_fp, x[1]))
    sh_candidates = set(cid for cid, _ in sh_distances[:30])

    all_candidates = mh_candidates | sh_candidates

    # if still too few, top up with minhash estimated jaccard over full index
    if len(all_candidates) < top_k:
        extra = sorted(minhash_objects.items(), key=lambda x: -x[1].jaccard(qmh))
        for cid, _ in extra[:top_k * 3]:
            all_candidates.add(cid)

    # re-rank by exact jaccard on shingles
    ranked = sorted(
        [(cid, jaccard(shingles, chunk_shingles[cid])) for cid in all_candidates],
        key=lambda x: -x[1]
    )

    chunk_map = {c["chunk_id"]: c for c in chunks}
    results = []
    for cid, score in ranked[:top_k]:
        c = chunk_map[cid]
        results.append({
            "chunk_id": cid,
            "source":   c["source"],
            "page":     c["page"],
            "score":    round(score, 4),
            "text":     c["text"],
            "method":   "hybrid",
        })
    return results


# ── run ───────────────────────────────────────────────────────────────────────

print("loading lsh index...")
lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunks = load_lsh_index()

test_queries = [
    "minimum GPA requirement",
    "what happens if a student fails a course",
    "attendance policy",
    "how many times can a course be repeated",
]

print("\n--- minhash lsh retrieval ---\n")
for q in test_queries:
    print(f"query: {q!r}")
    for r in search_minhash(q, lsh_index, minhash_objects, chunk_shingles, chunks, top_k=3):
        print(f"  [jaccard={r['score']}] {r['source']} p.{r['page']} — {r['text'][:110]}...")
    print()

print("\n--- simhash retrieval ---\n")
for q in test_queries:
    print(f"query: {q!r}")
    for r in search_simhash(q, simhash_fps, chunks, threshold=10, top_k=3):
        print(f"  [hamming={r['score']}] {r['source']} p.{r['page']} — {r['text'][:110]}...")
    print()

print("\n--- hybrid retrieval ---\n")
for q in test_queries:
    print(f"query: {q!r}")
    for r in hybrid_search(q, lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunks, top_k=3):
        print(f"  [jaccard={r['score']}] {r['source']} p.{r['page']} — {r['text'][:110]}...")
    print()
