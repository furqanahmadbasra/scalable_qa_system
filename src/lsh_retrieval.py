import os
import sys
import pickle
import numpy as np
from nltk.stem import PorterStemmer

sys.path.append(os.path.dirname(__file__))
from lsh_indexing import clean_string, clean_tokens, make_shingles, compute_minhash, compute_simhash, NUM_PERM, SIMHASH_BITS

INDICES_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "indices")
MINHASH_INDEX  = os.path.join(INDICES_DIR, "minhash_lsh_index.pkl")
MINHASH_FILE   = os.path.join(INDICES_DIR, "minhash_objects.pkl")
SIMHASH_FILE   = os.path.join(INDICES_DIR, "simhash_fingerprints.pkl")
SHINGLES_FILE  = os.path.join(INDICES_DIR, "chunk_shingles.pkl")
CHUNKS_FILE    = os.path.join(os.path.dirname(__file__), "..", "data", "chunks", "chunks.json")

stemmer = PorterStemmer()

def load_lsh_index():
    import json
    with open(MINHASH_INDEX, "rb") as f:
        lsh_index = pickle.load(f)
    with open(MINHASH_FILE, "rb") as f:
        minhash_objects = pickle.load(f)
    with open(SIMHASH_FILE, "rb") as f:
        simhash_fps = pickle.load(f)
    with open(SHINGLES_FILE, "rb") as f:
        chunk_shingles = pickle.load(f)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunks

def expand_query(query):
    """Expanded synonym dictionary for high-value NUST terms."""
    query = query.lower()
    expansions = {
        "gpa": "cgpa sgpa grade point average cumulative",
        "fail": "f grade academic deficiency withdrawn repeat repetition retake clear improve",
        "attendance": "absent participation present 75%",
        "repeat": "repetition retake clear improve",
        "graduation": "degree award complete requirement pass",
    }
    for k, v in expansions.items():
        if k in query:
            query += " " + v
    return query

def jaccard(s1, s2):
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def hamming(f1, f2, bits=SIMHASH_BITS):
    x = f1 ^ f2
    dist = 0
    while x:
        dist += 1
        x &= x - 1
    return dist

def search_minhash(query, lsh_index, minhash_objects, chunk_shingles, chunks, top_k=5):
    query = expand_query(query)
    text_c   = clean_string(query)
    shingles = make_shingles(text_c)
    qmh      = compute_minhash(shingles)

    candidate_ids = lsh_index.query(qmh)

    if not candidate_ids:
        scores = [
            (cid, minhash_objects[cid].jaccard(qmh))
            for cid in minhash_objects
        ]
        candidate_ids = [str(cid) for cid, _ in sorted(scores, key=lambda x: -x[1])[:top_k * 3]]

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

def search_simhash(query, simhash_fps, chunks, threshold=15, top_k=5):
    query = expand_query(query)
    text_c = clean_string(query)
    tokens = text_c.split()
    q_fp   = compute_simhash(tokens)

    distances = []
    for cid, fp in simhash_fps.items():
        d = hamming(q_fp, fp)
        distances.append((cid, d))

    distances.sort(key=lambda x: x[1])
    top = [x for x in distances if x[1] <= threshold][:top_k]

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
            "score":    dist,
            "text":     c["text"],
            "method":   "simhash",
        })
    return results

def hybrid_search(query, lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunks, top_k=5):
    query = expand_query(query)
    text_c   = clean_string(query)
    tokens   = text_c.split()
    shingles = make_shingles(text_c)
    
    qmh      = compute_minhash(shingles)
    q_fp     = compute_simhash(tokens)

    mh_candidates = set(int(x) for x in lsh_index.query(qmh))

    sh_distances  = sorted(simhash_fps.items(), key=lambda x: hamming(q_fp, x[1]))
    sh_candidates = set(cid for cid, _ in sh_distances[:30])

    all_candidates = mh_candidates | sh_candidates

    if len(all_candidates) < top_k:
        extra = sorted(minhash_objects.items(), key=lambda x: -x[1].jaccard(qmh))
        for cid, _ in extra[:top_k * 3]:
            all_candidates.add(cid)

    # Hybrid Scoring
    ranked = []
    for cid in all_candidates:
        # 1. Exact Jaccard
        j_score = jaccard(shingles, chunk_shingles[cid])
        
        # 2. Normalized Simhash (1 - dist/64)
        c_fp = simhash_fps[cid]
        h_dist = hamming(q_fp, c_fp)
        h_score = max(0, 1.0 - (h_dist / SIMHASH_BITS))
        
        # Combined weight (50% Jaccard, 50% Hamming)
        hybrid_score = (0.5 * j_score) + (0.5 * h_score)
        ranked.append((cid, hybrid_score))

    ranked.sort(key=lambda x: -x[1])

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

if __name__ == "__main__":
    print("loading lsh index...")
    lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunks = load_lsh_index()
    
    test_queries = [
        "minimum GPA requirement",
        "what happens if a student fails a course",
        "attendance policy",
        "how many times can a course be repeated",
    ]

    print("\n--- hybrid retrieval test ---")
    for q in test_queries:
        print(f"\nquery: '{q}'")
        res = hybrid_search(q, lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunks, top_k=3)
        for r in res:
            print(f"  [score={r['score']}] {r['source']} p.{r['page']} — {r['text'][:100]}...")
