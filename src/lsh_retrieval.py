import os
import sys
import pickle
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(__file__))
from lsh_indexing import clean_string, clean_tokens, make_shingles, compute_minhash, compute_simhash, NUM_PERM, SIMHASH_BITS
from indexing import preprocess_and_stem

INDICES_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "indices")
MINHASH_INDEX  = os.path.join(INDICES_DIR, "minhash_lsh_index.pkl")
MINHASH_FILE   = os.path.join(INDICES_DIR, "minhash_objects.pkl")
SIMHASH_FILE   = os.path.join(INDICES_DIR, "simhash_fingerprints.pkl")
SHINGLES_FILE  = os.path.join(INDICES_DIR, "chunk_shingles.pkl")
TOKENS_FILE    = os.path.join(INDICES_DIR, "chunk_tokens.pkl")
CHUNKS_FILE    = os.path.join(os.path.dirname(__file__), "..", "data", "chunks", "chunks.json")

stemmer = PorterStemmer()
FUSED_DEFAULT_TFIDF_WEIGHT = 0.50
FUSED_DEFAULT_HYBRID_WEIGHT = 0.50
FUSED_DEFAULT_INTENT_WEIGHT = 0.10
FUSED_DEFAULT_CANDIDATE_POOL = 50
SIMHASH_DEFAULT_THRESHOLD = 12
INTENT_KEYWORDS = {
    "attendance": {"attend", "short", "xf", "absent"},
    "probation": {"probat", "warn", "withdraw", "academ", "defici"},
    "repeat_course": {"repeat", "retak", "retest", "course"},
    "hostel": {"hostel", "accommod", "allot", "qalam", "resid"},
    "rechecking": {"recheck", "reassess", "exam", "paper", "annex"},
    "graduation": {"degre", "graduat", "award", "credit", "cgpa"},
    "fee_penalty": {"fee", "fine", "dues", "penalti", "deposit"},
    "credit_hours": {"credit", "hour", "semester", "cours"},
    "plagiarism": {"plagiar", "dishonesti", "cheat", "academ"},
    "gold_medal": {"medal", "convoc", "presid", "rector", "academ"},
}


def detect_query_intents(query_tokens):
    token_set = set(query_tokens)
    intents = set()
    for intent, kws in INTENT_KEYWORDS.items():
        if token_set & kws:
            intents.add(intent)
    return intents

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
    with open(TOKENS_FILE, "rb") as f:
        chunk_tokens = pickle.load(f)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunk_tokens, chunks


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
    tokens   = clean_tokens(query)
    shingles = make_shingles(tokens)
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

def search_simhash(query, simhash_fps, chunks, threshold=SIMHASH_DEFAULT_THRESHOLD, top_k=5):
    tokens = clean_tokens(query)
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

def hybrid_search(query, lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunk_tokens, chunks, top_k=5):
    tokens   = clean_tokens(query)
    shingles = make_shingles(tokens)
    
    qmh      = compute_minhash(shingles)
    q_fp     = compute_simhash(tokens)

    mh_candidates = set(int(x) for x in lsh_index.query(qmh))

    sh_distances  = sorted(simhash_fps.items(), key=lambda x: hamming(q_fp, x[1]))
    sh_candidates = set(cid for cid, _ in sh_distances[:50])

    all_candidates = mh_candidates | sh_candidates

    if len(all_candidates) < top_k:
        extra = sorted(minhash_objects.items(), key=lambda x: -x[1].jaccard(qmh))
        for cid, _ in extra[:top_k * 3]:
            all_candidates.add(cid)

    # Hybrid Scoring
    ranked = []
    q_token_set = set(tokens)
    for cid in all_candidates:
        # 1. Exact Jaccard on word-level shingles
        j_score = jaccard(shingles, chunk_shingles[cid])
        m_score = minhash_objects[cid].jaccard(qmh)

        # 2. Normalised SimHash
        c_fp    = simhash_fps[cid]
        h_dist  = hamming(q_fp, c_fp)
        h_score = max(0.0, 1.0 - (2.0 * h_dist / SIMHASH_BITS))

        c_token_set = set(chunk_tokens.get(cid, []))
        token_overlap = len(q_token_set & c_token_set) / max(1, len(q_token_set))

        # Combined score: overlap-first ranking for short policy queries.
        hybrid_score = (
            (0.45 * j_score)
            + (0.20 * m_score)
            + (0.20 * h_score)
            + (0.15 * token_overlap)
        )
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


def fused_search(
    query,
    lsh_index,
    minhash_objects,
    simhash_fps,
    chunk_shingles,
    chunk_tokens,
    chunks,
    vectorizer,
    matrix,
    top_k=5,
    candidate_pool=FUSED_DEFAULT_CANDIDATE_POOL,
    tfidf_weight=FUSED_DEFAULT_TFIDF_WEIGHT,
    hybrid_weight=FUSED_DEFAULT_HYBRID_WEIGHT,
    intent_weight=FUSED_DEFAULT_INTENT_WEIGHT,
):
    """
    Candidate generation with LSH-hybrid, then rerank those candidates using TF-IDF cosine.
    This keeps approximate retrieval scalable while improving lexical relevance.
    """
    # Reuse hybrid scoring to produce candidates with rich similarity signals.
    hybrid_candidates = hybrid_search(
        query,
        lsh_index,
        minhash_objects,
        simhash_fps,
        chunk_shingles,
        chunk_tokens,
        chunks,
        top_k=candidate_pool,
    )
    if not hybrid_candidates:
        return []

    cid_to_hybrid = {r["chunk_id"]: r["score"] for r in hybrid_candidates}
    candidate_ids = [r["chunk_id"] for r in hybrid_candidates]

    # Compute lexical score only on candidate set.
    q_vec = vectorizer.transform([preprocess_and_stem(query)])
    sim_all = cosine_similarity(q_vec, matrix).flatten()
    max_sim = max(float(sim_all[cid]) for cid in candidate_ids) if candidate_ids else 1.0
    max_sim = max(max_sim, 1e-9)

    ranked = []
    chunk_map = {c["chunk_id"]: c for c in chunks}
    q_tokens = clean_tokens(query)
    query_intents = detect_query_intents(q_tokens)
    for cid in candidate_ids:
        tfidf_score = float(sim_all[cid]) / max_sim
        hybrid_score = float(cid_to_hybrid.get(cid, 0.0))
        chunk_token_set = set(chunk_tokens.get(cid, []))

        intent_hits = 0
        for intent in query_intents:
            if chunk_token_set & INTENT_KEYWORDS[intent]:
                intent_hits += 1
        intent_score = (
            (intent_hits / len(query_intents)) if query_intents else 0.0
        )

        fused_score = (
            (tfidf_weight * tfidf_score)
            + (hybrid_weight * hybrid_score)
            + (intent_weight * intent_score)
        )
        ranked.append((cid, fused_score, tfidf_score, hybrid_score, intent_score))

    ranked.sort(key=lambda x: -x[1])
    results = []
    for cid, fused_score, tfidf_score, hybrid_score, intent_score in ranked[:top_k]:
        c = chunk_map[cid]
        results.append({
            "chunk_id": cid,
            "source": c["source"],
            "page": c["page"],
            "score": round(fused_score, 4),
            "text": c["text"],
            "method": "fused_lsh_tfidf",
            "tfidf_score": round(tfidf_score, 4),
            "hybrid_score": round(hybrid_score, 4),
            "intent_score": round(intent_score, 4),
        })
    return results

if __name__ == "__main__":
    print("loading lsh index...")
    lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunk_tokens, chunks = load_lsh_index()
    
    test_queries = [
        "minimum GPA requirement",
        "what happens if a student fails a course",
        "attendance policy",
        "how many times can a course be repeated",
    ]

    print("\n--- hybrid retrieval test ---")
    for q in test_queries:
        print(f"\nquery: '{q}'")
        res = hybrid_search(q, lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunk_tokens, chunks, top_k=3)
        for r in res:
            print(f"  [score={r['score']}] {r['source']} p.{r['page']} — {r['text'][:100]}...")
