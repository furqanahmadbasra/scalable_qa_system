import os
import sys
import time
import json
import pickle
import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import from existing modules
from indexing import preprocess_and_stem
from lsh_indexing import clean_string, make_shingles, compute_minhash, compute_simhash
from lsh_retrieval import jaccard, hamming, search_simhash, search_minhash, hybrid_search
from answer_generation import generate_answer

CHUNKS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "chunks", "chunks.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
REPORT_FILE = os.path.join(RESULTS_DIR, "comprehensive_report.txt")

# 15 Sample Queries for robust evaluation
TEST_QUERIES = [
    "What is the minimum GPA requirement?",
    "What happens if a student fails a course?",
    "What is the attendance policy?",
    "How many times can a course be repeated?",
    "What is the policy on plagiarism?",
    "How are final grades calculated?",
    "Can a student freeze a semester?",
    "What are the requirements for graduation?",
    "Is there a penalty for late fee submission?",
    "What is the maximum credit hour limit?",
    "How to apply for a hostel?",
    "What is the policy for dropping a course?",
    "Who is eligible for a gold medal?",
    "What are the rules for exam re-checking?",
    "How is a student placed on academic probation?"
]

def log_print(msg, f):
    """Prints to console and writes to log file."""
    print(msg)
    f.write(msg + "\n")

def get_memory_footprint(obj):
    """Helper to get actual memory size by pickling."""
    return len(pickle.dumps(obj)) / (1024 * 1024)  # Return in MB

def get_exact_jaccard_ground_truth(query, chunks, top_k=10):
    """Computes EXACT Jaccard similarity across ALL chunks to serve as ground truth for Recall."""
    q_shingles = make_shingles(clean_string(query))
    scores = []
    for c in chunks:
        c_shingles = make_shingles(clean_string(c['text']))
        scores.append((c['chunk_id'], jaccard(q_shingles, c_shingles)))
    scores.sort(key=lambda x: -x[1])
    return [cid for cid, score in scores[:top_k]]

def run_experiments():
    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        log_print("="*60, report)
        log_print(" SCALABLE QA SYSTEM - COMPREHENSIVE EXPERIMENTAL ANALYSIS", report)
        log_print("="*60 + "\n", report)

        # Load base chunks
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            base_chunks = json.load(f)
        log_print(f"Loaded {len(base_chunks)} chunks for baseline testing.\n", report)

        # ---------------------------------------------------------------------
        # 1. EXACT VS APPROXIMATE RETRIEVAL (Time, Memory, Accuracy)
        # ---------------------------------------------------------------------
        log_print(">>> 1. EXACT (TF-IDF) VS APPROXIMATE (LSH) RETRIEVAL", report)
        
        # Build TF-IDF
        start_time = time.time()
        texts = [preprocess_and_stem(c["text"]) for c in base_chunks]
        vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.95, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        tfidf_build_time = time.time() - start_time
        tfidf_memory = get_memory_footprint(vectorizer) + get_memory_footprint(tfidf_matrix)

        # Build LSH (Baseline parameters)
        start_time = time.time()
        lsh_index = MinHashLSH(threshold=0.01, num_perm=128)
        minhash_objs, simhash_fps, chunk_shingles = {}, {}, {}
        for c in base_chunks:
            cid = c["chunk_id"]
            text_c = clean_string(c["text"])
            sh = make_shingles(text_c)
            mh = compute_minhash(sh, num_perm=128)
            lsh_index.insert(str(cid), mh)
            minhash_objs[cid] = mh
            simhash_fps[cid] = compute_simhash(text_c.split(), num_bits=64)
            chunk_shingles[cid] = sh
        lsh_build_time = time.time() - start_time
        lsh_memory = get_memory_footprint(lsh_index) + get_memory_footprint(minhash_objs) + get_memory_footprint(simhash_fps)

        # Query Latency & Recall Testing
        tfidf_latencies, lsh_latencies, lsh_recalls = [], [], []
        
        for q in TEST_QUERIES:
            # TF-IDF Latency
            t0 = time.perf_counter()
            q_vec = vectorizer.transform([preprocess_and_stem(q)])
            _ = cosine_similarity(q_vec, tfidf_matrix).flatten()
            tfidf_latencies.append(time.perf_counter() - t0)

            # LSH Latency
            t0 = time.perf_counter()
            _ = hybrid_search(q, lsh_index, minhash_objs, simhash_fps, chunk_shingles, base_chunks, top_k=10)
            lsh_latencies.append(time.perf_counter() - t0)

            # Accuracy (Recall@10 against Exact Jaccard)
            ground_truth = get_exact_jaccard_ground_truth(q, base_chunks, top_k=10)
            lsh_results = search_minhash(q, lsh_index, minhash_objs, chunk_shingles, base_chunks, top_k=10)
            lsh_retrieved_ids = [r['chunk_id'] for r in lsh_results]
            intersection = len(set(ground_truth) & set(lsh_retrieved_ids))
            lsh_recalls.append(intersection / 10.0 if ground_truth else 0)

        log_print(f"  [TF-IDF] Build Time: {tfidf_build_time:.4f}s | Memory: {tfidf_memory:.2f} MB | Avg Latency: {np.mean(tfidf_latencies)*1000:.2f} ms", report)
        log_print(f"  [LSH]    Build Time: {lsh_build_time:.4f}s | Memory: {lsh_memory:.2f} MB | Avg Latency: {np.mean(lsh_latencies)*1000:.2f} ms", report)
        log_print(f"  [LSH]    Average Recall@10 (vs Exact Jaccard): {np.mean(lsh_recalls):.2%}\n", report)

        # ---------------------------------------------------------------------
        # 2. PARAMETER SENSITIVITY
        # ---------------------------------------------------------------------
        log_print(">>> 2. PARAMETER SENSITIVITY ANALYSIS", report)
        
        # A. Number of Hash Functions (MinHash):
        log_print("  A. Number of Hash Functions (MinHash):", report)
        for perms in [32, 64, 128, 256]:
            t0 = time.time()
            temp_lsh = MinHashLSH(threshold=0.01, num_perm=perms)
            temp_objs = {}
            for c in base_chunks:
                # We use the 'perms' variable here for the index
                mh = compute_minhash(chunk_shingles[c["chunk_id"]], num_perm=perms)
                temp_lsh.insert(str(c["chunk_id"]), mh)
                temp_objs[c["chunk_id"]] = mh
            
            latencies, recalls = [], []
            for q in TEST_QUERIES[:5]:
                t1 = time.perf_counter()
                
                # Instead of calling search_minhash (which uses default 128),
                # manually compute the query minhash with the CURRENT 'perms'
                q_text = clean_string(q)
                q_shingles = make_shingles(q_text)
                q_mh = compute_minhash(q_shingles, num_perm=perms) # Use the loop variable!
                
                # Query the LSH
                candidate_ids = temp_lsh.query(q_mh)
                
                # Re-rank (Standard Jaccard logic)
                ranked = []
                for cid_str in candidate_ids:
                    cid = int(cid_str)
                    j = jaccard(q_shingles, chunk_shingles[cid])
                    ranked.append((cid, j))
                ranked.sort(key=lambda x: -x[1])
                res_ids = [cid for cid, score in ranked[:10]]
                # --- FIX ENDS HERE ---

                latencies.append(time.perf_counter() - t1)
                
                gt = get_exact_jaccard_ground_truth(q, base_chunks, top_k=10)
                intersect = len(set(gt) & set(res_ids))
                recalls.append(intersect / 10.0)
            
            log_print(f"    Perms={perms:<3} | Indexing: {time.time()-t0:.2f}s | Avg Latency: {np.mean(latencies)*1000:.2f}ms | Recall@10: {np.mean(recalls):.2%}", report)

        # B. LSH Threshold (Bands)
        log_print("\n  B. Jaccard Threshold (implicitly affects LSH Bands):", report)
        for thresh in [0.01, 0.1, 0.5]:
            temp_lsh = MinHashLSH(threshold=thresh, num_perm=128)
            for cid, mh in minhash_objs.items():
                temp_lsh.insert(str(cid), mh)
            
            latencies = []
            for q in TEST_QUERIES[:5]:
                t1 = time.perf_counter()
                q_mh = compute_minhash(make_shingles(clean_string(q)), num_perm=128)
                _ = temp_lsh.query(q_mh)
                latencies.append(time.perf_counter() - t1)
            # Retrieve bands info
            log_print(f"    Threshold={thresh:<4} -> (Bands={temp_lsh.b}, Rows={temp_lsh.r}) | LSH Query Latency: {np.mean(latencies)*1000:.2f}ms", report)

        # C. SimHash Hamming Threshold
        log_print("\n  C. Hamming Threshold (SimHash):", report)
        for h_thresh in [5, 15, 25]:
            counts = []
            for q in TEST_QUERIES[:5]:
                res = search_simhash(q, simhash_fps, base_chunks, threshold=h_thresh, top_k=100)
                counts.append(len(res))
            log_print(f"    Threshold={h_thresh:<2} | Avg candidates retrieved: {np.mean(counts):.1f}", report)

        log_print("\n", report)

        # ---------------------------------------------------------------------
        # 3. SCALABILITY TEST
        # ---------------------------------------------------------------------
        log_print(">>> 3. SCALABILITY TEST (Simulating Larger Datasets)", report)
        
        multipliers = [1, 2, 5]
        for m in multipliers:
            # Duplicate corpus and assign unique IDs
            scaled_chunks = []
            new_cid = 0
            for i in range(m):
                for c in base_chunks:
                    new_c = dict(c)
                    new_c['chunk_id'] = new_cid
                    scaled_chunks.append(new_c)
                    new_cid += 1
            
            # Measure LSH Indexing
            t0 = time.time()
            s_lsh = MinHashLSH(threshold=0.01, num_perm=128)
            s_objs = {}
            for c in scaled_chunks:
                sh = make_shingles(clean_string(c["text"]))
                mh = compute_minhash(sh, num_perm=128)
                s_lsh.insert(str(c["chunk_id"]), mh)
                s_objs[c["chunk_id"]] = mh
            idx_time = time.time() - t0

            # Measure Query Latency
            latencies = []
            for q in TEST_QUERIES[:5]:
                t1 = time.perf_counter()
                q_mh = compute_minhash(make_shingles(clean_string(q)), num_perm=128)
                _ = s_lsh.query(q_mh)
                latencies.append(time.perf_counter() - t1)
            
            log_print(f"  {m}x Corpus ({len(scaled_chunks)} chunks) | LSH Index Time: {idx_time:.2f}s | Query Latency: {np.mean(latencies)*1000:.2f}ms", report)
        log_print("\n", report)

        # ---------------------------------------------------------------------
        # 4. QUALITATIVE EVALUATION
        # ---------------------------------------------------------------------
        log_print(">>> 4. QUALITATIVE EVALUATION (Answers for 15 Queries)", report)
        
        for idx, q in enumerate(TEST_QUERIES, 1):
            log_print(f"\n[Query {idx}] {q}", report)
            res = hybrid_search(q, lsh_index, minhash_objs, simhash_fps, chunk_shingles, base_chunks, top_k=3)
            
            if os.environ.get("GROQ_API_KEY"):
                answer = generate_answer(q, res)
                log_print(f"  [LLM Answer] {answer}", report)
            else:
                log_print("  [LLM Answer] Skipped (No GROQ_API_KEY found). Showing top retrieved context instead:", report)
            
            log_print(f"  [Evidence] Source: {res[0]['source']} (Pg {res[0]['page']}) - Score: {res[0]['score']}", report)
            log_print(f"             \"{res[0]['text'][:150]}...\"", report)

    print(f"\n[SUCCESS] Comprehensive experiments completed! Check {REPORT_FILE}")

if __name__ == "__main__":
    run_experiments()