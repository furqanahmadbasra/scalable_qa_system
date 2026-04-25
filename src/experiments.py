import os
import sys
import time
import json
import pickle
import csv
import re
import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

# Import from existing modules
from indexing import preprocess_and_stem
from lsh_indexing import clean_tokens, make_shingles, compute_minhash, compute_simhash
from lsh_retrieval import (
    jaccard,
    hamming,
    search_simhash,
    search_minhash,
    hybrid_search,
    fused_search,
    FUSED_DEFAULT_CANDIDATE_POOL,
    FUSED_DEFAULT_HYBRID_WEIGHT,
    FUSED_DEFAULT_TFIDF_WEIGHT,
)
from answer_generation import generate_answer
from extensions.frequent_patterns import (
    apriori_frequent_itemsets,
    build_query_log_records,
    build_transactions,
    write_itemsets_csv,
    write_itemsets_text_report,
    write_query_log_csv,
)

CHUNKS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "chunks", "chunks.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
REPORT_FILE = os.path.join(RESULTS_DIR, "comprehensive_report.txt")
PER_QUERY_METRICS_FILE = os.path.join(RESULTS_DIR, "per_query_metrics.csv")
QUERY_LOG_FILE = os.path.join(RESULTS_DIR, "query_log.csv")
FREQ_ITEMSETS_CSV = os.path.join(RESULTS_DIR, "frequent_itemsets_report.csv")
FREQ_ITEMSETS_TXT = os.path.join(RESULTS_DIR, "frequent_itemsets_report.txt")

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
FUSION_WEIGHT_CANDIDATES = [0.5, 0.6, 0.7]
WEAK_QUERY_INDICES = [3, 9, 12]  # Query numbers 4, 10, 13 (0-based)
METRIC_KS = [3, 5, 10]
PRESERVED_POLICY_WORDS = {"not", "no", "nor", "must", "shall", "may"}
EXPERIMENT_STOPWORDS = ENGLISH_STOP_WORDS - PRESERVED_POLICY_WORDS
stemmer = PorterStemmer()

def log_print(msg, f):
    """Prints to console and writes to log file."""
    print(msg)
    f.write(msg + "\n")

def get_memory_footprint(obj):
    """Helper to get actual memory size by pickling."""
    return len(pickle.dumps(obj)) / (1024 * 1024)  # Return in MB

def get_exact_jaccard_ground_truth(query, chunks, top_k=10):
    """Computes EXACT Jaccard similarity across ALL chunks to serve as ground truth for Recall."""
    q_shingles = make_shingles(clean_tokens(query))
    scores = []
    for c in chunks:
        c_shingles = make_shingles(clean_tokens(c['text']))
        scores.append((c['chunk_id'], jaccard(q_shingles, c_shingles)))
    scores.sort(key=lambda x: -x[1])
    return [cid for cid, score in scores[:top_k]]


def compute_precision_recall(retrieved_ids, ground_truth_ids, k):
    truth = set(ground_truth_ids[:k])
    pred = set(retrieved_ids[:k])
    if not pred:
        return 0.0, 0.0
    inter = len(pred & truth)
    precision = inter / float(k)
    recall = inter / float(len(truth) if truth else 1)
    return precision, recall


def clean_tokens_variant(text, remove_stopwords=True):
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    normalized = []
    for token in text.split():
        if "-" in token:
            parts = [p for p in token.split("-") if p]
            if not parts:
                continue
            normalized.extend(parts)
            normalized.append("".join(parts))
        else:
            normalized.append(token)

    final_tokens = []
    for token in normalized:
        if len(token) <= 1:
            continue
        if remove_stopwords and token in EXPERIMENT_STOPWORDS:
            continue
        final_tokens.append(stemmer.stem(token))
    return final_tokens


def make_char_shingles_from_tokens(tokens, k):
    text = " ".join(tokens)
    if len(text) < k:
        return {text} if text else set()
    return {text[i:i+k] for i in range(len(text) - k + 1)}

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
        lsh_index = MinHashLSH(threshold=0.2, num_perm=128)
        minhash_objs, simhash_fps, chunk_shingles, chunk_tokens = {}, {}, {}, {}
        for c in base_chunks:
            cid = c["chunk_id"]
            tokens = clean_tokens(c["text"])
            sh = make_shingles(tokens)
            mh = compute_minhash(sh, num_perm=128)
            lsh_index.insert(str(cid), mh)
            minhash_objs[cid] = mh
            simhash_fps[cid] = compute_simhash(tokens, num_bits=64)
            chunk_shingles[cid] = sh
            chunk_tokens[cid] = tokens
        lsh_build_time = time.time() - start_time
        lsh_memory = get_memory_footprint(lsh_index) + get_memory_footprint(minhash_objs) + get_memory_footprint(simhash_fps)

        # ---------------------------------------------------------------------
        # 1B. FUSION WEIGHT TUNING (target weak queries without hurting overall)
        # ---------------------------------------------------------------------
        best_tfidf_weight = FUSED_DEFAULT_TFIDF_WEIGHT
        best_hybrid_weight = FUSED_DEFAULT_HYBRID_WEIGHT
        best_score = -1.0
        tuning_rows = []

        for w in FUSION_WEIGHT_CANDIDATES:
            h = 1.0 - w
            all_recalls = []
            weak_recalls = []
            for qi, q in enumerate(TEST_QUERIES):
                gt = get_exact_jaccard_ground_truth(q, base_chunks, top_k=10)
                fused_results = fused_search(
                    q,
                    lsh_index,
                    minhash_objs,
                    simhash_fps,
                    chunk_shingles,
                    chunk_tokens,
                    base_chunks,
                    vectorizer,
                    tfidf_matrix,
                    top_k=10,
                    candidate_pool=FUSED_DEFAULT_CANDIDATE_POOL,
                    tfidf_weight=w,
                    hybrid_weight=h,
                )
                fused_ids = [r["chunk_id"] for r in fused_results]
                recall = len(set(gt) & set(fused_ids)) / 10.0 if gt else 0.0
                all_recalls.append(recall)
                if qi in WEAK_QUERY_INDICES:
                    weak_recalls.append(recall)

            avg_all = float(np.mean(all_recalls))
            avg_weak = float(np.mean(weak_recalls)) if weak_recalls else 0.0
            score = (0.65 * avg_all) + (0.35 * avg_weak)
            tuning_rows.append((w, h, avg_all, avg_weak, score))

            if score > best_score:
                best_score = score
                best_tfidf_weight = w
                best_hybrid_weight = h

        log_print(">>> 1B. FUSION WEIGHT TUNING (TF-IDF vs Hybrid)", report)
        for w, h, avg_all, avg_weak, score in tuning_rows:
            log_print(
                f"  TF-IDF={w:.2f}, Hybrid={h:.2f} | Avg Recall@10={avg_all:.2%} | "
                f"Weak(4/10/13) Recall@10={avg_weak:.2%} | Combined={score:.4f}",
                report,
            )
        log_print(
            f"  Selected Fusion Weights -> TF-IDF={best_tfidf_weight:.2f}, Hybrid={best_hybrid_weight:.2f}\n",
            report,
        )

        # Query Latency, Recall and Precision Testing
        tfidf_latencies, lsh_latencies, hybrid_latencies, fused_latencies = [], [], [], []
        lsh_recalls, hybrid_recalls, fused_recalls = [], [], []
        metric_summary = {
            "tfidf": {k: {"p": [], "r": []} for k in METRIC_KS},
            "minhash": {k: {"p": [], "r": []} for k in METRIC_KS},
            "hybrid": {k: {"p": [], "r": []} for k in METRIC_KS},
            "fused": {k: {"p": [], "r": []} for k in METRIC_KS},
        }
        per_query_rows = []
        
        for q in TEST_QUERIES:
            # TF-IDF Latency
            t0 = time.perf_counter()
            q_vec = vectorizer.transform([preprocess_and_stem(q)])
            _ = cosine_similarity(q_vec, tfidf_matrix).flatten()
            tfidf_latencies.append(time.perf_counter() - t0)

            # MinHash LSH Latency
            t0 = time.perf_counter()
            _ = search_minhash(q, lsh_index, minhash_objs, chunk_shingles, base_chunks, top_k=10)
            lsh_latencies.append(time.perf_counter() - t0)

            ground_truth = get_exact_jaccard_ground_truth(q, base_chunks, top_k=10)
            lsh_results = search_minhash(q, lsh_index, minhash_objs, chunk_shingles, base_chunks, top_k=10)
            lsh_retrieved_ids = [r['chunk_id'] for r in lsh_results]
            intersection = len(set(ground_truth) & set(lsh_retrieved_ids))
            lsh_recalls.append(intersection / 10.0 if ground_truth else 0)

            t0 = time.perf_counter()
            hybrid_results = hybrid_search(q, lsh_index, minhash_objs, simhash_fps, chunk_shingles, chunk_tokens, base_chunks, top_k=10)
            hybrid_latencies.append(time.perf_counter() - t0)
            hybrid_retrieved_ids = [r["chunk_id"] for r in hybrid_results]
            hybrid_intersection = len(set(ground_truth) & set(hybrid_retrieved_ids))
            hybrid_recalls.append(hybrid_intersection / 10.0 if ground_truth else 0)

            t0 = time.perf_counter()
            fused_results = fused_search(
                q,
                lsh_index,
                minhash_objs,
                simhash_fps,
                chunk_shingles,
                chunk_tokens,
                base_chunks,
                vectorizer,
                tfidf_matrix,
                top_k=10,
                candidate_pool=FUSED_DEFAULT_CANDIDATE_POOL,
                tfidf_weight=best_tfidf_weight,
                hybrid_weight=best_hybrid_weight,
            )
            fused_latencies.append(time.perf_counter() - t0)
            fused_ids = [r["chunk_id"] for r in fused_results]
            fused_intersection = len(set(ground_truth) & set(fused_ids))
            fused_recalls.append(fused_intersection / 10.0 if ground_truth else 0)

            tfidf_scores = cosine_similarity(
                vectorizer.transform([preprocess_and_stem(q)]), tfidf_matrix
            ).flatten()
            tfidf_ids = list(np.argsort(tfidf_scores)[::-1][:10])

            for k in METRIC_KS:
                p, r = compute_precision_recall(tfidf_ids, ground_truth, k)
                metric_summary["tfidf"][k]["p"].append(p)
                metric_summary["tfidf"][k]["r"].append(r)

                p, r = compute_precision_recall(lsh_retrieved_ids, ground_truth, k)
                metric_summary["minhash"][k]["p"].append(p)
                metric_summary["minhash"][k]["r"].append(r)

                p, r = compute_precision_recall(hybrid_retrieved_ids, ground_truth, k)
                metric_summary["hybrid"][k]["p"].append(p)
                metric_summary["hybrid"][k]["r"].append(r)

                p, r = compute_precision_recall(fused_ids, ground_truth, k)
                metric_summary["fused"][k]["p"].append(p)
                metric_summary["fused"][k]["r"].append(r)

            top_fused = fused_results[0] if fused_results else None
            row = {
                "query": q,
                "top_source": top_fused["source"] if top_fused else "n/a",
                "top_page": top_fused["page"] if top_fused else -1,
                "tfidf_latency_ms": round(tfidf_latencies[-1] * 1000, 3),
                "minhash_latency_ms": round(lsh_latencies[-1] * 1000, 3),
                "hybrid_latency_ms": round(hybrid_latencies[-1] * 1000, 3),
                "fused_latency_ms": round(fused_latencies[-1] * 1000, 3),
            }
            for k in METRIC_KS:
                p, r = compute_precision_recall(fused_ids, ground_truth, k)
                row[f"fused_p@{k}"] = round(p, 4)
                row[f"fused_r@{k}"] = round(r, 4)
            per_query_rows.append(row)

        log_print(f"  [TF-IDF] Build Time: {tfidf_build_time:.4f}s | Memory: {tfidf_memory:.2f} MB | Avg Latency: {np.mean(tfidf_latencies)*1000:.2f} ms", report)
        log_print(f"  [LSH]    Build Time: {lsh_build_time:.4f}s | Memory: {lsh_memory:.2f} MB | Avg Latency: {np.mean(lsh_latencies)*1000:.2f} ms", report)
        log_print(f"  [LSH-MinHash] Average Recall@10 (vs Exact Jaccard): {np.mean(lsh_recalls):.2%}", report)
        log_print(f"  [LSH-Hybrid]  Average Recall@10 (vs Exact Jaccard): {np.mean(hybrid_recalls):.2%}", report)
        log_print(f"  [LSH-Fused]   Average Recall@10 (vs Exact Jaccard): {np.mean(fused_recalls):.2%}", report)
        log_print(f"  [LSH-Hybrid]  Avg Latency: {np.mean(hybrid_latencies)*1000:.2f} ms", report)
        log_print(f"  [LSH-Fused]   Avg Latency: {np.mean(fused_latencies)*1000:.2f} ms\n", report)

        log_print(">>> 1C. PRECISION/RECALL@K SUMMARY (mean +/- std)", report)
        for method in ["tfidf", "minhash", "hybrid", "fused"]:
            label = method.upper()
            for k in METRIC_KS:
                p_vals = metric_summary[method][k]["p"]
                r_vals = metric_summary[method][k]["r"]
                log_print(
                    f"  [{label}] P@{k}: {np.mean(p_vals):.3f} +/- {np.std(p_vals):.3f} | "
                    f"R@{k}: {np.mean(r_vals):.3f} +/- {np.std(r_vals):.3f}",
                    report,
                )
        log_print("", report)

        # ---------------------------------------------------------------------
        # 2. PARAMETER SENSITIVITY
        # ---------------------------------------------------------------------
        log_print(">>> 2. PARAMETER SENSITIVITY ANALYSIS", report)
        
        # A. Number of Hash Functions (MinHash):
        log_print("  A. Number of Hash Functions (MinHash):", report)
        for perms in [32, 64, 128, 256]:
            t0 = time.time()
            temp_lsh = MinHashLSH(threshold=0.2, num_perm=perms)
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
                q_tokens = clean_tokens(q)
                q_shingles = make_shingles(q_tokens)
                q_mh = compute_minhash(q_shingles, num_perm=perms) # Use the loop variable!
                
                # Query the LSH
                candidate_ids = temp_lsh.query(q_mh)
                if not candidate_ids:
                    fallback = sorted(
                        temp_objs.items(),
                        key=lambda x: -x[1].jaccard(q_mh),
                    )[:30]
                    candidate_ids = [str(cid) for cid, _ in fallback]
                
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
        for thresh in [0.1, 0.2, 0.3]:
            temp_lsh = MinHashLSH(threshold=thresh, num_perm=128)
            for cid, mh in minhash_objs.items():
                temp_lsh.insert(str(cid), mh)
            
            latencies = []
            for q in TEST_QUERIES[:5]:
                t1 = time.perf_counter()
                q_mh = compute_minhash(make_shingles(clean_tokens(q)), num_perm=128)
                _ = temp_lsh.query(q_mh)
                latencies.append(time.perf_counter() - t1)
            # Retrieve bands info
            log_print(f"    Threshold={thresh:<4} -> (Bands={temp_lsh.b}, Rows={temp_lsh.r}) | LSH Query Latency: {np.mean(latencies)*1000:.2f}ms", report)

        # C. SimHash Hamming Threshold
        log_print("\n  C. Hamming Threshold (SimHash):", report)
        for h_thresh in [10, 12, 15]:
            counts = []
            for q in TEST_QUERIES[:5]:
                res = search_simhash(q, simhash_fps, base_chunks, threshold=h_thresh, top_k=100)
                counts.append(len(res))
            log_print(f"    Threshold={h_thresh:<2} | Avg candidates retrieved: {np.mean(counts):.1f}", report)

        # D. Stopwords On/Off in LSH preprocessing
        log_print("\n  D. Stopwords Ablation (LSH preprocessing):", report)
        for remove_sw in [True, False]:
            t0 = time.time()
            temp_lsh = MinHashLSH(threshold=0.2, num_perm=128)
            temp_objs = {}
            temp_shingles = {}
            for c in base_chunks:
                toks = clean_tokens_variant(c["text"], remove_stopwords=remove_sw)
                sh = make_shingles(toks)
                mh = compute_minhash(sh, num_perm=128)
                temp_lsh.insert(str(c["chunk_id"]), mh)
                temp_objs[c["chunk_id"]] = mh
                temp_shingles[c["chunk_id"]] = sh
            idx_time = time.time() - t0

            recalls, latencies = [], []
            for q in TEST_QUERIES[:5]:
                q_toks = clean_tokens_variant(q, remove_stopwords=remove_sw)
                q_sh = make_shingles(q_toks)
                q_mh = compute_minhash(q_sh, num_perm=128)
                t1 = time.perf_counter()
                cands = temp_lsh.query(q_mh)
                latencies.append(time.perf_counter() - t1)
                if not cands:
                    cands = [str(cid) for cid, _ in sorted(temp_objs.items(), key=lambda x: -x[1].jaccard(q_mh))[:30]]

                ranked = []
                for cid_str in cands:
                    cid = int(cid_str)
                    ranked.append((cid, jaccard(q_sh, temp_shingles[cid])))
                ranked.sort(key=lambda x: -x[1])
                ids = [cid for cid, _ in ranked[:10]]
                gt = get_exact_jaccard_ground_truth(q, base_chunks, top_k=10)
                recalls.append(len(set(ids) & set(gt)) / 10.0)

            mode = "with_stopword_removal" if remove_sw else "without_stopword_removal"
            log_print(
                f"    {mode}: Indexing={idx_time:.2f}s | Avg Latency={np.mean(latencies)*1000:.2f}ms | Recall@10={np.mean(recalls):.2%}",
                report,
            )

        # E. Word vs Char shingle ablation
        log_print("\n  E. Word vs Char Shingle Ablation:", report)
        # Word-level baseline
        word_recalls = []
        for q in TEST_QUERIES[:5]:
            gt = get_exact_jaccard_ground_truth(q, base_chunks, top_k=10)
            ids = [r["chunk_id"] for r in search_minhash(q, lsh_index, minhash_objs, chunk_shingles, base_chunks, top_k=10)]
            word_recalls.append(len(set(ids) & set(gt)) / 10.0)
        log_print(f"    Word shingles (1,2): Recall@10={np.mean(word_recalls):.2%}", report)

        for k in [4, 5, 6]:
            t0 = time.time()
            temp_lsh = MinHashLSH(threshold=0.2, num_perm=128)
            temp_objs = {}
            temp_shingles = {}
            for c in base_chunks:
                toks = clean_tokens(c["text"])
                sh = make_char_shingles_from_tokens(toks, k)
                mh = compute_minhash(sh, num_perm=128)
                temp_lsh.insert(str(c["chunk_id"]), mh)
                temp_objs[c["chunk_id"]] = mh
                temp_shingles[c["chunk_id"]] = sh
            idx_time = time.time() - t0

            recalls, latencies = [], []
            for q in TEST_QUERIES[:5]:
                q_toks = clean_tokens(q)
                q_sh = make_char_shingles_from_tokens(q_toks, k)
                q_mh = compute_minhash(q_sh, num_perm=128)
                t1 = time.perf_counter()
                cands = temp_lsh.query(q_mh)
                latencies.append(time.perf_counter() - t1)
                if not cands:
                    cands = [str(cid) for cid, _ in sorted(temp_objs.items(), key=lambda x: -x[1].jaccard(q_mh))[:30]]
                ranked = []
                for cid_str in cands:
                    cid = int(cid_str)
                    ranked.append((cid, jaccard(q_sh, temp_shingles[cid])))
                ranked.sort(key=lambda x: -x[1])
                ids = [cid for cid, _ in ranked[:10]]
                gt = get_exact_jaccard_ground_truth(q, base_chunks, top_k=10)
                recalls.append(len(set(ids) & set(gt)) / 10.0)
            log_print(
                f"    Char shingles k={k}: Indexing={idx_time:.2f}s | Avg Latency={np.mean(latencies)*1000:.2f}ms | Recall@10={np.mean(recalls):.2%}",
                report,
            )

        log_print("\n", report)

        # ---------------------------------------------------------------------
        # 3. SCALABILITY TEST
        # ---------------------------------------------------------------------
        log_print(">>> 3. SCALABILITY TEST (Simulating Larger Datasets)", report)
        
        multipliers = [1, 2, 5, 10, 20]
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
            s_lsh = MinHashLSH(threshold=0.2, num_perm=128)
            s_objs, s_shingles = {}, {}
            for c in scaled_chunks:
                sh = make_shingles(clean_tokens(c["text"]))
                mh = compute_minhash(sh, num_perm=128)
                s_lsh.insert(str(c["chunk_id"]), mh)
                s_objs[c["chunk_id"]] = mh
                s_shingles[c["chunk_id"]] = sh
            idx_time = time.time() - t0
            idx_memory = get_memory_footprint(s_lsh) + get_memory_footprint(s_objs)

            # Measure Query Latency
            latencies = []
            recalls = []
            for q in TEST_QUERIES[:5]:
                t1 = time.perf_counter()
                q_mh = compute_minhash(make_shingles(clean_tokens(q)), num_perm=128)
                candidate_ids = s_lsh.query(q_mh)
                latencies.append(time.perf_counter() - t1)
                if not candidate_ids:
                    candidate_ids = [str(cid) for cid, _ in sorted(s_objs.items(), key=lambda x: -x[1].jaccard(q_mh))[:30]]
                ranked = []
                q_sh = make_shingles(clean_tokens(q))
                for cid_str in candidate_ids:
                    cid = int(cid_str)
                    ranked.append((cid, jaccard(q_sh, s_shingles[cid])))
                ranked.sort(key=lambda x: -x[1])
                ids = [cid for cid, _ in ranked[:10]]
                gt = get_exact_jaccard_ground_truth(q, base_chunks, top_k=10)
                recalls.append(len(set(ids) & set(gt)) / 10.0)

            log_print(
                f"  {m}x Corpus ({len(scaled_chunks)} chunks) | LSH Index Time: {idx_time:.2f}s | "
                f"Memory: {idx_memory:.2f}MB | Query Latency: {np.mean(latencies)*1000:.2f}ms | "
                f"Recall@10 trend: {np.mean(recalls):.2%}",
                report,
            )
        log_print("\n", report)

        # ---------------------------------------------------------------------
        # 4. FREQUENT ITEMSET MINING (Query Pattern Extension)
        # ---------------------------------------------------------------------
        log_print(">>> 4. FREQUENT ITEMSET MINING (QUERY PATTERNS)", report)
        query_records = build_query_log_records(TEST_QUERIES, "2026-04-25")
        transactions = build_transactions(query_records)
        frequent_itemsets = apriori_frequent_itemsets(
            transactions=transactions,
            min_support_count=2,
            max_k=3,
        )
        write_query_log_csv(query_records, QUERY_LOG_FILE)
        write_itemsets_csv(frequent_itemsets, FREQ_ITEMSETS_CSV)
        write_itemsets_text_report(frequent_itemsets, FREQ_ITEMSETS_TXT)

        log_print(
            f"  Query log records: {len(query_records)} | transactions: {len(transactions)}",
            report,
        )
        for k in sorted(frequent_itemsets.keys()):
            log_print(f"  Frequent itemsets k={k}: {len(frequent_itemsets[k])}", report)
        if frequent_itemsets.get(2):
            top_pair = frequent_itemsets[2][0]
            log_print(
                f"  Top pair pattern: {' | '.join(top_pair[0])} (support={top_pair[1]})",
                report,
            )
        log_print(f"  Saved query log: {QUERY_LOG_FILE}", report)
        log_print(f"  Saved itemsets CSV: {FREQ_ITEMSETS_CSV}", report)
        log_print(f"  Saved itemsets report: {FREQ_ITEMSETS_TXT}\n", report)

        # ---------------------------------------------------------------------
        # 5. QUALITATIVE EVALUATION
        # ---------------------------------------------------------------------
        log_print(">>> 5. QUALITATIVE EVALUATION (Answers for 15 Queries)", report)
        
        for idx, q in enumerate(TEST_QUERIES, 1):
            log_print(f"\n[Query {idx}] {q}", report)
            res = fused_search(
                q,
                lsh_index,
                minhash_objs,
                simhash_fps,
                chunk_shingles,
                chunk_tokens,
                base_chunks,
                vectorizer,
                tfidf_matrix,
                top_k=3,
                candidate_pool=FUSED_DEFAULT_CANDIDATE_POOL,
                tfidf_weight=best_tfidf_weight,
                hybrid_weight=best_hybrid_weight,
            )
            
            if os.environ.get("GROQ_API_KEY"):
                answer = generate_answer(q, res)
                log_print(f"  [LLM Answer] {answer}", report)
            else:
                log_print("  [LLM Answer] Skipped (No GROQ_API_KEY found). Showing top retrieved context instead:", report)
            
            log_print(f"  [Evidence] Source: {res[0]['source']} (Pg {res[0]['page']}) - Score: {res[0]['score']}", report)
            log_print(f"             \"{res[0]['text'][:150]}...\"", report)

    if per_query_rows:
        fieldnames = [
            "query",
            "top_source",
            "top_page",
            "tfidf_latency_ms",
            "minhash_latency_ms",
            "hybrid_latency_ms",
            "fused_latency_ms",
        ] + [f"fused_p@{k}" for k in METRIC_KS] + [f"fused_r@{k}" for k in METRIC_KS]
        with open(PER_QUERY_METRICS_FILE, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_query_rows)

    print(f"\n[SUCCESS] Comprehensive experiments completed! Check {REPORT_FILE}")
    print(f"[SUCCESS] Per-query metrics table written to {PER_QUERY_METRICS_FILE}")
    print(f"[SUCCESS] Frequent itemset reports written to {FREQ_ITEMSETS_CSV} and {FREQ_ITEMSETS_TXT}")

if __name__ == "__main__":
    run_experiments()