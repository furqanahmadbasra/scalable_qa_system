import argparse
import csv
import os
from typing import Dict, List

from answer_generation import generate_answer
from indexing import load_index
from lsh_retrieval import (
    FUSED_DEFAULT_CANDIDATE_POOL,
    load_lsh_index,
    search_minhash,
    hybrid_search,
    fused_search,
)
from retrieval import search as search_tfidf
from extensions.pagerank_ranker import build_pagerank_scores


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
PAGERANK_FILE = os.path.join(RESULTS_DIR, "pagerank_scores.csv")


def load_pagerank_scores(chunks: List[dict]) -> Dict[int, float]:
    if os.path.exists(PAGERANK_FILE):
        scores = {}
        with open(PAGERANK_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scores[int(row["chunk_id"])] = float(row["pagerank_score"])
        if scores:
            return scores
    return build_pagerank_scores(chunks)


def run_query(
    query: str,
    method: str,
    top_k: int,
    tfidf_vectorizer,
    tfidf_matrix,
    tfidf_chunks,
    lsh_index,
    minhash_objects,
    simhash_fps,
    chunk_shingles,
    chunk_tokens,
    lsh_chunks,
    pagerank_scores,
):
    if method == "tfidf":
        return search_tfidf(query, tfidf_vectorizer, tfidf_matrix, tfidf_chunks, top_k=top_k)
    if method == "minhash":
        return search_minhash(query, lsh_index, minhash_objects, chunk_shingles, lsh_chunks, top_k=top_k)
    if method == "hybrid":
        return hybrid_search(
            query, lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunk_tokens, lsh_chunks, top_k=top_k
        )
    if method == "fused":
        return fused_search(
            query,
            lsh_index,
            minhash_objects,
            simhash_fps,
            chunk_shingles,
            chunk_tokens,
            lsh_chunks,
            tfidf_vectorizer,
            tfidf_matrix,
            top_k=top_k,
            candidate_pool=FUSED_DEFAULT_CANDIDATE_POOL,
        )
    if method == "fused_pagerank":
        return fused_search(
            query,
            lsh_index,
            minhash_objects,
            simhash_fps,
            chunk_shingles,
            chunk_tokens,
            lsh_chunks,
            tfidf_vectorizer,
            tfidf_matrix,
            top_k=top_k,
            candidate_pool=FUSED_DEFAULT_CANDIDATE_POOL,
            pagerank_scores=pagerank_scores,
            pagerank_weight=0.10,
        )
    raise ValueError(f"Unsupported method: {method}")


def print_results(query: str, method: str, chunks: List[dict]):
    print("\n" + "=" * 72)
    print(f"QUERY: {query}")
    print(f"METHOD: {method}")
    print("=" * 72)

    if not chunks:
        print("No retrieval results found.")
        return

    if os.environ.get("GROQ_API_KEY"):
        answer = generate_answer(query, chunks)
        print("\nANSWER")
        print("-" * 72)
        print(answer)
    else:
        print("\nANSWER")
        print("-" * 72)
        print("[Skipped: GROQ_API_KEY not set]")

    print("\nTOP-K RETRIEVED CHUNKS")
    print("-" * 72)
    for i, c in enumerate(chunks, start=1):
        print(f"[{i}] source={c['source']} page={c['page']} score={c['score']}")
        print(f"    snippet: {c['text'][:220]}...")

    print("\nSOURCE REFERENCES")
    print("-" * 72)
    for i, c in enumerate(chunks, start=1):
        print(f"[{i}] {c['source']} (Page {c['page']})")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive QA CLI for handbook policy system.")
    parser.add_argument("--query", type=str, default="", help="Single query to run (non-interactive mode).")
    parser.add_argument(
        "--method",
        type=str,
        default="fused_pagerank",
        choices=["tfidf", "minhash", "hybrid", "fused", "fused_pagerank"],
        help="Retrieval method to use.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    print("Loading indices...")
    tfidf_vectorizer, tfidf_matrix, tfidf_chunks = load_index()
    lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunk_tokens, lsh_chunks = load_lsh_index()
    pagerank_scores = load_pagerank_scores(lsh_chunks)

    if args.query:
        results = run_query(
            args.query,
            args.method,
            args.top_k,
            tfidf_vectorizer,
            tfidf_matrix,
            tfidf_chunks,
            lsh_index,
            minhash_objects,
            simhash_fps,
            chunk_shingles,
            chunk_tokens,
            lsh_chunks,
            pagerank_scores,
        )
        print_results(args.query, args.method, results)
        return

    print("Interactive QA mode. Type 'exit' to quit.")
    print("Methods: tfidf | minhash | hybrid | fused | fused_pagerank")
    method = args.method
    while True:
        user_q = input("\nEnter query: ").strip()
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            print("Exiting QA CLI.")
            break

        chosen = input(f"Method [{method}]: ").strip()
        if chosen:
            method = chosen if chosen in {"tfidf", "minhash", "hybrid", "fused", "fused_pagerank"} else method

        results = run_query(
            user_q,
            method,
            args.top_k,
            tfidf_vectorizer,
            tfidf_matrix,
            tfidf_chunks,
            lsh_index,
            minhash_objects,
            simhash_fps,
            chunk_shingles,
            chunk_tokens,
            lsh_chunks,
            pagerank_scores,
        )
        print_results(user_q, method, results)


if __name__ == "__main__":
    main()
