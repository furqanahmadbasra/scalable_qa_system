import sys
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# pull preprocess and load_index from indexing since they live there
sys.path.append(os.path.dirname(__file__))
from indexing import preprocess, preprocess_and_stem, load_index


def search(query, vectorizer, matrix, chunks, top_k=5):
    q_vec = vectorizer.transform([preprocess_and_stem(query)])
    scores = cosine_similarity(q_vec, matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "source":   chunks[idx]["source"],
            "page":     chunks[idx]["page"],
            "score":    round(float(scores[idx]), 4),
            "text":     chunks[idx]["text"],
        })
    return results


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("loading index...")
    vectorizer, matrix, chunks = load_index()

    test_queries = [
        "minimum GPA requirement",
        "what happens if a student fails a course",
        "attendance policy",
        "how many times can a course be repeated",
    ]

    print("\n--- tfidf retrieval test ---\n")
    for q in test_queries:
        print(f"query: {q!r}")
        results = search(q, vectorizer, matrix, chunks, top_k=3)
        for r in results:
            print(f"  [{r['score']}] {r['source']} p.{r['page']} — {r['text'][:120]}...")
        print()
