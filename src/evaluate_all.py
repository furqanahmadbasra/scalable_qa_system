import os
import sys

# Import our retrieval and generation modules
from retrieval import load_index as load_tfidf_index, search as search_tfidf
from lsh_retrieval import load_lsh_index, hybrid_search, fused_search
from answer_generation import generate_answer

# Setup output file path
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_FILE = os.path.join(RESULTS_DIR, "evaluation_log.txt")

def format_chunks(chunks):
    """Helper to convert list of chunks into readable string."""
    out = ""
    for idx, c in enumerate(chunks, 1):
        out += f"  [{idx}] Source: {c['source']} (Page {c['page']}) | Score: {c['score']:.4f}\n"
        out += f"      Snippet: {c['text'][:150]}...\n"
    return out

def run_evaluation():
    print("Loading indices... (This may take a second)")
    
    # Load indices
    tfidf_vectorizer, tfidf_matrix, tfidf_chunks = load_tfidf_index()
    lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunk_tokens, lsh_chunks = load_lsh_index()
    
    test_queries = [
        "What is the minimum GPA requirement for graduation?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?",
    ]
    
    output_log = []
    output_log.append("==========================================")
    output_log.append("   SYSTEM EVALUATION LOG (QUALITATIVE)")
    output_log.append("==========================================\n")
    
    for query in test_queries:
        print(f"Processing query: '{query}'")
        output_log.append(f"QUERY: {query}\n" + "-"*50)
        
        # 1. Baseline (TF-IDF)
        tfidf_results = search_tfidf(query, tfidf_vectorizer, tfidf_matrix, tfidf_chunks, top_k=10)
        output_log.append(">>> BASELINE RETRIEVAL (TF-IDF):")
        output_log.append(format_chunks(tfidf_results))
        
        # 2. LSH + TF-IDF fusion re-ranking
        lsh_results = fused_search(
            query,
            lsh_index,
            minhash_objects,
            simhash_fps,
            chunk_shingles,
            chunk_tokens,
            lsh_chunks,
            tfidf_vectorizer,
            tfidf_matrix,
            top_k=10,
            candidate_pool=50,
        )
        output_log.append(">>> FUSED RETRIEVAL (LSH Hybrid candidates + TF-IDF rerank):")
        output_log.append(format_chunks(lsh_results))
        
        # 3. LLM Generation
        # We pass the LSH results (since LSH is the core of this project) to the LLM
        if os.environ.get("GROQ_API_KEY"):
            answer = generate_answer(query, lsh_results)
            output_log.append(">>> GENERATED LLM ANSWER (using LSH Context):")
            output_log.append(f"  {answer}\n")
        else:
            output_log.append(">>> GENERATED LLM ANSWER:")
            output_log.append("  [SKIPPED - No GROQ_API_KEY provided]")
            
        output_log.append("\n==========================================\n")
        
    # Write to file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_log))
        
    print(f"\nEvaluation complete. Results saved to: {LOG_FILE}")

if __name__ == "__main__":
    run_evaluation()
