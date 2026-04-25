import os
import re
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS_DIR = "experiments/results"
REPORT_PATH = os.path.join(RESULTS_DIR, "comprehensive_report.txt")
PER_QUERY_METRICS_PATH = os.path.join(RESULTS_DIR, "per_query_metrics.csv")
PAGERANK_PATH = os.path.join(RESULTS_DIR, "pagerank_scores.csv")
FREQ_PATH = os.path.join(RESULTS_DIR, "frequent_itemsets_report.csv")
DIST_PATH = os.path.join(RESULTS_DIR, "distributed_scaling_report.csv")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from answer_generation import generate_answer
from indexing import load_index as load_tfidf_index
from retrieval import search as search_tfidf
from lsh_retrieval import (
    FUSED_DEFAULT_CANDIDATE_POOL,
    FUSED_DEFAULT_PAGERANK_WEIGHT,
    load_lsh_index,
    search_minhash,
    hybrid_search,
    fused_search,
)


def _safe_read_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def parse_report_text(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    out = {}
    baseline_matches = re.findall(r"\[(TF-IDF|LSH|LSH-MinHash|LSH-Hybrid|LSH-Fused\+PageRank|LSH-Fused)\].*?Avg Latency: ([\d.]+) ms", txt)
    if baseline_matches:
        out["latency_summary"] = pd.DataFrame(
            [{"Method": m, "Avg Latency (ms)": float(lat)} for m, lat in baseline_matches]
        )

    minhash_results = re.findall(
        r"Perms=(\d+)\s+\| Indexing: ([\d.]+)s \| Avg Latency: ([\d.]+)ms \| Recall@10: ([\d.]+)%",
        txt,
    )
    if minhash_results:
        df = pd.DataFrame(
            minhash_results,
            columns=["Permutations", "Indexing Time (s)", "Latency (ms)", "Recall (%)"],
        ).apply(pd.to_numeric)
        out["minhash_sensitivity"] = df

    scale_results = re.findall(
        r"(\d+)x Corpus \((\d+) chunks\) \| LSH Index Time: ([\d.]+)s \| Memory: ([\d.]+)MB \| Query Latency: ([\d.]+)ms \| Recall@10 trend: ([\d.]+)%",
        txt,
    )
    if scale_results:
        out["scalability"] = pd.DataFrame(
            scale_results,
            columns=["Scale", "Chunks", "Index Time (s)", "Memory (MB)", "Latency (ms)", "Recall (%)"],
        ).apply(pd.to_numeric)

    qual = re.findall(
        r"\[Query (\d+)\] (.*?)\n\s+\[LLM Answer\] (.*?)\n\s+\[Evidence\] Source: (.*?) \(Pg (\d+)\) - Score: ([\d.]+)\n\s+\"(.*?)\"",
        txt,
        re.DOTALL,
    )
    out["qualitative"] = qual
    return out


def _load_pagerank_map(path):
    df = _safe_read_csv(path)
    if df is None or "chunk_id" not in df.columns or "pagerank_score" not in df.columns:
        return {}
    return {
        int(row["chunk_id"]): float(row["pagerank_score"])
        for _, row in df.iterrows()
    }


@st.cache_resource(show_spinner=True)
def load_runtime_assets():
    tfidf_vectorizer, tfidf_matrix, tfidf_chunks = load_tfidf_index()
    lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunk_tokens, lsh_chunks = load_lsh_index()
    pagerank_scores = _load_pagerank_map(PAGERANK_PATH)
    return {
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_chunks": tfidf_chunks,
        "lsh_index": lsh_index,
        "minhash_objects": minhash_objects,
        "simhash_fps": simhash_fps,
        "chunk_shingles": chunk_shingles,
        "chunk_tokens": chunk_tokens,
        "lsh_chunks": lsh_chunks,
        "pagerank_scores": pagerank_scores,
    }


def run_qa_query(query, method, top_k, assets):
    if method == "tfidf":
        return search_tfidf(
            query,
            assets["tfidf_vectorizer"],
            assets["tfidf_matrix"],
            assets["tfidf_chunks"],
            top_k=top_k,
        )
    if method == "minhash":
        return search_minhash(
            query,
            assets["lsh_index"],
            assets["minhash_objects"],
            assets["chunk_shingles"],
            assets["lsh_chunks"],
            top_k=top_k,
        )
    if method == "hybrid":
        return hybrid_search(
            query,
            assets["lsh_index"],
            assets["minhash_objects"],
            assets["simhash_fps"],
            assets["chunk_shingles"],
            assets["chunk_tokens"],
            assets["lsh_chunks"],
            top_k=top_k,
        )
    if method == "fused":
        return fused_search(
            query,
            assets["lsh_index"],
            assets["minhash_objects"],
            assets["simhash_fps"],
            assets["chunk_shingles"],
            assets["chunk_tokens"],
            assets["lsh_chunks"],
            assets["tfidf_vectorizer"],
            assets["tfidf_matrix"],
            top_k=top_k,
            candidate_pool=FUSED_DEFAULT_CANDIDATE_POOL,
        )
    if method == "fused_pagerank":
        return fused_search(
            query,
            assets["lsh_index"],
            assets["minhash_objects"],
            assets["simhash_fps"],
            assets["chunk_shingles"],
            assets["chunk_tokens"],
            assets["lsh_chunks"],
            assets["tfidf_vectorizer"],
            assets["tfidf_matrix"],
            top_k=top_k,
            candidate_pool=FUSED_DEFAULT_CANDIDATE_POOL,
            pagerank_scores=assets["pagerank_scores"],
            pagerank_weight=FUSED_DEFAULT_PAGERANK_WEIGHT,
        )
    return []


st.set_page_config(page_title="NUST QA Analytics", layout="wide")
st.title("Scalable Academic Policy QA System - Dashboard")
st.caption("Phase 4 dashboard: robust artifact-driven analytics")

report_data = parse_report_text(REPORT_PATH)
per_query = _safe_read_csv(PER_QUERY_METRICS_PATH)
pagerank = _safe_read_csv(PAGERANK_PATH)
freq = _safe_read_csv(FREQ_PATH)
dist = _safe_read_csv(DIST_PATH)

if not os.path.exists(REPORT_PATH):
    st.warning(f"`{REPORT_PATH}` is missing. Analytics tabs may be partial. Run `python src/experiments.py`.")

tab_perf, tab_sens, tab_ext, tab_qual, tab_qa = st.tabs(
    ["Performance", "Sensitivity & Scalability", "Extension Impact", "Qualitative Review", "Ask QA"]
)

with tab_perf:
    st.subheader("Latency Summary")
    if "latency_summary" in report_data:
        st.dataframe(report_data["latency_summary"], use_container_width=True)
        fig = px.bar(report_data["latency_summary"], x="Method", y="Avg Latency (ms)", title="Avg Latency by Method")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Latency summary not parsed from report.")

    st.subheader("Per-query Metrics")
    if per_query is not None:
        st.dataframe(per_query, use_container_width=True)
    else:
        st.warning("Missing per-query metrics CSV.")

with tab_sens:
    st.subheader("MinHash Sensitivity")
    if "minhash_sensitivity" in report_data:
        df = report_data["minhash_sensitivity"]
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.line(df, x="Permutations", y="Recall (%)", markers=True, title="Recall vs MinHash Permutations"), use_container_width=True)
        with col2:
            st.plotly_chart(px.line(df, x="Permutations", y="Latency (ms)", markers=True, title="Latency vs MinHash Permutations"), use_container_width=True)
    else:
        st.warning("MinHash sensitivity section not found.")

    st.subheader("Scalability")
    if dist is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.line(dist, x="chunks", y="index_time_s", markers=True, title="Distributed Index Time vs Chunks"), use_container_width=True)
        with col2:
            st.plotly_chart(px.line(dist, x="chunks", y="avg_query_latency_ms", markers=True, title="Distributed Query Latency vs Chunks"), use_container_width=True)
        st.dataframe(dist, use_container_width=True)
    elif "scalability" in report_data:
        st.dataframe(report_data["scalability"], use_container_width=True)
    else:
        st.warning("No scalability data found.")

with tab_ext:
    st.subheader("PageRank Scores")
    if pagerank is not None:
        topn = pagerank.sort_values("pagerank_score", ascending=False).head(20)
        st.dataframe(topn, use_container_width=True)
        st.plotly_chart(px.bar(topn.head(10), x="chunk_id", y="pagerank_score", title="Top 10 PageRank Chunks"), use_container_width=True)
    else:
        st.warning("Missing pagerank_scores.csv")

    st.subheader("Frequent Itemsets")
    if freq is not None:
        st.dataframe(freq, use_container_width=True)
    else:
        st.warning("Missing frequent_itemsets_report.csv")

with tab_qual:
    st.subheader("Qualitative Output Review")
    quals = report_data.get("qualitative", [])
    if not quals:
        st.warning("Could not parse qualitative section from report.")
    else:
        for q_num, q_text, answer, source, page, score, snippet in quals:
            with st.expander(f"Q{q_num}: {q_text}"):
                st.markdown(f"**Answer:** {answer}")
                st.info(f"Evidence: {source} (Page {page}) | Score={score}")
                st.caption(snippet)

with tab_qa:
    st.subheader("Interactive QA")
    st.caption("Ask handbook questions directly from the dashboard.")

    try:
        assets = load_runtime_assets()
    except Exception as e:
        st.error("Could not load runtime indices for QA.")
        st.code(str(e))
        assets = None

    if assets is not None:
        method = st.selectbox(
            "Retrieval method",
            options=["fused_pagerank", "fused", "hybrid", "minhash", "tfidf"],
            index=0,
        )
        top_k = st.slider("Top-k retrieved chunks", min_value=3, max_value=10, value=5, step=1)
        query = st.text_input("Question", placeholder="What is the attendance policy?")
        run = st.button("Run QA", type="primary")

        if run and query.strip():
            with st.spinner("Retrieving evidence..."):
                results = run_qa_query(query.strip(), method, top_k, assets)

            st.markdown(f"**Method used:** `{method}`")
            if not results:
                st.warning("No retrieval results found.")
            else:
                if os.environ.get("GROQ_API_KEY"):
                    with st.spinner("Generating grounded answer..."):
                        answer = generate_answer(query.strip(), results)
                    st.markdown("### Answer")
                    st.write(answer)
                else:
                    st.info("`GROQ_API_KEY` not set. Showing evidence-only mode.")

                st.markdown("### Top-k Evidence")
                evidence_df = pd.DataFrame(
                    [
                        {
                            "rank": idx + 1,
                            "source": item.get("source"),
                            "page": item.get("page"),
                            "score": item.get("score"),
                            "chunk_id": item.get("chunk_id"),
                        }
                        for idx, item in enumerate(results)
                    ]
                )
                st.dataframe(evidence_df, use_container_width=True)

                st.markdown("### Retrieved Snippets")
                for idx, item in enumerate(results, start=1):
                    with st.expander(
                        f"#{idx} {item.get('source')} (Page {item.get('page')}) | Score={item.get('score')}"
                    ):
                        st.write(item.get("text", ""))

st.sidebar.markdown("### Inputs")
st.sidebar.write(f"- Report: `{REPORT_PATH}`")
st.sidebar.write(f"- Per-query: `{PER_QUERY_METRICS_PATH}`")
st.sidebar.write(f"- PageRank: `{PAGERANK_PATH}`")
st.sidebar.write(f"- Frequent itemsets: `{FREQ_PATH}`")
st.sidebar.write(f"- Distributed scaling: `{DIST_PATH}`")