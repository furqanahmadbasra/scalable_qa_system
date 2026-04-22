import streamlit as st
import pandas as pd
import plotly.express as px
import re
import os

# Configuration
REPORT_PATH = "experiments/results/comprehensive_report.txt"

def parse_report(path):
    if not os.path.exists(path):
        return None
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    data = {}

    # 1. Extract Exact vs Approx Comparison
    tfidf_match = re.search(r"\[TF-IDF\] Build Time: ([\d.]+)s \| Memory: ([\d.]+) MB \|\s+Avg Latency: ([\d.]+) ms", content)
    lsh_match = re.search(r"\[LSH\]\s+Build Time: ([\d.]+)s \| Memory: ([\d.]+) MB \|\s+Avg Latency: ([\d.]+) ms", content)
    recall_match = re.search(r"Average Recall@10 \(vs Exact Jaccard\): ([\d.]+)%", content)

    if tfidf_match and lsh_match:
        data['baseline'] = {
            "Metric": ["Build Time (s)", "Memory (MB)", "Avg Latency (ms)", "Recall@10"],
            "TF-IDF (Exact)": [float(tfidf_match.group(1)), float(tfidf_match.group(2)), float(tfidf_match.group(3)), 100.0],
            "LSH (Approx)": [float(lsh_match.group(1)), float(lsh_match.group(2)), float(lsh_match.group(3)), float(recall_match.group(1)) if recall_match else 0.0]
        }

    # 2. Extract MinHash Sensitivity
    minhash_results = re.findall(r"Perms=(\d+)\s+\| Indexing: ([\d.]+)s \| Avg Latency: ([\d.]+)ms \| Recall@10: ([\d.]+)%", content)
    if minhash_results:
        data['minhash_sensitivity'] = pd.DataFrame(minhash_results, columns=["Permutations", "Indexing Time (s)", "Latency (ms)", "Recall (%)"])
        data['minhash_sensitivity'] = data['minhash_sensitivity'].apply(pd.to_numeric)

    # 3. Extract Scalability Data
    scale_results = re.findall(r"(\d)x Corpus \((\d+) chunks\) \| LSH Index Time: ([\d.]+)s \| Query Latency: ([\d.]+)ms", content)
    if scale_results:
        data['scalability'] = pd.DataFrame(scale_results, columns=["Scale", "Chunks", "Index Time (s)", "Latency (ms)"])
        data['scalability'] = data['scalability'].apply(pd.to_numeric)

    # 4. Extract Qualitative Queries
    queries = re.findall(r"\[Query (\d+)\] (.*?)\n\s+\[LLM Answer\] (.*?)\n\s+\[Evidence\] (.*?)\n\s+\"(.*?)\"", content, re.DOTALL)
    data['queries'] = queries

    return data

# --- Streamlit UI ---
st.set_page_config(page_title="NUST QA Analytics", layout="wide")

st.title("🏛️ Scalable Academic Policy QA System")
st.subheader("Experimental Results & Performance Analysis")

report_data = parse_report(REPORT_PATH)

if not report_data:
    st.error(f"Could not find report file at `{REPORT_PATH}`. Please run the experimental pipeline first.")
else:
    tab1, tab2, tab3 = st.tabs(["🚀 Performance Metrics", "⚙️ Parameter Sensitivity", "📝 Qualitative Review"])

    with tab1:
        st.header("1. Exact vs. Approximate Retrieval")
        # Display as a clean dataframe
        comp_df = pd.DataFrame(report_data['baseline']).set_index("Metric")
        st.dataframe(comp_df, use_container_width=True)
        
        st.markdown("---")
        st.header("2. Scalability Test (LSH Performance)")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_idx = px.line(report_data['scalability'], x="Chunks", y="Index Time (s)", 
                              title="Index Construction Time vs. Dataset Size",
                              labels={"Index Time (s)": "Time (Seconds)", "Chunks": "Total Chunks"},
                              markers=True)
            st.plotly_chart(fig_idx, use_container_width=True)
        
        with col2:
            fig_lat = px.line(report_data['scalability'], x="Chunks", y="Latency (ms)", 
                              title="Query Latency vs. Dataset Size",
                              labels={"Latency (ms)": "Latency (Milliseconds)", "Chunks": "Total Chunks"},
                              markers=True)
            # Ensure Y-axis starts near data to show stability
            fig_lat.update_yaxes(range=[0, max(report_data['scalability']['Latency (ms)']) * 1.5])
            st.plotly_chart(fig_lat, use_container_width=True)

    with tab2:
        st.header("3. MinHash Parameter Sensitivity")
        st.write("Analyzing how the number of hash functions (permutations) affects accuracy and retrieval speed.")
        
        c1, c2 = st.columns(2)
        with c1:
            fig_recall = px.line(report_data['minhash_sensitivity'], x="Permutations", y="Recall (%)",
                                 title="Recall @ 10 vs. Number of Hash Functions",
                                 labels={"Recall (%)": "Recall Accuracy (%)", "Permutations": "Hash Functions"},
                                 markers=True)
            st.plotly_chart(fig_recall, use_container_width=True)
        
        with c2:
            fig_sens_lat = px.line(report_data['minhash_sensitivity'], x="Permutations", y="Latency (ms)",
                                   title="Retrieval Latency vs. Number of Hash Functions",
                                   labels={"Latency (ms)": "Latency (Milliseconds)", "Permutations": "Hash Functions"},
                                   markers=True)
            st.plotly_chart(fig_sens_lat, use_container_width=True)

    with tab3:
        st.header("4. Qualitative Review")
        for q_num, q_text, a_text, evidence, snippet in report_data['queries']:
            with st.expander(f"Q{q_num}: {q_text}"):
                st.markdown(f"**🤖 LLM Answer:**\n{a_text}")
                st.info(f"**📄 Evidence Reference:** {evidence}")
                st.markdown("**🔍 Context Snippet:**")
                st.caption(snippet)

st.sidebar.info("Data extracted from the most recent system evaluation log.")