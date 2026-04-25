# Scalable Academic Policy QA System

Retrieval-first Question Answering system over UG/PG handbook documents using Big Data techniques.

This project is designed for academic policy question answering with a strict retrieval pipeline:
- exact baseline retrieval (`TF-IDF + cosine`),
- approximate retrieval (`MinHash + LSH`, `SimHash`),
- hybrid and fused reranking,
- grounded answer generation using retrieved evidence only.

## Table of Contents
- Overview
- System Architecture
- Repository Layout
- Features
- Setup
- Data and Inputs
- Runbook
- Outputs and Artifacts
- Evaluation
- Extensions Implemented
- Demo and Presentation
- Reproducibility
- Troubleshooting
- Project Status

## Overview

### Problem
Students ask policy questions (GPA, attendance, probation, course repeat, graduation, etc.).  
The system retrieves relevant policy chunks from handbook documents and produces evidence-grounded answers.

### Core Objective
Demonstrate approximate-vs-exact retrieval tradeoffs while maintaining scalable performance and explainable outputs.

## System Architecture

1. **Ingestion**
   - Parse handbook PDFs
   - Clean text and split into chunks
2. **Indexing**
   - Exact baseline: TF-IDF matrix
   - Approximate indexes: MinHash-LSH and SimHash fingerprints
3. **Retrieval**
   - Baseline search (TF-IDF)
   - MinHash candidate search
   - Hybrid candidate scoring (Jaccard + MinHash + Hamming + token overlap)
   - Fused reranking with TF-IDF and optional PageRank prior
4. **Answer Generation**
   - LLM response from top-k retrieved evidence (if `GROQ_API_KEY` available)
5. **Evaluation + Reporting**
   - Recall@k, Precision@k, latency, memory, scalability, qualitative analysis
6. **Interface**
   - Interactive CLI (`src/qa_cli.py`)
   - Streamlit dashboard (`dash.py`)

## Repository Layout

```text
.
├── data/
│   └── chunks/                         # generated chunk corpus
├── docs/
│   └── baselines/                      # phase locks and baseline snapshots
├── experiments/
│   └── results/                        # reports, CSVs, evaluation artifacts
├── src/
│   ├── data_ingestion.py
│   ├── indexing.py
│   ├── retrieval.py
│   ├── lsh_indexing.py
│   ├── lsh_retrieval.py
│   ├── experiments.py
│   ├── evaluate_all.py
│   ├── answer_generation.py
│   ├── qa_cli.py
│   └── extensions/
│       ├── frequent_patterns.py
│       ├── section_graph.py
│       ├── pagerank_ranker.py
│       └── distributed_sim.py
├── dash.py
├── run_pipeline.py
├── REPORT.md
├── DEMO_SCRIPT.md
├── PRESENTATION_SCRIPT.md
└── SLIDES_OUTLINE.md
```

## Features

- **Exact baseline:** TF-IDF retrieval.
- **Approximate retrieval:** MinHash-LSH and SimHash.
- **Hybrid retrieval:** combines lexical + approximate signals.
- **Fused reranking:** TF-IDF + hybrid score + intent boost + PageRank prior.
- **Extension 1:** Frequent Itemset Mining over query patterns.
- **Extension 2:** PageRank ranking of section/chunk graph.
- **Extension 3:** MapReduce/SON-style distributed simulation.
- **Productized interface:** interactive CLI + visual analytics dashboard.

## Setup

### Prerequisites
- Python 3.10+ recommended
- macOS/Linux shell (zsh/bash)

### 1) Clone and install dependencies

```bash
git clone <your-repo-url>
cd scalable_qa_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit
```

### 2) Environment configuration

```bash
cp .env.example .env
```

Set:
- `GROQ_API_KEY=<your_key>` for LLM answer generation
- leave empty to run retrieval-only flow

## Data and Inputs

- Source handbooks (UG/PG PDFs) are ingested by `src/data_ingestion.py`.
- Output chunks are written to `data/chunks/chunks.json`.

If you replace handbook PDFs, rerun the full pipeline to regenerate all indexes and reports.

## Runbook

### Full end-to-end pipeline

```bash
python run_pipeline.py
```

### Run experiments only

```bash
python src/experiments.py
```

### Run qualitative evaluation log

```bash
python src/evaluate_all.py
```

### Launch interactive CLI

```bash
python src/qa_cli.py
```

Single query mode:

```bash
python src/qa_cli.py --query "What is attendance policy?" --method fused_pagerank --top-k 5
```

### Launch dashboard

```bash
streamlit run dash.py
```

## Outputs and Artifacts

Key generated files under `experiments/results`:
- `comprehensive_report.txt`
- `evaluation_log.txt`
- `per_query_metrics.csv`
- `query_log.csv`
- `frequent_itemsets_report.csv`
- `frequent_itemsets_report.txt`
- `pagerank_scores.csv`
- `distributed_scaling_report.csv`
- `distributed_scaling_report.txt`
- `son_itemsets_report.csv`
- `son_itemsets_report.txt`

## Evaluation

### Quantitative
- Recall@k (`k=3,5,10`)
- Precision@k (`k=3,5,10`)
- Query latency (ms)
- Index memory footprint (MB)
- Scalability trend (1x, 2x, 5x, 10x, 20x)

### Qualitative
- 15 curated policy queries
- evidence-backed answer inspection
- failure/edge case behavior discussion

## Extensions Implemented

1. **Frequent Itemset Mining**
   - detects common co-occurring query patterns
   - supports interpretation of user policy intent clusters

2. **PageRank over Sections**
   - graph-based global importance scores
   - integrated as retrieval prior in fused reranking

3. **MapReduce/SON Simulation**
   - shard-level index simulation
   - distributed query and frequent itemset analysis
   - reports computational trend under synthetic scale

## Demo and Presentation

- Demo runbook: `DEMO_SCRIPT.md`
- Presentation speaking notes: `PRESENTATION_SCRIPT.md`
- Slide-by-slide structure: `SLIDES_OUTLINE.md`
- Full submission report draft: `REPORT.md`

## Reproducibility

For retrieval-only deterministic run:

```bash
GROQ_API_KEY='' python run_pipeline.py
```

Phase locks are documented in `docs/baselines`.

## Troubleshooting

- **Missing report files in dashboard**
  - Run `python src/experiments.py` first.
- **No LLM answer in CLI**
  - Set `GROQ_API_KEY` in `.env`.
- **NLTK resource errors**
  - Re-run ingestion/indexing once to trigger downloads, or install required corpora manually.
- **Streamlit command not found**
  - `pip install streamlit`.

## Project Status

- Phase 0: Baseline freeze complete
- Phase 1: Retrieval improvements complete
- Phase 2: Extension tracks complete
- Phase 3: Evaluation suite complete
- Phase 4: Productization complete
- Phase 5: Submission documentation pack complete
