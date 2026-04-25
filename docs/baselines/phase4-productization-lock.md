# Phase 4 Productization Lock

## Scope Completed
- Final interface flow implemented via `src/qa_cli.py`.
- Dashboard hardened and expanded in `dash.py`.
- Demo runbook added in `DEMO_SCRIPT.md`.

## 4.1 Final Interface Flow (CLI)
- Supports live query input and single-query mode (`--query`).
- Supports retrieval method selection:
  - `tfidf`
  - `minhash`
  - `hybrid`
  - `fused`
  - `fused_pagerank`
- Shows:
  - grounded answer (if `GROQ_API_KEY` set),
  - top-k retrieved chunks,
  - source/page references,
  - retrieval method used.

## 4.2 Dashboard Hardening
- Parser made resilient with structured CSV loading + regex fallback from report text.
- Added tabs:
  - Performance
  - Sensitivity & Scalability
  - Extension Impact
  - Qualitative Review
- Visualizes extension artifacts:
  - `pagerank_scores.csv`
  - `frequent_itemsets_report.csv`
  - `distributed_scaling_report.csv`

## 4.3 Demo Script
- Curated 8+ queries included.
- Includes edge-case query and expected interpretation.
- Includes exact commands for CLI and dashboard startup.

## Verification Commands
- `python -m py_compile src/qa_cli.py dash.py`
- `python src/qa_cli.py --query "What is attendance policy?" --method fused_pagerank --top-k 3`
- `streamlit run dash.py`

## Status
Phase 4 is complete and locked for submission demo usage.
