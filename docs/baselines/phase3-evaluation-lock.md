# Phase 3 Evaluation Lock

Date: 2026-04-25

## Scope Completed

Phase 3 (Evaluation Upgrade) is implemented in `src/experiments.py` and verified via:

`GROQ_API_KEY='' python src/experiments.py`

## Implemented Requirements

1. Metrics completion
   - Added Precision@k and Recall@k for `k={3,5,10}`.
   - Report includes mean and standard deviation across 15 queries.
   - Per-query table exported to:
     - `experiments/results/per_query_metrics.csv`

2. Comparative experiments
   - Baseline TF-IDF vs MinHash LSH vs Hybrid vs Fused reranking reported.
   - Added missing ablations:
     - stopwords on/off in LSH preprocessing,
     - word shingles vs char shingles (`k=4,5,6`).
   - Existing sensitivity sweeps retained:
     - number of hash functions,
     - LSH threshold,
     - SimHash threshold.

3. Scalability suite
   - Extended multipliers to `1x, 2x, 5x, 10x, 20x`.
   - Reports:
     - index build time,
     - memory usage,
     - query latency,
     - quality trend (`Recall@10`) under scaling.

## Key Output Artifacts

- `experiments/results/comprehensive_report.txt`
- `experiments/results/per_query_metrics.csv`

These artifacts are ready to use directly in report tables/plots and presentation slides.
