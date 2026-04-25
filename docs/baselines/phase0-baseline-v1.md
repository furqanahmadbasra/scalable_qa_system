# Phase 0 Baseline Lock (baseline-v1)

Date: 2026-04-25  
Branch: `feat/extended-functionality`

## Scope

This document freezes the project baseline for Phase 0 by recording:

- reproducibility command and observed outputs,
- baseline artifact checksums,
- locked evaluation criteria for subsequent phases.

## Reproducibility Run

Command used for deterministic baseline run:

`GROQ_API_KEY='' python run_pipeline.py`

Observed pipeline status:

- data ingestion completed (`178` chunks generated),
- TF-IDF indexing completed (matrix shape `178 x 10937`),
- MinHash/SimHash indexing completed (LSH bands/rows `b=28`, `r=2`),
- experiments completed successfully and reports regenerated.

## Baseline Artifacts (Before-Reference)

- `experiments/results/comprehensive_report.txt`
- `experiments/results/evaluation_log.txt`
- `data/chunks/chunks.json`

SHA1 checksums:

- `235c73feb82828320d195e8eea74f00d4a52ea28`  `experiments/results/comprehensive_report.txt`
- `ef3a73120a4c2fc17a23b220527466c811373389`  `experiments/results/evaluation_log.txt`
- `fc490c73e8ece918bca4574d91f2aa6ab909c9c6`  `data/chunks/chunks.json`

## Locked Acceptance Metrics (Phase 0.3)

These metrics are fixed as project acceptance criteria for later phases:

1. Retrieval quality
   - Recall@k (k = 3, 5, 10)
   - Precision@k (k = 3, 5, 10)  **[to be implemented in Phase 3]**
2. Efficiency
   - Query latency (ms/query)
3. Resource cost
   - Memory usage (MB) for indexes/models
4. Scalability trend
   - Index build time and query latency vs corpus growth (`1x, 2x, 5x`, extend to `10x, 20x` later)

## Baseline Snapshot from Current Report

- TF-IDF avg latency: `0.57 ms`
- LSH avg latency: `1.59 ms`
- LSH-MinHash Recall@10: `59.33%`
- LSH-Hybrid Recall@10: `40.67%`
- LSH-Fused Recall@10: `45.33%`

This file is the authoritative baseline reference for before/after comparisons in later phases.
