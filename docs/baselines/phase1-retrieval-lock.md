# Phase 1 Retrieval Lock

Date: 2026-04-25

## Phase 1 Status

Phase 1 retrieval core improvements are complete and locked.

## Locked Retrieval Decisions

1. TF-IDF baseline remains unchanged:
   - word-level preprocessing
   - stopword removal
   - stemming

2. LSH preprocessing:
   - stopwords removed with policy-word preservation (`not`, `no`, `nor`, `must`, `shall`, `may`)
   - `_` normalized to space
   - hyphen tokens preserve split and joined variants

3. Feature representation:
   - MinHash shingles: word unigrams + bigrams
   - SimHash: token-based features (no char shingles in production path)

4. Retrieval defaults:
   - `num_perm = 128`
   - LSH threshold default `0.2` (with sensitivity tested in `0.1-0.3`)
   - SimHash bits `64`
   - SimHash retrieval default threshold `12`

5. Scoring defaults:
   - Hybrid score uses Jaccard + MinHash estimate + SimHash + token overlap
   - Fused reranker defaults:
     - TF-IDF weight `0.50`
     - Hybrid weight `0.50`
     - Intent boost enabled
     - Candidate pool `50`

## Verification Snapshot

Run command:

`GROQ_API_KEY='' python src/experiments.py`

Expected quality/efficiency profile (approximate):

- LSH-Fused Recall@10 around `46%`
- LSH latency still in low milliseconds
- no major regression in runtime compared to pre-lock runs

This lock establishes the submission candidate retrieval configuration for subsequent phases.
