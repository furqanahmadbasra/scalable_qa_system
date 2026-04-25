from collections import defaultdict
from typing import Dict, List

from lsh_indexing import clean_tokens


def _token_jaccard(tokens_a, tokens_b) -> float:
    a, b = set(tokens_a), set(tokens_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_section_graph(
    chunks: List[dict],
    lexical_threshold: float = 0.08,
    top_similar_edges: int = 3,
    adjacency_weight: float = 0.6,
    lexical_weight: float = 0.4,
) -> Dict[int, Dict[int, float]]:
    """
    Build a weighted chunk graph for PageRank:
    - structural edges between nearby chunks in same source
    - lexical similarity edges for semantically related sections
    """
    graph = defaultdict(dict)
    by_source = defaultdict(list)
    chunk_tokens = {}

    for c in chunks:
        cid = c["chunk_id"]
        by_source[c["source"]].append(c)
        chunk_tokens[cid] = clean_tokens(c["text"])

    # Structural adjacency edges (same handbook, neighboring pages/chunks)
    for source_chunks in by_source.values():
        source_chunks = sorted(source_chunks, key=lambda x: (x.get("page", 0), x["chunk_id"]))
        for i in range(len(source_chunks) - 1):
            a = source_chunks[i]["chunk_id"]
            b = source_chunks[i + 1]["chunk_id"]
            graph[a][b] = graph[a].get(b, 0.0) + adjacency_weight
            graph[b][a] = graph[b].get(a, 0.0) + adjacency_weight

    # Lexical similarity edges
    ids = [c["chunk_id"] for c in chunks]
    for cid in ids:
        sims = []
        for other in ids:
            if cid == other:
                continue
            sim = _token_jaccard(chunk_tokens[cid], chunk_tokens[other])
            if sim >= lexical_threshold:
                sims.append((other, sim))
        sims.sort(key=lambda x: -x[1])
        for other, sim in sims[:top_similar_edges]:
            w = lexical_weight * sim
            graph[cid][other] = graph[cid].get(other, 0.0) + w
            graph[other][cid] = graph[other].get(cid, 0.0) + w

    # Ensure isolated nodes still exist in graph
    for cid in ids:
        _ = graph[cid]

    return dict(graph)

