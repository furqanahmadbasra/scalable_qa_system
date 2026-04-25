from typing import Dict, List

from extensions.section_graph import build_section_graph


def compute_pagerank(
    graph: Dict[int, Dict[int, float]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Dict[int, float]:
    nodes = list(graph.keys())
    n = len(nodes)
    if n == 0:
        return {}

    ranks = {node: 1.0 / n for node in nodes}
    out_weight_sum = {node: sum(graph[node].values()) for node in nodes}

    for _ in range(max_iter):
        new_ranks = {node: (1.0 - damping) / n for node in nodes}
        for src in nodes:
            if out_weight_sum[src] <= 0.0:
                # Distribute dangling node weight uniformly
                spread = damping * ranks[src] / n
                for dst in nodes:
                    new_ranks[dst] += spread
                continue

            for dst, w in graph[src].items():
                contrib = damping * ranks[src] * (w / out_weight_sum[src])
                new_ranks[dst] += contrib

        delta = sum(abs(new_ranks[node] - ranks[node]) for node in nodes)
        ranks = new_ranks
        if delta < tol:
            break

    total = sum(ranks.values()) or 1.0
    return {k: v / total for k, v in ranks.items()}


def build_pagerank_scores(chunks: List[dict]) -> Dict[int, float]:
    graph = build_section_graph(chunks)
    return compute_pagerank(graph)

