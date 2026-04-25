import math
import time
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple

from datasketch import MinHashLSH

from lsh_indexing import clean_tokens, compute_minhash, make_shingles
from lsh_retrieval import jaccard


def _split_into_shards(items: List[dict], shard_count: int) -> List[List[dict]]:
    shards = [[] for _ in range(shard_count)]
    for idx, item in enumerate(items):
        shards[idx % shard_count].append(item)
    return shards


def _build_local_shard_index(shard_chunks: List[dict], threshold: float, num_perm: int):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhash_map = {}
    shingle_map = {}
    for c in shard_chunks:
        cid = c["chunk_id"]
        sh = make_shingles(clean_tokens(c["text"]))
        mh = compute_minhash(sh, num_perm=num_perm)
        lsh.insert(str(cid), mh)
        minhash_map[cid] = mh
        shingle_map[cid] = sh
    return lsh, minhash_map, shingle_map


def distributed_lsh_query(
    query: str,
    shard_indexes,
    shard_minhash_maps,
    shard_shingle_maps,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    q_shingles = make_shingles(clean_tokens(query))
    q_mh = compute_minhash(q_shingles)

    # Map stage: retrieve local candidates per shard
    map_candidates = set()
    for lsh, mh_map in zip(shard_indexes, shard_minhash_maps):
        local = lsh.query(q_mh)
        if not local:
            fallback = sorted(mh_map.items(), key=lambda x: -x[1].jaccard(q_mh))[:20]
            local = [str(cid) for cid, _ in fallback]
        map_candidates.update(int(cid) for cid in local)

    # Reduce stage: global rerank by exact jaccard using shard-local shingle maps
    merged_shingles = {}
    for sh_map in shard_shingle_maps:
        merged_shingles.update(sh_map)

    ranked = []
    for cid in map_candidates:
        ranked.append((cid, jaccard(q_shingles, merged_shingles[cid])))
    ranked.sort(key=lambda x: -x[1])
    return ranked[:top_k]


def simulate_distributed_lsh(
    base_chunks: List[dict],
    test_queries: List[str],
    multipliers: List[int],
    shard_count: int = 4,
    threshold: float = 0.2,
    num_perm: int = 128,
) -> List[dict]:
    rows = []
    for m in multipliers:
        scaled = []
        new_cid = 0
        for _ in range(m):
            for c in base_chunks:
                cc = dict(c)
                cc["chunk_id"] = new_cid
                scaled.append(cc)
                new_cid += 1

        shards = _split_into_shards(scaled, shard_count)

        t0 = time.time()
        shard_indexes = []
        shard_mh_maps = []
        shard_sh_maps = []
        for shard in shards:
            idx, mh_map, sh_map = _build_local_shard_index(
                shard, threshold=threshold, num_perm=num_perm
            )
            shard_indexes.append(idx)
            shard_mh_maps.append(mh_map)
            shard_sh_maps.append(sh_map)
        index_time = time.time() - t0

        query_latencies = []
        avg_candidates = []
        for q in test_queries:
            t1 = time.perf_counter()
            ranked = distributed_lsh_query(
                q,
                shard_indexes,
                shard_mh_maps,
                shard_sh_maps,
                top_k=10,
            )
            query_latencies.append(time.perf_counter() - t1)
            avg_candidates.append(len(ranked))

        rows.append(
            {
                "multiplier": m,
                "chunks": len(scaled),
                "shards": shard_count,
                "index_time_s": index_time,
                "avg_query_latency_ms": (sum(query_latencies) / len(query_latencies)) * 1000,
                "avg_topk_size": sum(avg_candidates) / len(avg_candidates),
            }
        )
    return rows


def _all_subsets_up_to_k(tx: Set[str], max_k: int):
    vals = sorted(tx)
    for k in range(1, max_k + 1):
        for comb in combinations(vals, k):
            yield comb


def son_frequent_itemsets(
    transactions: List[Set[str]],
    global_min_support_count: int = 2,
    max_k: int = 3,
    shard_count: int = 4,
) -> Dict[int, List[Tuple[Tuple[str, ...], int, float]]]:
    if not transactions:
        return {}

    shards = _split_into_shards([{"tx": t} for t in transactions], shard_count)
    shard_txs = [[r["tx"] for r in shard] for shard in shards]
    total_tx = len(transactions)

    # Pass 1: local candidate generation
    global_candidates = set()
    for shard in shard_txs:
        local_min = max(1, math.ceil(global_min_support_count * (len(shard) / total_tx)))
        local_counts = Counter()
        for tx in shard:
            for itemset in _all_subsets_up_to_k(tx, max_k):
                local_counts[itemset] += 1
        for itemset, cnt in local_counts.items():
            if cnt >= local_min:
                global_candidates.add(itemset)

    # Pass 2: global counting
    global_counts = Counter()
    for tx in transactions:
        for itemset in global_candidates:
            if set(itemset).issubset(tx):
                global_counts[itemset] += 1

    result = defaultdict(list)
    for itemset, cnt in global_counts.items():
        if cnt >= global_min_support_count:
            k = len(itemset)
            result[k].append((itemset, cnt, cnt / total_tx))
    for k in result:
        result[k].sort(key=lambda x: (-x[1], x[0]))
    return dict(result)

