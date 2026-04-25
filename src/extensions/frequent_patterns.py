import csv
import os
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Set, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer


PRESERVED_POLICY_WORDS = {"not", "no", "nor", "must", "shall", "may"}
STOPWORDS = ENGLISH_STOP_WORDS - PRESERVED_POLICY_WORDS
stemmer = PorterStemmer()


def normalize_query_tokens(text: str) -> List[str]:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)

    normalized = []
    for token in text.split():
        if "-" in token:
            parts = [p for p in token.split("-") if p]
            normalized.extend(parts)
            if parts:
                normalized.append("".join(parts))
        else:
            normalized.append(token)

    tokens = []
    for token in normalized:
        if len(token) <= 1:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(stemmer.stem(token))
    return tokens


def detect_intents(tokens: List[str]) -> List[str]:
    token_set = set(tokens)
    intent_map = {
        "attendance": {"attend", "short", "xf", "absent"},
        "probation": {"probat", "warn", "withdraw"},
        "repeat_course": {"repeat", "retak", "retest", "course"},
        "hostel": {"hostel", "accommod", "allot"},
        "rechecking": {"recheck", "reassess", "exam", "paper"},
        "graduation": {"degre", "graduat", "award", "credit", "cgpa"},
        "fees": {"fee", "fine", "dues", "deposit"},
        "plagiarism": {"plagiar", "dishonesti", "cheat"},
        "medals": {"medal", "convoc", "presid", "rector"},
    }
    intents = [name for name, kws in intent_map.items() if token_set & kws]
    return intents if intents else ["general_policy"]


def build_query_log_records(queries: List[str], start_timestamp: str) -> List[Dict[str, str]]:
    records = []
    for idx, query in enumerate(queries, start=1):
        tokens = normalize_query_tokens(query)
        intents = detect_intents(tokens)
        records.append(
            {
                "query_id": idx,
                "query_text": query,
                "timestamp": f"{start_timestamp}T00:{idx:02d}:00Z",
                "normalized_tokens": tokens,
                "intent_tags": intents,
            }
        )
    return records


def build_transactions(
    query_records: List[Dict[str, str]],
    include_intents: bool = True,
) -> List[Set[str]]:
    transactions = []
    for row in query_records:
        items = set(row["normalized_tokens"])
        if include_intents:
            items.update({f"intent:{tag}" for tag in row["intent_tags"]})
        if items:
            transactions.append(items)
    return transactions


def _candidate_from_prev(prev_itemsets: List[Tuple[str, ...]], k: int) -> Set[Tuple[str, ...]]:
    prev_set = set(prev_itemsets)
    candidates = set()
    for i in range(len(prev_itemsets)):
        for j in range(i + 1, len(prev_itemsets)):
            merged = tuple(sorted(set(prev_itemsets[i]) | set(prev_itemsets[j])))
            if len(merged) != k:
                continue
            if all(tuple(sorted(sub)) in prev_set for sub in combinations(merged, k - 1)):
                candidates.add(merged)
    return candidates


def apriori_frequent_itemsets(
    transactions: List[Set[str]],
    min_support_count: int = 2,
    max_k: int = 3,
) -> Dict[int, List[Tuple[Tuple[str, ...], int, float]]]:
    if not transactions:
        return {}

    total_tx = len(transactions)
    results: Dict[int, List[Tuple[Tuple[str, ...], int, float]]] = {}

    token_counter = Counter()
    for tx in transactions:
        token_counter.update(tx)

    l1 = []
    for token, cnt in token_counter.items():
        if cnt >= min_support_count:
            l1.append(((token,), cnt, cnt / total_tx))
    l1.sort(key=lambda x: (-x[1], x[0]))
    if not l1:
        return {}
    results[1] = l1

    prev_level = [item for item, _, _ in l1]
    for k in range(2, max_k + 1):
        candidates = _candidate_from_prev(prev_level, k)
        if not candidates:
            break

        counts = Counter()
        for tx in transactions:
            for cand in candidates:
                if set(cand).issubset(tx):
                    counts[cand] += 1

        level = []
        for cand, cnt in counts.items():
            if cnt >= min_support_count:
                level.append((cand, cnt, cnt / total_tx))
        level.sort(key=lambda x: (-x[1], x[0]))
        if not level:
            break
        results[k] = level
        prev_level = [item for item, _, _ in level]

    return results


def write_query_log_csv(query_records: List[Dict[str, str]], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_id", "query_text", "timestamp", "normalized_tokens", "intent_tags"],
        )
        writer.writeheader()
        for row in query_records:
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "query_text": row["query_text"],
                    "timestamp": row["timestamp"],
                    "normalized_tokens": "|".join(row["normalized_tokens"]),
                    "intent_tags": "|".join(row["intent_tags"]),
                }
            )


def write_itemsets_csv(
    itemsets: Dict[int, List[Tuple[Tuple[str, ...], int, float]]], output_file: str
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["k", "itemset", "support_count", "support_ratio"],
        )
        writer.writeheader()
        for k in sorted(itemsets.keys()):
            for items, support_count, support_ratio in itemsets[k]:
                writer.writerow(
                    {
                        "k": k,
                        "itemset": " | ".join(items),
                        "support_count": support_count,
                        "support_ratio": round(support_ratio, 4),
                    }
                )


def write_itemsets_text_report(
    itemsets: Dict[int, List[Tuple[Tuple[str, ...], int, float]]], output_file: str
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("FREQUENT ITEMSET MINING REPORT\n")
        f.write("================================\n\n")
        if not itemsets:
            f.write("No frequent itemsets discovered.\n")
            return

        for k in sorted(itemsets.keys()):
            f.write(f"k={k} itemsets\n")
            f.write("-" * 40 + "\n")
            for items, support_count, support_ratio in itemsets[k]:
                f.write(
                    f"{' | '.join(items)} :: support_count={support_count}, support_ratio={support_ratio:.2f}\n"
                )
            f.write("\n")

