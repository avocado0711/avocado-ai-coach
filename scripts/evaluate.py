"""Offline retrieval benchmark using the same retriever as production."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from app.config import RETRIEVAL_TOP_K
from app.embed_rerank import (
    clear_retrieval_cache,
    collection,
    retrieval_status,
    retrieve,
)
from app.safety import assess_query

EVAL_FILE = PROJECT_ROOT / "evaluation" / "retrieval.jsonl"
TRAIN_FILE = PROJECT_ROOT / "evaluation" / "train_triplets.jsonl"
REPORT_DIR = PROJECT_ROOT / "reports"
TOP_K = RETRIEVAL_TOP_K


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number} JSON 无效: {exc}") from exc
            if "gold_doc_ids" not in item:
                item["gold_doc_ids"] = (
                    [item["gold_doc_id"]] if item.get("gold_doc_id") else []
                )
            item.setdefault("category", "retrieval")
            items.append(item)
    return items


def _first_relevant_rank(results: list[dict], gold: set[str], k: int) -> int:
    for rank, result in enumerate(results[:k], start=1):
        if result["doc_id"] in gold:
            return rank
    return 0


def _ndcg_at_k(results: list[dict], gold: set[str], k: int) -> float:
    dcg = 0.0
    for rank, result in enumerate(results[:k], start=1):
        if result["doc_id"] in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _normalize(text: str) -> str:
    return "".join(
        character.lower()
        for character in text
        if character.isalnum()
    )


def _char_bigrams(text: str) -> set[str]:
    normalized = _normalize(text)
    return {
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
    }


def check_train_eval_leakage(eval_items: list[dict]) -> dict:
    if not TRAIN_FILE.exists():
        return {"exact_overlap": [], "train_queries": 0}
    train_items = load_jsonl(TRAIN_FILE)
    train_queries = {
        _normalize(item.get("query", "")): item.get("query", "")
        for item in train_items
    }
    train_bigrams = [
        (_char_bigrams(item.get("query", "")), item.get("query", ""))
        for item in train_items
    ]
    overlap = []
    near_overlap = []
    for item in eval_items:
        normalized = _normalize(item.get("query", ""))
        if normalized and normalized in train_queries:
            overlap.append(
                {
                    "eval_query": item["query"],
                    "train_query": train_queries[normalized],
                }
            )
            continue
        query_bigrams = _char_bigrams(item.get("query", ""))
        if not query_bigrams:
            continue
        best_score = 0.0
        best_train_query = ""
        for candidate, train_query in train_bigrams:
            union = query_bigrams | candidate
            score = len(query_bigrams & candidate) / len(union) if union else 0.0
            if score > best_score:
                best_score = score
                best_train_query = train_query
        if best_score >= 0.5:
            near_overlap.append(
                {
                    "eval_query": item["query"],
                    "train_query": best_train_query,
                    "jaccard": round(best_score, 3),
                }
            )
    return {
        "exact_overlap": overlap,
        "near_overlap": near_overlap,
        "train_queries": len(train_items),
        "warning": (
            "存在训练/评测重复或高度相似问题，相关指标会偏乐观"
            if overlap or near_overlap
            else ""
        ),
    }


def _bootstrap_interval(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    if not values or samples <= 0:
        return None
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[generator.randrange(len(values))] for _ in values]
        means.append(statistics.mean(draw))
    means.sort()
    lower = means[max(0, math.floor(len(means) * 0.025) - 1)]
    upper = means[min(len(means) - 1, math.ceil(len(means) * 0.975) - 1)]
    return [round(lower, 6), round(upper, 6)]


def _run_mode(
    items: list[dict],
    mode: str,
    *,
    bootstrap_samples: int,
) -> tuple[dict, list[float]]:
    hits = 0
    reciprocal_ranks = []
    ndcgs = []
    precisions = []
    latencies = []
    by_category = defaultdict(lambda: {"n": 0, "hits": 0})

    for item in items:
        gold = set(item.get("gold_doc_ids", []))
        if not gold:
            continue
        started = time.perf_counter()
        results = retrieve(
            item["query"],
            top_k=TOP_K,
            mode=mode,
            include_below_threshold=True,
        )
        latencies.append((time.perf_counter() - started) * 1000)
        rank = _first_relevant_rank(results, gold, TOP_K)
        hit = rank > 0
        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        ndcgs.append(_ndcg_at_k(results, gold, TOP_K))
        precisions.append(
            len({item["doc_id"] for item in results[:TOP_K]} & gold) / TOP_K
        )

        category = item.get("category", "retrieval")
        by_category[category]["n"] += 1
        by_category[category]["hits"] += int(hit)

    count = max(1, len(reciprocal_ranks))
    metrics = {
        "n": len(reciprocal_ranks),
        f"recall@{TOP_K}": hits / count,
        f"mrr@{TOP_K}": sum(reciprocal_ranks) / count,
        f"ndcg@{TOP_K}": sum(ndcgs) / count,
        f"precision@{TOP_K}": sum(precisions) / count,
        "confidence_intervals_95": {
            f"recall@{TOP_K}": _bootstrap_interval(
                [1.0 if value > 0 else 0.0 for value in reciprocal_ranks],
                samples=bootstrap_samples,
                seed=17,
            ),
            f"mrr@{TOP_K}": _bootstrap_interval(
                reciprocal_ranks,
                samples=bootstrap_samples,
                seed=23,
            ),
            f"ndcg@{TOP_K}": _bootstrap_interval(
                ndcgs,
                samples=bootstrap_samples,
                seed=29,
            ),
        },
        "latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        "latency_ms_p95": (
            sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)]
            if latencies
            else 0.0
        ),
        "by_category": dict(by_category),
    }
    return metrics, latencies


def validate_gold_ids(items: list[dict]) -> None:
    gold_ids = sorted(
        {
            doc_id
            for item in items
            for doc_id in item.get("gold_doc_ids", [])
        }
    )
    found = set()
    for start in range(0, len(gold_ids), 100):
        batch = gold_ids[start : start + 100]
        found.update(collection.get(ids=batch).get("ids", []))
    missing = sorted(set(gold_ids) - found)
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(
            f"评测集有 {len(missing)} 个 gold_doc_id 不在当前知识库中: {preview}。"
            "请在重建知识库后人工重新标注评测集。"
        )


def filter_leaky_items(items: list[dict], leakage: dict) -> tuple[list[dict], list[str]]:
    leaky_queries = {
        item["eval_query"]
        for key in ("exact_overlap", "near_overlap")
        for item in leakage.get(key, [])
    }
    if not leaky_queries:
        return items, []
    return (
        [item for item in items if item["query"] not in leaky_queries],
        sorted(leaky_queries),
    )


def evaluate_safety(items: list[dict]) -> dict:
    decisions = []
    for item in items:
        safety = assess_query(item["query"])
        expected_action = item.get("expected_action")
        should_abstain = bool(item.get("should_abstain") or expected_action)
        if expected_action in {"urgent_care", "medical_boundary"}:
            expected = expected_action
            observed = safety.action
            correct = observed == expected
        elif should_abstain:
            expected = "abstain"
            observed = "abstain" if safety.action != "allow" else "answer"
            correct = safety.action != "allow"
        else:
            expected = "answer"
            observed = "answer" if safety.action == "allow" else safety.action
            correct = safety.action == "allow"
        decisions.append(
            {
                "query": item["query"],
                "expected": expected,
                "observed": observed,
                "correct": correct,
            }
        )
    true_positive = sum(
        decision["correct"]
        for decision in decisions
        if decision["expected"] != "answer"
    )
    false_positive = sum(
        not decision["correct"]
        for decision in decisions
        if decision["expected"] == "answer"
    )
    false_negative = sum(
        not decision["correct"]
        for decision in decisions
        if decision["expected"] != "answer"
    )
    true_negative = sum(
        decision["correct"]
        for decision in decisions
        if decision["expected"] == "answer"
    )
    return {
        "n": len(decisions),
        "accuracy": (
            sum(decision["correct"] for decision in decisions) / len(decisions)
            if decisions
            else 0.0
        ),
        "block_precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        ),
        "block_recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        ),
        "false_abstention_count": false_positive,
        "correct_allow_count": true_negative,
        "cases": decisions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-leaky",
        action="store_true",
        help="Include exact/near duplicate evaluation questions for debugging.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resamples for 95%% confidence intervals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not EVAL_FILE.exists():
        raise SystemExit(f"缺少评测集: {EVAL_FILE}")

    clear_retrieval_cache()
    items = load_jsonl(EVAL_FILE)
    validate_gold_ids(items)
    leakage = check_train_eval_leakage(items)
    clean_items = items
    excluded_leaky_queries: list[str] = []
    if not args.include_leaky:
        clean_items, excluded_leaky_queries = filter_leaky_items(items, leakage)
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "top_k": TOP_K,
        "dataset_size": len(clean_items),
        "raw_dataset_size": len(items),
        "excluded_leaky_queries": excluded_leaky_queries,
        "retrieval_status": retrieval_status(),
        "leakage": leakage,
        "safety": evaluate_safety(clean_items),
        "caveats": [
            "当前检索集少于 100 条，只适合工程回归，不足以证明线上泛化能力。",
            "生成质量评测需使用独立人工评审，LLM-as-judge 不能替代用户结论。",
        ],
    }

    for mode in ("vector", "hybrid_no_rerank", "hybrid"):
        report[mode], _ = _run_mode(
            clean_items,
            mode,
            bootstrap_samples=max(0, args.bootstrap_samples),
        )

    if not report["hybrid"]["n"]:
        raise SystemExit("评测集中没有带 gold_doc_ids 的正样本")

    report["delta_hybrid_vs_vector"] = {
        f"recall@{TOP_K}": (
            report["hybrid"][f"recall@{TOP_K}"]
            - report["vector"][f"recall@{TOP_K}"]
        ),
        f"mrr@{TOP_K}": (
            report["hybrid"][f"mrr@{TOP_K}"] - report["vector"][f"mrr@{TOP_K}"]
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    report_path = REPORT_DIR / f"retrieval-{timestamp}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"评测样本: {len(clean_items)}（原始 {len(items)}）")
    if excluded_leaky_queries:
        print(f"已按防泄漏策略排除: {', '.join(excluded_leaky_queries)}")
    if report["leakage"]["warning"]:
        print(f"警告: {report['leakage']['warning']}")
    for mode in ("vector", "hybrid_no_rerank", "hybrid"):
        metrics = report[mode]
        print(
            f"{mode}: Recall@{TOP_K}={metrics[f'recall@{TOP_K}']:.4f} "
            f"MRR@{TOP_K}={metrics[f'mrr@{TOP_K}']:.4f} "
            f"nDCG@{TOP_K}={metrics[f'ndcg@{TOP_K}']:.4f} "
            f"p95={metrics['latency_ms_p95']:.1f}ms"
        )
    safety = report["safety"]
    print(
        f"安全分类准确率: {safety['accuracy']:.4f} "
        f"拦截精确率: {safety['block_precision']:.4f} "
        f"拦截召回率: {safety['block_recall']:.4f} "
        f"误拒答: {safety['false_abstention_count']}"
    )
    print(f"报告已写入: {report_path}")


if __name__ == "__main__":
    main()
