"""Optional end-to-end answer evaluation against a running pilot server.

This script makes real model calls. Use a dedicated test account and review the
cost before running it against production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import httpx
from _bootstrap import PROJECT_ROOT

from app.config import LLM_MODEL, require_llm_client

client = require_llm_client()


DEFAULT_DATA = PROJECT_ROOT / "evaluation" / "generation.jsonl"
REPORT_DIR = PROJECT_ROOT / "reports"


def parse_sse(text: str) -> tuple[str, str, list[dict]]:
    answer = ""
    answer_id = ""
    sources = []
    for raw_block in text.split("\n\n"):
        event = "message"
        data_lines = []
        for line in raw_block.splitlines():
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                data_lines.append(line[6:])
        payload = "\n".join(data_lines)
        if not payload or payload == "[DONE]" or payload.startswith("[ERROR]"):
            continue
        if event == "meta":
            meta = json.loads(payload)
            answer_id = meta.get("answer_id", "")
            sources = meta.get("sources", [])
        else:
            answer += payload
    return answer, answer_id, sources


def judge_answer(case: dict, answer: str, sources: list[dict]) -> dict:
    prompt = {
        "query": case["query"],
        "category": case.get("category", "general"),
        "rubric": case.get("rubric", ""),
        "answer": answer,
        "sources": sources,
    }
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是严格的健身应用质量评审。根据 rubric 评分，1 最差，5 最好。"
                    "输出 JSON，字段必须是 faithfulness, safety, personalization, "
                    "actionability, citation, rationale。不要输出其他文字。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ],
        temperature=0,
        timeout=30,
    )
    text = (response.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()
    result = json.loads(text)
    for key in (
        "faithfulness",
        "safety",
        "personalization",
        "actionability",
        "citation",
    ):
        result[key] = max(1, min(5, int(result[key])))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / f"generation-{time.strftime('%Y%m%d-%H%M%S')}.json",
    )
    parser.add_argument(
        "--minimum-cases",
        type=int,
        default=30,
        help="Fail when the evaluation set is too small for a pilot smoke test.",
    )
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cases) < args.minimum_cases:
        raise SystemExit(
            f"生成评测只有 {len(cases)} 条，少于要求的 {args.minimum_cases} 条。"
        )
    queries = [str(case.get("query") or "").strip() for case in cases]
    if any(not query for query in queries):
        raise SystemExit("生成评测集存在空 query")
    if len(set(queries)) != len(queries):
        raise SystemExit("生成评测集存在重复 query")
    results = []
    with httpx.Client(timeout=120) as http:
        for case in cases:
            started = time.perf_counter()
            response = http.post(
                f"{args.base_url.rstrip('/')}/api/chat",
                headers={"Authorization": f"Bearer {args.token}"},
                json={"query": case["query"]},
            )
            response.raise_for_status()
            elapsed_ms = (time.perf_counter() - started) * 1000
            answer, answer_id, sources = parse_sse(response.text)
            evaluation = judge_answer(case, answer, sources)
            results.append(
                {
                    **case,
                    "answer": answer,
                    "answer_id": answer_id,
                    "sources": sources,
                    "evaluation": evaluation,
                    "latency_ms": round(elapsed_ms, 1),
                }
            )

    score_keys = (
        "faithfulness",
        "safety",
        "personalization",
        "actionability",
        "citation",
    )
    summary = {
        key: statistics.mean(item["evaluation"][key] for item in results)
        for key in score_keys
    }
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n": len(results),
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "judge_model": LLM_MODEL,
        "human_review_required": True,
        "summary": summary,
        "latency_ms_p95": sorted(
            item["latency_ms"] for item in results
        )[max(0, math.ceil(len(results) * 0.95) - 1)],
        "cases": results,
        "warning": (
            "LLM-as-judge 只用于筛查，不能替代人工评审；"
            "上线结论必须由人工抽检确认。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"报告已写入: {args.output}")


if __name__ == "__main__":
    main()
