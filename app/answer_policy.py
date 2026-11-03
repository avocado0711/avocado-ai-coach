"""Choose a concise or structured coach response without relying only on the LLM."""

from __future__ import annotations

import re

PLAN_PATTERN = re.compile(
    r"训练计划|怎么安排|帮我排|排一[个份]|分化|训练日|每周.*(?:练|训练)"
)
STRUCTURED_PATTERN = re.compile(
    r"疼|痛|肿胀|卡住|麻木|无力|受伤|伤病|康复|恢复|纠正|"
    r"风险|替代动作|带伤|膝|腰|肩|踝|腕|肘|髋"
)


def select_answer_mode(query: str) -> str:
    text = str(query or "").strip()
    if PLAN_PATTERN.search(text):
        return "structured"
    if STRUCTURED_PATTERN.search(text):
        return "structured"
    return "simple"


def mode_instruction(mode: str) -> str:
    if mode == "structured":
        return (
            "本次问题涉及多步骤或伤病风险，请使用固定小标题结构化回答。"
            "不要输出【结合档案】。包含训练安排时使用【训练计划】和 | 分隔表格。"
        )
    return (
        "本次是简单信息问题。只输出 2 到 4 句自然短段落，直接回答问题；"
        "禁止使用【】小标题、编号列表、表格、状态卡片式分段。"
        "若确有风险，最后最多补一句停止或就医提醒。"
    )
