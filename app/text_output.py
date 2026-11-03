"""Normalize model output for a plain-text chat surface."""

from __future__ import annotations

import re

INTERNAL_META_SENTENCE = re.compile(
    r"[^。！？\n]*(?:"
    r"知识库|"
    r"检索到|"
    r"命中(?:了)?[^。！？\n]{0,12}(?:资料|片段)|"
    r"参考(?:了)?[^。！？\n]{0,12}(?:资料|片段)"
    r")[^。！？\n]*[。！？]?"
)
SECTION_LABEL = re.compile(
    r"(?<!\n)(【(?:结论|原因|建议|停止信号|结合档案|直接回答|先判断|"
    r"现在怎么做|风险提示|需要就医|你的情况|训练计划|计划安排)】)"
)
SECTION_LABEL_CONTENT = re.compile(
    r"(【(?:结论|原因|建议|停止信号|结合档案|直接回答|先判断|"
    r"现在怎么做|风险提示|需要就医|你的情况|训练计划|计划安排)】)[ \t]*(?=\S)"
)
NUMBERED_ITEM = re.compile(
    r"(?<=[。！？；])(?=(?:[一二三四五六七八九十]+[、.]|"
    r"\d{1,2}[.、](?!\d)))"
)


def normalize_plain_text(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"__(.*?)__", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"^#{1,6}\s*", "", value, flags=re.MULTILINE)
    value = re.sub(r"^[ \t]*[-*+][ \t]+", "• ", value, flags=re.MULTILINE)
    value = value.replace("```", "")
    value = re.sub(r"\[?资料\s*\d+\]?", "", value)
    value = INTERNAL_META_SENTENCE.sub("", value)
    value = SECTION_LABEL.sub(r"\n\1", value)
    value = SECTION_LABEL_CONTENT.sub(r"\1\n", value)
    value = NUMBERED_ITEM.sub("\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"^[ \t]*\n+", "", value)
    return value.strip()
