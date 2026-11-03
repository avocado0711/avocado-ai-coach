"""Normalize a free-text or multi-select injury profile."""

from __future__ import annotations

import re

_SEPARATOR = re.compile(r"[,，、;；/|\\\n]+")


def parse_injuries(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return ["无"]
    raw_values = value if isinstance(value, (list, tuple)) else [str(value)]
    injuries: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for item in _SEPARATOR.split(str(raw)):
            injury = re.sub(r"\s+", " ", item).strip()
            if not injury or injury in {"无", "无伤病", "没有", "否"}:
                continue
            key = injury.casefold()
            if key not in seen:
                seen.add(key)
                injuries.append(injury)
    return injuries or ["无"]


def normalize_injury_text(value: str | list[str] | tuple[str, ...] | None) -> str:
    return "、".join(parse_injuries(value))[:200]


def has_multiple_injuries(value: str | list[str] | tuple[str, ...] | None) -> bool:
    injuries = parse_injuries(value)
    return injuries != ["无"] and len(injuries) > 1
