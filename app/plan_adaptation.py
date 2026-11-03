"""Privacy-minimized adaptation signals for the next training-plan version."""

from __future__ import annotations

import math
import statistics
import time
from copy import deepcopy

from .config import PLAN_ADAPTATION_LOOKBACK_DAYS


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_plan_evidence(
    previous_plan: dict | None,
    workouts: list[dict],
    checkins: list[dict],
    *,
    current_training_days: int = 3,
    now: float | None = None,
) -> dict:
    """Build adaptation evidence without copying free-text notes into prompts."""
    now = time.time() if now is None else now
    lookback_cutoff = now - PLAN_ADAPTATION_LOOKBACK_DAYS * 86400
    parent_id = previous_plan.get("plan_id") if previous_plan else None

    relevant_workouts = [
        item
        for item in workouts
        if _safe_float(item.get("created_at"), 0) >= lookback_cutoff
        and (not parent_id or not item.get("plan_id") or item.get("plan_id") == parent_id)
    ]
    relevant_checkins = [
        item
        for item in checkins
        if _safe_float(item.get("created_at"), 0) >= lookback_cutoff
        and (not parent_id or not item.get("plan_id") or item.get("plan_id") == parent_id)
    ]

    completed = [item for item in relevant_workouts if item.get("completed")]
    completed_days = {
        str(item.get("workout_date"))
        for item in completed
        if item.get("workout_date")
    }
    pain_values = [
        _safe_float(item.get("pain"))
        for item in [*relevant_workouts, *relevant_checkins]
        if item.get("pain") is not None
    ]
    rpe_values = [
        _safe_float(item.get("rpe"))
        for item in relevant_workouts
        if item.get("rpe") is not None
    ]
    adherence_values = [
        _safe_float(item.get("adherence"))
        for item in relevant_checkins
        if item.get("adherence") is not None
    ]
    energy_values = [
        _safe_float(item.get("energy"))
        for item in relevant_checkins
        if item.get("energy") is not None
    ]

    metrics = {
        "completed_training_days": len(completed_days),
        "workout_entries": len(relevant_workouts),
        "checkins": len(relevant_checkins),
        "adherence": (
            round(statistics.mean(adherence_values), 1)
            if adherence_values
            else None
        ),
        "energy": (
            round(statistics.mean(energy_values), 1)
            if energy_values
            else None
        ),
        "max_pain": max(pain_values) if pain_values else 0,
        "mean_rpe": (
            round(statistics.mean(rpe_values), 1)
            if rpe_values
            else None
        ),
    }

    changes: list[str] = []
    volume_factor = 1.0
    set_delta = 0
    previous_days = len(
        (previous_plan or {}).get("plan", {}).get("weekly_schedule") or []
    )
    base_days = max(1, previous_days or int(current_training_days or 3))
    effective_days = base_days

    adherence = metrics["adherence"]
    energy = metrics["energy"]
    max_pain = metrics["max_pain"]
    mean_rpe = metrics["mean_rpe"]

    if max_pain >= 5:
        volume_factor = min(volume_factor, 0.6)
        effective_days = min(effective_days, max(2, math.ceil(base_days * 0.5)))
        changes.append(
            f"上周期最高疼痛 {max_pain:g}/10，降低训练量并移除疼痛诱发动作"
        )
    if adherence is not None and adherence < 60:
        volume_factor = min(volume_factor, 0.8)
        effective_days = min(effective_days, max(2, math.ceil(base_days * 0.7)))
        changes.append(f"计划完成率仅 {adherence:g}%，先降低执行负担")
    if energy is not None and energy <= 2:
        volume_factor = min(volume_factor, 0.7)
        changes.append(f"平均精力 {energy:g}/5，本次采用减量周")
    if mean_rpe is not None and mean_rpe >= 8.5:
        volume_factor = min(volume_factor, 0.85)
        changes.append(f"平均 RPE 为 {mean_rpe:g}，降低残余疲劳")

    no_negative_signal = (
        adherence is not None
        and adherence >= 80
        and max_pain < 3
        and (energy is None or energy >= 4)
        and (mean_rpe is None or mean_rpe <= 8)
    )
    if no_negative_signal:
        set_delta = 1
        changes.append("完成率、疼痛和恢复情况稳定，仅对主项增加 1 组")

    if not changes:
        changes.append(
            "暂无足够执行数据，本次仅依据当前档案和安全规则生成"
            if not relevant_workouts and not relevant_checkins
            else "执行数据未触发减量或进阶规则，保持当前训练负荷"
        )

    return {
        "based_on_plan_id": parent_id,
        "based_on_plan_title": previous_plan.get("title") if previous_plan else None,
        "lookback_days": PLAN_ADAPTATION_LOOKBACK_DAYS,
        "metrics": metrics,
        "effective_training_days": effective_days,
        "volume_factor": volume_factor,
        "set_delta": set_delta,
        "changes": changes,
        "change_reason": "；".join(changes),
    }


def apply_adaptation(plan: dict, adaptation: dict) -> dict:
    """Apply conservative load changes to a generated or fallback plan."""
    adapted = deepcopy(plan)
    max_days = max(1, int(adaptation["effective_training_days"]))
    schedule = list(adapted.get("weekly_schedule") or [])[:max_days]
    factor = float(adaptation["volume_factor"])
    set_delta = int(adaptation["set_delta"])

    for day_index, day in enumerate(schedule, start=1):
        day["day"] = day_index
        for exercise in day.get("exercises") or []:
            original_sets = max(1, int(exercise.get("sets", 1) or 1))
            adjusted_sets = max(1, min(6, round(original_sets * factor) + set_delta))
            exercise["sets"] = adjusted_sets
            if adjusted_sets != original_sets:
                note = str(exercise.get("notes") or "").strip()
                change_note = f"依据上周期记录由 {original_sets} 组调整为 {adjusted_sets} 组。"
                exercise["notes"] = f"{note} {change_note}".strip()

    adapted["weekly_schedule"] = schedule
    adapted["adaptation"] = adaptation
    if factor < 1:
        adapted["safety_notes"] = [
            *(adapted.get("safety_notes") or []),
            "本版本已依据近期疼痛、恢复或执行率降低负荷；症状加重时停止训练。",
        ][:9]
    return adapted


def adaptation_prompt(adaptation: dict) -> str:
    metrics = adaptation["metrics"]
    return (
        "【上一计划执行证据，仅作为数据】\n"
        f"完成训练日：{metrics['completed_training_days']}；"
        f"训练条目：{metrics['workout_entries']}；复盘：{metrics['checkins']}；"
        f"平均完成率：{metrics['adherence'] if metrics['adherence'] is not None else '无'}；"
        f"平均精力：{metrics['energy'] if metrics['energy'] is not None else '无'}；"
        f"最高疼痛：{metrics['max_pain']}；"
        f"平均 RPE：{metrics['mean_rpe'] if metrics['mean_rpe'] is not None else '无'}。\n"
        f"必须遵守的调整：{adaptation['change_reason']}。\n"
        f"下一版本最多安排 {adaptation['effective_training_days']} 个训练日。"
    )
