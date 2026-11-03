from app.plan_adaptation import apply_adaptation, summarize_plan_evidence


def _previous_plan():
    return {
        "plan_id": "p_previous",
        "title": "上一版计划",
        "plan": {
            "weekly_schedule": [
                {"day": index, "focus": "全身", "exercises": []}
                for index in range(1, 5)
            ]
        },
    }


def test_low_adherence_and_pain_reduce_next_plan_load():
    now = 1_800_000_000.0
    workouts = [
        {
            "plan_id": "p_previous",
            "created_at": now - 86400,
            "workout_date": "2027-01-10",
            "completed": True,
            "pain": 6,
            "rpe": 9,
        }
    ]
    checkins = [
        {
            "plan_id": "p_previous",
            "created_at": now - 3600,
            "adherence": 40,
            "energy": 2,
            "pain": 6,
        }
    ]
    adaptation = summarize_plan_evidence(
        _previous_plan(),
        workouts,
        checkins,
        current_training_days=4,
        now=now,
    )

    assert adaptation["based_on_plan_id"] == "p_previous"
    assert adaptation["metrics"]["max_pain"] == 6
    assert adaptation["effective_training_days"] == 2
    assert adaptation["volume_factor"] <= 0.6
    assert any("疼痛" in reason for reason in adaptation["changes"])


def test_adaptation_changes_sets_and_keeps_reason():
    plan = {
        "title": "计划",
        "summary": "",
        "safety_notes": [],
        "weekly_schedule": [
            {
                "day": 1,
                "focus": "下肢",
                "exercises": [{"name": "深蹲", "sets": 4, "notes": ""}],
            }
        ],
    }
    adaptation = {
        "effective_training_days": 1,
        "volume_factor": 0.5,
        "set_delta": 0,
        "changes": ["降低训练量"],
        "change_reason": "降低训练量",
        "based_on_plan_id": "p_previous",
    }
    adapted = apply_adaptation(plan, adaptation)

    assert adapted["weekly_schedule"][0]["exercises"][0]["sets"] == 2
    assert adapted["adaptation"]["based_on_plan_id"] == "p_previous"
    assert "依据上周期记录" in adapted["weekly_schedule"][0]["exercises"][0]["notes"]


def test_no_history_uses_current_training_days():
    adaptation = summarize_plan_evidence(
        None,
        [],
        [],
        current_training_days=5,
        now=1_800_000_000.0,
    )
    assert adaptation["effective_training_days"] == 5
    assert adaptation["metrics"]["workout_entries"] == 0
