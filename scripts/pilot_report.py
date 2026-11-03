"""Print privacy-minimized pilot metrics without exposing user text."""

from __future__ import annotations

import json
import sqlite3

from _bootstrap import PROJECT_ROOT

from app.config import FEEDBACK_DB_PATH, USERS_DB_PATH


def has_table(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None


def main() -> None:
    summary: dict = {}
    if USERS_DB_PATH.exists():
        with sqlite3.connect(USERS_DB_PATH) as users:
            users.row_factory = sqlite3.Row
            summary["users"] = users.execute(
                "SELECT COUNT(*) AS n FROM users"
            ).fetchone()["n"]
            summary["profiles_completed"] = users.execute(
                "SELECT COUNT(*) AS n FROM user_profile "
                "WHERE height > 0 AND weight > 0"
            ).fetchone()["n"]
            summary["users_with_messages"] = users.execute(
                "SELECT COUNT(DISTINCT user_id) AS n FROM chat_message"
            ).fetchone()["n"]
            summary["users_active_on_2_days"] = users.execute(
                "SELECT COUNT(*) AS n FROM ("
                "SELECT user_id FROM chat_message "
                "GROUP BY user_id HAVING COUNT(DISTINCT date(created_at,'unixepoch')) >= 2"
                ")"
            ).fetchone()["n"]
            if has_table(users, "training_plan"):
                summary["training_plans"] = users.execute(
                    "SELECT COUNT(*) AS n FROM training_plan"
                ).fetchone()["n"]
            if has_table(users, "workout_log"):
                summary["workout_exercises_logged"] = users.execute(
                    "SELECT COUNT(*) AS n FROM workout_log WHERE completed=1"
                ).fetchone()["n"]
                summary["completed_training_days"] = users.execute(
                    "SELECT COUNT(DISTINCT user_id || ':' || workout_date) AS n "
                    "FROM workout_log WHERE completed=1"
                ).fetchone()["n"]

    if FEEDBACK_DB_PATH.exists():
        with sqlite3.connect(FEEDBACK_DB_PATH) as feedback:
            feedback.row_factory = sqlite3.Row
            counts = feedback.execute(
                "SELECT rating,COUNT(*) AS n FROM feedback GROUP BY rating"
            ).fetchall()
            summary["feedback"] = {
                "likes": sum(row["n"] for row in counts if row["rating"] == 1),
                "dislikes": sum(row["n"] for row in counts if row["rating"] == -1),
            }
            total = summary["feedback"]["likes"] + summary["feedback"]["dislikes"]
            summary["feedback"]["like_rate"] = (
                summary["feedback"]["likes"] / total if total else None
            )
            summary["feedback"]["reasons"] = {
                row["reason"] or "未填写": row["n"]
                for row in feedback.execute(
                    "SELECT reason,COUNT(*) AS n FROM feedback WHERE rating=-1 "
                    "GROUP BY reason ORDER BY n DESC"
                )
            }
            feedback_columns = {
                row["name"]
                for row in feedback.execute("PRAGMA table_info(feedback)")
            }
            summary["feedback"]["pending_review"] = (
                feedback.execute(
                    "SELECT COUNT(*) AS n FROM feedback "
                    "WHERE review_status='pending'"
                ).fetchone()["n"]
                if "review_status" in feedback_columns
                else None
            )

    summary["note"] = (
        "小样本只描述现象；不要用点赞率替代安全性、任务完成率或用户留存。"
    )
    output = PROJECT_ROOT / "reports" / "pilot_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"报告已写入: {output}")


if __name__ == "__main__":
    main()
