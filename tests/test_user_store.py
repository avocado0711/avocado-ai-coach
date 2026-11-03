from uuid import uuid4

from app import user_store


def test_account_profile_plan_and_deletion_flow():
    username = f"user_{uuid4().hex[:10]}"
    password = "valid-password-123"
    registered = user_store.register_user(username, password)
    assert registered["ok"]
    user_id = registered["user_id"]

    assert user_store.verify_user(username, password)["ok"]
    profile = user_store.upsert_profile(
        user_id,
        {
            "height": 175,
            "weight": 70,
            "age": 24,
            "gender": "男",
            "train_year": 2,
            "injury": "膝关节痛、腰肌劳损",
            "goal": "增肌",
            "training_days": 4,
            "session_minutes": 60,
        },
    )
    assert profile["exists"]
    assert profile["training_days"] == 4
    assert profile["injury"] == "膝关节痛、腰肌劳损"

    plan = user_store.create_plan(
        user_id,
        "测试计划",
        "增肌",
        4,
        {"weekly_schedule": []},
    )
    assert plan["plan_id"]
    assert plan["version"] == 1
    assert user_store.get_plan(user_id)["plan"]["weekly_schedule"] == []

    replacement = user_store.create_plan(
        user_id,
        "调整计划",
        "增肌",
        4,
        {"weekly_schedule": [], "adaptation": {"changes": ["降低负荷"]}},
        parent_plan_id=plan["plan_id"],
        change_reason="降低负荷",
    )
    assert replacement["version"] == 2
    assert replacement["parent_plan_id"] == plan["plan_id"]
    assert replacement["change_reason"] == "降低负荷"

    message_count_before = len(user_store.get_messages(user_id))
    user_store.add_message(user_id, "user", "测试消息")
    assert len(user_store.get_messages(user_id)) == message_count_before + 1

    exported = user_store.export_user_data(user_id)
    assert exported["account"]["username"] == username
    assert [item["title"] for item in exported["plans"]] == [
        "测试计划",
        "调整计划",
    ]

    user_store.delete_user(user_id)
    assert not user_store.user_exists(user_id)


def test_legacy_password_hash_still_verifies():
    # Existing installations used 120,000 PBKDF2 iterations without metadata.
    import hashlib
    import os
    import time

    username = f"legacy_{uuid4().hex[:8]}"
    password = "legacy-password"
    salt = os.urandom(16).hex()
    legacy_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        bytes.fromhex(salt),
        120_000,
    ).hex()
    user_id = "u_legacy_" + uuid4().hex[:8]
    with user_store._lock:
        user_store._conn.execute(
            "INSERT INTO users(user_id,username,pwd_hash,pwd_salt,created_at) "
            "VALUES (?,?,?,?,?)",
            (user_id, username, legacy_hash, salt, time.time()),
        )
        user_store._conn.commit()
    assert user_store.verify_user(username, password)["ok"]
    user_store.delete_user(user_id)

