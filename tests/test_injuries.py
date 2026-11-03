from app.injuries import has_multiple_injuries, normalize_injury_text, parse_injuries


def test_multiple_injuries_are_normalized_and_deduplicated():
    raw = "膝关节痛、腰肌劳损，膝关节痛；肩袖损伤"
    assert parse_injuries(raw) == ["膝关节痛", "腰肌劳损", "肩袖损伤"]
    assert normalize_injury_text(raw) == "膝关节痛、腰肌劳损、肩袖损伤"
    assert has_multiple_injuries(raw)


def test_none_is_exclusive():
    assert parse_injuries("无、膝关节痛") == ["膝关节痛"]
    assert normalize_injury_text(["无伤病", "无"]) == "无"


def test_list_payload_is_supported():
    assert normalize_injury_text(["肩袖损伤", "脚踝扭伤"]) == "肩袖损伤、脚踝扭伤"
