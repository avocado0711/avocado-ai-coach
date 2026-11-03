from app.text_output import normalize_plain_text


def test_model_markdown_is_removed_for_plain_text_chat():
    raw = "## 结论\n**动作不同**，不能直接比较。\n\n- 看动作标准\n* 看相对体重"
    assert normalize_plain_text(raw) == (
        "结论\n动作不同，不能直接比较。\n\n• 看动作标准\n• 看相对体重"
    )


def test_normal_text_newlines_are_preserved():
    raw = "第一段\n\n第二段"
    assert normalize_plain_text(raw) == raw


def test_internal_retrieval_terms_are_hidden_from_user():
    raw = (
        "知识库未命中，以下只给通用原则。\n"
        "高翻需要先练髋伸发力。\n"
        "这部分参考了资料2和知识片段，但可直接执行。"
    )
    assert normalize_plain_text(raw) == "高翻需要先练髋伸发力。"


def test_structured_sections_and_numbered_items_get_line_breaks():
    raw = "【结论】可以先观察。【建议】1. 降低重量。2. 放慢下蹲。"
    assert normalize_plain_text(raw) == (
        "【结论】\n可以先观察。\n【建议】\n1. 降低重量。\n2. 放慢下蹲。"
    )
