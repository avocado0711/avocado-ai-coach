from app.answer_policy import select_answer_mode


def test_simple_information_question_uses_plain_answer():
    assert select_answer_mode("深蹲120啥水平") == "simple"
    assert select_answer_mode("深蹲主要练哪些肌肉") == "simple"
    assert select_answer_mode("高翻和卧推有什么区别") == "simple"


def test_risk_and_plan_questions_use_structured_answer():
    assert select_answer_mode("深蹲膝盖响是为啥") == "structured"
    assert select_answer_mode("膝盖痛还能练腿吗") == "structured"
    assert select_answer_mode("每周练3天帮我排全身训练计划") == "structured"
