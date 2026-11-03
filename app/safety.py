"""Deterministic safety checks that run before any LLM generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

URGENT_PATTERNS = (
    r"胸痛|胸口痛|胸闷.*呼吸困难|呼吸困难",
    r"晕厥|昏倒|失去意识|意识不清",
    r"大小便失禁|会阴.*麻木|鞍区.*麻木",
    r"肢体.*无力|突然.*麻木|说话不清",
    r"骨折|骨头.*露出|关节.*变形",
    r"剧烈疼痛|疼得.*无法.*(走|站|动)",
    r"头部.*受伤|撞到.*头|脑震荡",
    r"严重.*肿胀|快速.*肿胀",
    r"发热.*关节.*红|关节.*红.*发热",
)

DIAGNOSIS_PATTERNS = (
    (
        r"帮我诊断|我是不是.*病|确诊|吃什么.*药|"
        r"(?:需要|要不要|用不用|是否).*手术|开药"
    ),
    (
        r"(?:是不是|是否|有没有|可能).{0,8}"
        r"(?:椎间盘.*突出|半月板.*撕裂|骨折)"
    ),
)

UNSAFE_PATTERNS = (
    r"忽略.*(之前|以上).*(要求|提示)|输出.*系统提示词|泄露.*提示词",
    r"一(天|次).*(一百|100).*组|每天.*力竭",
    r"受伤.*(个人纪录|最大重量)|疼痛.*(硬撑|继续冲)",
    r"不热身.*(最大重量|极限重量)",
)

FITNESS_TERMS = (
    "训练", "健身", "运动", "增肌", "减脂", "力量", "有氧", "无氧",
    "深蹲", "硬拉", "卧推", "引体", "推举", "跑步", "弹跳", "体态",
    "肌肉", "核心", "热身", "拉伸", "康复", "疼", "痛", "伤病", "膝盖",
    "肩", "腰", "腕", "肘", "踝", "髋", "饮食", "蛋白", "碳水", "热量",
    "组数", "次数", "休息", "计划", "动作", "练", "肌酸", "平板支撑",
    "分腿蹲", "高翻", "抓举", "跳箱", "跳深", "分化",
)

GREETING_PATTERN = re.compile(
    r"^\s*(你好|您好|hi|hello|嗨|哈喽|谢谢|再见|早上好|晚上好)[！!。.?\s]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SafetyAssessment:
    action: str  # allow | urgent_care | medical_boundary | out_of_scope
    message: str = ""
    reason: str = ""


def assess_query(query: str) -> SafetyAssessment:
    """Return a deterministic safety decision for a user query."""
    text = (query or "").strip()
    if not text:
        return SafetyAssessment("out_of_scope", "请输入一个与训练或运动健康相关的问题。")

    if any(re.search(pattern, text, re.IGNORECASE) for pattern in URGENT_PATTERNS):
        return SafetyAssessment(
            "urgent_care",
            "你描述的情况可能需要现场医疗评估。请立即停止训练，不要继续尝试动作，"
            "联系医生或当地急救服务；若出现胸痛、呼吸困难、意识异常、肢体无力或大小便异常，"
            "请优先就医。AI 教练不能替代急诊或医生诊断。",
            "urgent_symptom",
        )

    if any(re.search(pattern, text, re.IGNORECASE) for pattern in DIAGNOSIS_PATTERNS):
        return SafetyAssessment(
            "medical_boundary",
            "我不能根据文字为你做疾病诊断、开药或决定是否手术。建议由医生或注册康复师完成评估；"
            "我可以帮你整理需要向医生说明的症状，并提供在专业人员确认后的低风险训练原则。",
            "medical_diagnosis_request",
        )

    if any(re.search(pattern, text, re.IGNORECASE) for pattern in UNSAFE_PATTERNS):
        return SafetyAssessment(
            "unsafe_request",
            "这个请求包含明显超出安全范围或绕过系统边界的指令，我不能照做。"
            "训练应在无痛、动作可控和充分热身的前提下逐步增加负荷；有伤病时先由专业人员评估。",
            "unsafe_request",
        )

    if GREETING_PATTERN.match(text):
        return SafetyAssessment(
            "allow",
            "你好，我是你的训练助手。你可以告诉我训练目标、每周可训练天数、器械条件和伤病限制。",
            "greeting",
        )

    if not any(term in text for term in FITNESS_TERMS):
        return SafetyAssessment(
            "out_of_scope",
            "我只处理训练计划、动作技术、运动营养和训练安全相关问题。"
            "请把问题改写成与训练有关的内容。",
            "out_of_scope",
        )

    return SafetyAssessment("allow")

