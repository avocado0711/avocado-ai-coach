import json
# 删掉所有requests、TAVILY依赖，彻底断网
from config import client, LLM_MODEL

# FC工具描述：彻底删除 web_search，只保留本地工具，杜绝所有联网报错
tools = [
    {
        "type": "function",
        "function": {
            "name": "problem_parse",
            "description": "问题解读工具：解析用户原始问题，提取核心诉求、关键词、问题类型，输出结构化结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_question": {"type": "string", "description": "用户原始问题"}
                },
                "required": ["user_question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_rewrite",
            "description": "上下文Query改写：结合对话历史，把省略、指代句子改成完整独立查询；无依赖则原样返回",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_question": {"type": "string"},
                    "chat_history": {"type": "string", "description": "历史对话文本"}
                },
                "required": ["user_question", "chat_history"]
            }
        }
    }
]


def problem_parse(user_question: str):
    prompt = f"""
分析用户问题，输出JSON，不要额外文字。
字段：core_demand, keywords(数组), question_type, is_ambiguous(bool), remark
用户问题：{user_question}
"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(resp.choices[0].message.content.strip())


def query_rewrite(user_question: str, chat_history: str):
    prompt = f"""
结合对话历史改写问题，输出JSON：{{"rewritten_query":"","depend_history":true/false}}
只输出JSON，不要解释。
对话历史：{chat_history}
当前问题：{user_question}
"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(resp.choices[0].message.content.strip())
