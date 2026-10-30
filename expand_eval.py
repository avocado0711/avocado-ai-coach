# expand_eval.py
import json

EVAL_FILE = "D:/26572/dunk/eval_set.json1"

# 原始 20 条（gold 全部修正，根据 doc_0~doc_89 的内容重新标）
revised_original = [
    {"query": "深蹲主要练哪些肌肉", "gold_doc_ids": ["doc_0"]},
    {"query": "深蹲底部粘滞点怎么突破", "gold_doc_ids": ["doc_0", "doc_12", "doc_37", "doc_47"]},
    {"query": "深蹲粘滞点是什么原因", "gold_doc_ids": ["doc_0", "doc_12", "doc_19", "doc_47", "doc_51", "doc_52"]},
    {"query": "突破深蹲粘滞点有哪些训练手段", "gold_doc_ids": ["doc_0", "doc_12", "doc_37", "doc_47"]},
    {"query": "暂停深蹲箱式深蹲前蹲有用吗", "gold_doc_ids": ["doc_0", "doc_38"]},
    # 下面 15 条是你原来评估集里的，如果和你实际内容对不上，你自己核对改
    {"query": "硬拉腰疼怎么办", "gold_doc_ids": ["doc_1", "doc_28", "doc_37", "doc_77", "doc_78"]},
    {"query": "卧推肩膀疼怎么调整", "gold_doc_ids": ["doc_3", "doc_8", "doc_29", "doc_78", "doc_79"]},
    {"query": "增肌期每天吃多少蛋白质", "gold_doc_ids": ["doc_3", "doc_9", "doc_43", "doc_45"]},
    {"query": "引体向上做不了几个怎么练", "gold_doc_ids": ["doc_4", "doc_36"]},
    {"query": "核心训练对深蹲有帮助吗", "gold_doc_ids": ["doc_2", "doc_40", "doc_77"]},
    {"query": "训练后多久吃东西比较好", "gold_doc_ids": ["doc_3", "doc_43", "doc_45"]},
    {"query": "腰椎间盘突出能练硬拉吗", "gold_doc_ids": ["doc_28", "doc_77", "doc_78"]},
    {"query": "肩袖损伤怎么恢复", "gold_doc_ids": ["doc_8", "doc_79"]},
    {"query": "弹跳训练一周几次", "gold_doc_ids": ["doc_84", "doc_85"]},
    {"query": "深蹲能提升弹跳吗", "gold_doc_ids": ["doc_50", "doc_84"]},
    {"query": "圆肩驼背怎么练", "gold_doc_ids": ["doc_29", "doc_87"]},
    {"query": "骨盆前倾怎么纠正", "gold_doc_ids": ["doc_51", "doc_87"]},
    {"query": "新手应该怎么安排训练", "gold_doc_ids": ["doc_68", "doc_69", "doc_88"]},
    {"query": "训练组间休息多久", "gold_doc_ids": ["doc_6", "doc_71"]},
    {"query": "如何突破训练平台期", "gold_doc_ids": ["doc_14", "doc_72", "doc_73"]},
]

# 新增 30 条
new_items = [
    {"query": "深蹲膝盖内扣怎么纠正", "gold_doc_ids": ["doc_7", "doc_19", "doc_75", "doc_76"]},
    {"query": "深蹲和硬拉的区别", "gold_doc_ids": ["doc_0", "doc_1"]},
    {"query": "硬拉标准启动姿势", "gold_doc_ids": ["doc_1", "doc_37"]},
    {"query": "罗马尼亚硬拉练哪里", "gold_doc_ids": ["doc_4", "doc_5"]},
    {"query": "卧推握距多宽合适", "gold_doc_ids": ["doc_3", "doc_35"]},
    {"query": "引体向上练背还是练手臂", "gold_doc_ids": ["doc_4", "doc_36"]},
    {"query": "平板支撑能练腹肌吗", "gold_doc_ids": ["doc_2", "doc_40"]},
    {"query": "分腿蹲适合什么人练", "gold_doc_ids": ["doc_10", "doc_38"]},
    {"query": "训练前热身怎么做", "gold_doc_ids": ["doc_6", "doc_31"]},
    {"query": "延迟性肌肉酸痛怎么缓解", "gold_doc_ids": ["doc_7", "doc_18"]},
    {"query": "髌腱炎怎么恢复训练", "gold_doc_ids": ["doc_27", "doc_46", "doc_76"]},
    {"query": "肩峰撞击如何调整卧推", "gold_doc_ids": ["doc_8", "doc_29", "doc_79"]},
    {"query": "增肌期碳水怎么吃", "gold_doc_ids": ["doc_9", "doc_43", "doc_45"]},
    {"query": "肌酸怎么吃", "gold_doc_ids": ["doc_14", "doc_44"]},
    {"query": "咖啡因对训练有帮助吗", "gold_doc_ids": ["doc_14", "doc_44"]},
    {"query": "高翻怎么练", "gold_doc_ids": ["doc_0", "doc_22", "doc_48", "doc_49"]},
    {"query": "抓举和高翻的区别", "gold_doc_ids": ["doc_49"]},
    {"query": "如何选择训练鞋", "gold_doc_ids": ["doc_24"]},
    {"query": "负重腰带什么时候用", "gold_doc_ids": ["doc_24", "doc_55"]},
    {"query": "推拉腿分化适合新手吗", "gold_doc_ids": ["doc_16", "doc_41", "doc_68"]},
    {"query": "大体重男性怎么开始健身", "gold_doc_ids": ["doc_61", "doc_62", "doc_63"]},
    {"query": "小体重增肌训练怎么安排", "gold_doc_ids": ["doc_56", "doc_57", "doc_58"]},
    {"query": "女性减脂怎么练", "gold_doc_ids": ["doc_65", "doc_66", "doc_67", "doc_83"]},
    {"query": "减脂期如何保住肌肉", "gold_doc_ids": ["doc_67", "doc_83", "doc_84"]},
    {"query": "弹跳专项训练动作有哪些", "gold_doc_ids": ["doc_84", "doc_85"]},
    {"query": "箱跳和跳深的区别", "gold_doc_ids": ["doc_16", "doc_38", "doc_85"]},
    {"query": "高低肩怎么矫正", "gold_doc_ids": ["doc_87"]},
    {"query": "深蹲踝关节背屈受限怎么改善", "gold_doc_ids": ["doc_12", "doc_52"]},
    {"query": "训练时腰酸是为什么", "gold_doc_ids": ["doc_1", "doc_28", "doc_37", "doc_77"]},
    {"query": "新手多久能练出效果", "gold_doc_ids": ["doc_68", "doc_69"]},
]

all_items = revised_original + new_items

# 去重（按 query）
seen = set()
final = []
for it in all_items:
    if it["query"] in seen:
        continue
    seen.add(it["query"])
    final.append(it)

with open(EVAL_FILE, "w", encoding="utf-8") as f:
    for it in final:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")

print(f"✅ 已写入 {len(final)} 条评估集（修正原 20 条 + 新增 30 条）")