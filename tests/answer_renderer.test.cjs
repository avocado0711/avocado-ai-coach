const assert = require("node:assert/strict");
const renderer = require("../static/answer-renderer.js");

const structured = `【结论】
只有弹响、不痛不肿时可以先观察。
【原因】
1. 关节内气体变化
2. 动作控制不足
【建议】
1. 先做徒手慢蹲检查
2. 改用箱式深蹲
3. 每周练 1 到 2 次
【停止信号】
出现疼痛、肿胀或卡住时停止训练。
【训练计划】
训练日 | 重点 | 动作与组次 | 休息 | 注意
第1天 | 下肢 | 箱式深蹲 3×8 | 90秒 | 无痛范围
第2天 | 上肢 | 哑铃卧推 3×10 | 90秒 | 保留余力`;

const sections = renderer.parseAnswerSections(structured);
assert.deepEqual(
    sections.map(section => section.type),
    ["conclusion", "reason", "actions", "warning", "plan"],
);
assert.equal(renderer.splitActionItems(sections[2].text).length, 3);
assert.deepEqual(
    renderer.splitActionItems("先降低重量。再放慢下蹲。最后检查膝盖方向。"),
    ["先降低重量。", "再放慢下蹲。", "最后检查膝盖方向。"],
);
assert.deepEqual(
    renderer.splitActionItems("1. 先判断动作标准。2. 降低重量。3. 检查疼痛。"),
    ["先判断动作标准。", "降低重量。", "检查疼痛。"],
);
assert.deepEqual(renderer.parsePipeTable(sections[4].text), [
    ["训练日", "重点", "动作与组次", "休息", "注意"],
    ["第1天", "下肢", "箱式深蹲 3×8", "90秒", "无痛范围"],
    ["第2天", "上肢", "哑铃卧推 3×10", "90秒", "保留余力"],
]);
assert.deepEqual(
    renderer.parsePipeTable(
        "| 训练日 | 重点 | 动作 | 休息 ||---|---|---|---|| 第1天 | 全身 | 深蹲 | 90秒 || 第2天 | 上肢 | 卧推 | 90秒 |",
    ),
    [
        ["训练日", "重点", "动作", "休息"],
        ["第1天", "全身", "深蹲", "90秒"],
        ["第2天", "上肢", "卧推", "90秒"],
    ],
);
assert.deepEqual(
    renderer.parseProfileTags("膝关节痛、颈椎不适"),
    ["膝关节痛", "颈椎不适"],
);
assert.equal(
    renderer.normalizeModelText("知识库未命中。\n**膝盖**弹响先观察。"),
    "膝盖弹响先观察。",
);

const fallback = renderer.parseAnswerSections("第一段。\n\n第二段。");
assert.equal(fallback.every(section => section.type === "plain"), true);

console.log("answer renderer parser ok");
