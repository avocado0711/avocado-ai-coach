(function (root, factory) {
    const api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.CoachAnswerRenderer = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
    const SECTION_ALIASES = {
        conclusion: ["结论", "直接回答", "先判断", "一句话结论"],
        reason: ["原因", "为什么会这样", "常见原因"],
        actions: ["建议", "现在怎么做", "你可以这样做", "可执行建议", "训练建议"],
        warning: ["停止信号", "停止训练", "风险提示", "需要就医", "就医边界"],
        plan: ["训练计划", "计划安排"],
        profile: ["结合档案", "你的情况", "档案关联"],
    };
    const labelToType = Object.fromEntries(
        Object.entries(SECTION_ALIASES).flatMap(([type, labels]) =>
            labels.map(label => [label, type]),
        ),
    );

    function normalizeModelText(text) {
        return String(text || "")
            .replace(/\*\*([\s\S]*?)\*\*/g, "$1")
            .replace(/__([\s\S]*?)__/g, "$1")
            .replace(/^#{1,6}\s*/gm, "")
            .replace(/^[ \t]*[-*+][ \t]+/gm, "• ")
            .replace(/```/g, "")
            .replace(/\[?资料\s*\d+\]?/g, "")
            .replace(/[^。！？\n]*(?:知识库|检索到|命中(?:了)?[^。！？\n]{0,12}(?:资料|片段)|参考(?:了)?[^。！？\n]{0,12}(?:资料|片段))[^。！？\n]*[。！？]?/g, "")
            .replace(/\n{3,}/g, "\n\n")
            .replace(/^[ \t]*\n+/, "")
            .trim();
    }

    function parseAnswerSections(text) {
        const normalized = normalizeModelText(text);
        const sections = [];
        let current = null;
        for (const rawLine of normalized.split("\n")) {
            const line = rawLine.trim();
            const heading = line.match(/^【([^】]+)】\s*(.*)$/);
            if (heading && labelToType[heading[1]]) {
                current = { type: labelToType[heading[1]], lines: [] };
                sections.push(current);
                if (heading[2]) current.lines.push(heading[2]);
                continue;
            }
            if (current) current.lines.push(line);
            else if (line) sections.push({ type: "plain", lines: [line] });
        }
        return sections
            .map(section => ({
                ...section,
                text: section.lines.join("\n").trim(),
            }))
            .filter(section => section.text);
    }

    function splitActionItems(text) {
        const source = String(text || "")
            .trim()
            .replace(
                /(?<=[。！？；])\s*(?=(?:[①-⑳]|[一二三四五六七八九十]+[、.]|\d{1,2}[.、]))/g,
                "\n",
            );
        const lines = source.split("\n").map(line => line.trim()).filter(Boolean);
        const hasExplicitMarkers = lines.some(line =>
            /^(?:[①-⑳]|[一二三四五六七八九十]+[、.]|\d{1,2}[.、]|[-*+•·])/.test(line),
        );
        if (!hasExplicitMarkers) {
            return source
                .split(/(?<=[。！？；])\s*/)
                .map(item => item.trim())
                .filter(Boolean)
                .slice(0, 4);
        }
        const items = [];
        for (const line of lines) {
            const cleaned = line.replace(
                /^(?:[①-⑳]|(?:[一二三四五六七八九十]+[、.])|(?:\d{1,2}[.、]))\s*/,
                "",
            );
            if (items.length && /^[•·]/.test(line)) {
                items.push(cleaned.replace(/^[•·]\s*/, ""));
            } else if (/^(?:[①-⑳]|[一二三四五六七八九十]+[、.]|\d{1,2}[.、]|[-*+])/.test(line)) {
                items.push(cleaned.replace(/^[-*+]\s*/, ""));
            } else if (items.length) {
                items[items.length - 1] += ` ${cleaned}`;
            } else {
                items.push(cleaned);
            }
        }
        return items.filter(Boolean).slice(0, 4);
    }

    function parseProfileTags(text) {
        return String(text || "")
            .split(/[\n、，,；;·|]+/)
            .map(item => item.trim())
            .filter(Boolean)
            .slice(0, 6);
    }

    function parsePipeTable(text) {
        const normalizedTable = String(text || "").replace(/\|\|+/g, "\n");
        const rows = normalizedTable
            .split("\n")
            .map(line => line.trim())
            .filter(line => line.includes("|"))
            .map(line => line.replace(/^\||\|$/g, "").split("|").map(cell => cell.trim()))
            .filter(row => row.length >= 2)
            .filter(row => !row.every(cell => /^:?-{3,}:?$/.test(cell)));
        return rows.length >= 2 ? rows.slice(0, 15) : [];
    }

    function statusFor(conclusion) {
        const text = String(conclusion || "");
        if (/(立即停止|立即就医|急救|急诊|红旗|不能继续训练|拨打急救)/.test(text)) {
            return { type: "danger", icon: "!", label: "先停止训练" };
        }
        if (/(谨慎|评估|避免|降低|疼痛|受伤|观察|调整|不适)/.test(text)) {
            return { type: "caution", icon: "!", label: "需要谨慎调整" };
        }
        return { type: "safe", icon: "✓", label: "可以参考执行" };
    }

    function appendTextBlock(container, className, text) {
        const block = document.createElement("div");
        block.className = className;
        block.textContent = text;
        container.appendChild(block);
        return block;
    }

    function renderPlainFallback(container, text) {
        const wrapper = document.createElement("div");
        wrapper.className = "answer-structured answer-fallback";
        const normalized = normalizeModelText(text);
        const status = statusFor(normalized);
        if (status.type === "danger") {
            const summary = document.createElement("div");
            summary.className = "answer-summary answer-summary-danger";
            const icon = document.createElement("span");
            icon.className = "answer-status-icon";
            icon.textContent = status.icon;
            const body = document.createElement("div");
            const label = document.createElement("span");
            label.className = "answer-status-label";
            label.textContent = status.label;
            const content = document.createElement("div");
            content.className = "answer-summary-text";
            content.textContent = normalized;
            body.append(label, content);
            summary.append(icon, body);
            wrapper.appendChild(summary);
            container.appendChild(wrapper);
            return;
        }
        const paragraphs = normalized
            .split(/\n{2,}/)
            .map(item => item.trim())
            .filter(Boolean);
        for (const paragraph of paragraphs.length ? paragraphs : [normalized]) {
            appendTextBlock(wrapper, "answer-text", paragraph);
        }
        container.appendChild(wrapper);
    }

    function renderStructuredAnswer(container, text) {
        if (!container) return;
        container.replaceChildren();
        const sections = parseAnswerSections(text);
        if (!sections.some(section => section.type !== "plain")) {
            renderPlainFallback(container, normalizeModelText(text));
            return;
        }

        const wrapper = document.createElement("div");
        wrapper.className = "answer-structured";
        const conclusion = sections.find(section => section.type === "conclusion");
        if (conclusion) {
            const status = statusFor(conclusion.text);
            const summary = document.createElement("div");
            summary.className = `answer-summary answer-summary-${status.type}`;
            const icon = document.createElement("span");
            icon.className = "answer-status-icon";
            icon.textContent = status.icon;
            const body = document.createElement("div");
            const label = document.createElement("span");
            label.className = "answer-status-label";
            label.textContent = status.label;
            const content = document.createElement("div");
            content.className = "answer-summary-text";
            content.textContent = conclusion.text;
            body.append(label, content);
            summary.append(icon, body);
            wrapper.appendChild(summary);
        }

        for (const section of sections) {
            if (section.type === "conclusion") continue;
            if (section.type === "plain") {
                appendTextBlock(wrapper, "answer-text answer-intro", section.text);
            } else if (section.type === "reason") {
                const details = document.createElement("details");
                details.className = "answer-reason";
                const summary = document.createElement("summary");
                summary.textContent = "展开原因";
                details.appendChild(summary);
                appendTextBlock(details, "answer-text", section.text);
                wrapper.appendChild(details);
            } else if (section.type === "actions") {
                const block = document.createElement("section");
                block.className = "answer-section answer-actions";
                const heading = document.createElement("h4");
                heading.textContent = "建议这么做";
                block.appendChild(heading);
                const list = document.createElement("ol");
                for (const item of splitActionItems(section.text)) {
                    const row = document.createElement("li");
                    const marker = document.createElement("span");
                    marker.className = "answer-step-marker";
                    marker.textContent = String(list.children.length + 1);
                    const copy = document.createElement("span");
                    copy.className = "answer-step-text";
                    copy.textContent = item;
                    row.append(marker, copy);
                    list.appendChild(row);
                }
                block.appendChild(list);
                wrapper.appendChild(block);
            } else if (section.type === "warning") {
                const warning = document.createElement("section");
                warning.className = "answer-warning";
                const heading = document.createElement("h4");
                heading.textContent = "出现这些情况就停止";
                warning.appendChild(heading);
                appendTextBlock(warning, "answer-warning-text", section.text);
                wrapper.appendChild(warning);
            } else if (section.type === "plan") {
                const block = document.createElement("section");
                block.className = "answer-section answer-plan";
                const heading = document.createElement("h4");
                heading.textContent = "训练安排";
                block.appendChild(heading);
                const rows = parsePipeTable(section.text);
                if (rows.length) {
                    const scroller = document.createElement("div");
                    scroller.className = "answer-plan-table-wrap";
                    const table = document.createElement("table");
                    table.className = "answer-plan-table";
                    const head = document.createElement("thead");
                    const headRow = document.createElement("tr");
                    for (const cellText of rows[0]) {
                        const cell = document.createElement("th");
                        cell.textContent = cellText;
                        headRow.appendChild(cell);
                    }
                    head.appendChild(headRow);
                    table.appendChild(head);
                    const body = document.createElement("tbody");
                    for (const row of rows.slice(1)) {
                        const rowEl = document.createElement("tr");
                        for (let index = 0; index < rows[0].length; index += 1) {
                            const cell = document.createElement("td");
                            cell.textContent = row[index] || "";
                            rowEl.appendChild(cell);
                        }
                        body.appendChild(rowEl);
                    }
                    table.appendChild(body);
                    scroller.appendChild(table);
                    block.appendChild(scroller);
                } else {
                    appendTextBlock(block, "answer-text", section.text);
                }
                wrapper.appendChild(block);
            } else if (section.type === "profile") {
                // Old answers may contain this section; hide it for a cleaner UI.
                continue;
            }
        }
        container.appendChild(wrapper);
    }

    return {
        normalizeModelText,
        parseAnswerSections,
        splitActionItems,
        parseProfileTags,
        parsePipeTable,
        renderStructuredAnswer,
    };
});
