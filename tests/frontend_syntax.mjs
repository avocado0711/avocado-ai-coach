import fs from "node:fs";

const html = fs.readFileSync(new URL("../static/index.html", import.meta.url), "utf8");
const answerRenderer = fs.readFileSync(
  new URL("../static/answer-renderer.js", import.meta.url),
  "utf8",
);
new Function(answerRenderer);
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(match => match[1]);
if (!scripts.length) throw new Error("No inline script found");
for (const script of scripts) new Function(script);

const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map(match => match[1]);
const duplicates = [...new Set(ids.filter((id, index) => ids.indexOf(id) !== index))];
if (duplicates.length) throw new Error(`Duplicate ids: ${duplicates.join(", ")}`);

const references = [
  ...html.matchAll(/getElementById\("([^"]+)"\)/g),
].map(match => match[1]);
const missingReferences = [
  ...new Set(references.filter(id => !ids.includes(id))),
];
if (missingReferences.length) {
  throw new Error(`Missing element ids: ${missingReferences.join(", ")}`);
}

const handlers = [
  ...html.matchAll(/onclick="([A-Za-z_$][\w$]*)\(/g),
].map(match => match[1]);
const functions = [
  ...html.matchAll(/function\s+([A-Za-z_$][\w$]*)\s*\(/g),
].map(match => match[1]);
const missingHandlers = [
  ...new Set(handlers.filter(name => !functions.includes(name))),
];
if (missingHandlers.length) {
  throw new Error(`Missing handlers: ${missingHandlers.join(", ")}`);
}

const backend = fs.readFileSync(new URL("../app/backend.py", import.meta.url), "utf8");
const backendRoutes = [
  ...backend.matchAll(/@app\.(?:get|post|delete)\("([^"]+)"/g),
].map(match => match[1]);
const apiCalls = [
  ...html.matchAll(/fetch\("(\/api\/[^"]+)"/g),
].map(match => match[1]);
const missingRoutes = [
  ...new Set(apiCalls.filter(route => !backendRoutes.includes(route))),
];
if (missingRoutes.length) {
  throw new Error(`Frontend routes missing in backend: ${missingRoutes.join(", ")}`);
}

console.log(
  `frontend syntax ok (${scripts.length} inline + answer renderer, ${ids.length} ids, ` +
    `${new Set(apiCalls).size} API routes)`,
);

