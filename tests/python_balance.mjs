import fs from "node:fs";
import path from "node:path";

const root = new URL("..", import.meta.url);
const ignored = new Set([".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache"]);

function findPythonFiles(directory, relative = "") {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    if (ignored.has(entry.name)) return [];
    const childRelative = relative ? `${relative}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      return findPythonFiles(new URL(`${entry.name}/`, directory), childRelative);
    }
    return entry.name.endsWith(".py") ? [childRelative] : [];
  });
}

const files = findPythonFiles(root);
const pairs = {")": "(", "]": "[", "}": "{"};

for (const file of files) {
  const text = fs.readFileSync(new URL(file, root), "utf8");
  const stack = [];
  let line = 1;
  let inString = false;
  let quote = "";
  let escaped = false;
  let comment = false;
  let triple = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === "\n") {
      line += 1;
      comment = false;
      continue;
    }
    if (comment) continue;
    if (inString) {
      if (triple && text.slice(index, index + 3) === quote.repeat(3)) {
        inString = false;
        triple = false;
        index += 2;
        continue;
      }
      if (!triple && escaped) {
        escaped = false;
        continue;
      }
      if (!triple && character === "\\") {
        escaped = true;
        continue;
      }
      if (!triple && character === quote) inString = false;
      continue;
    }
    if (character === "#") {
      comment = true;
      continue;
    }
    if (character === '"' || character === "'") {
      quote = character;
      inString = true;
      triple = text.slice(index, index + 3) === character.repeat(3);
      if (triple) index += 2;
      continue;
    }
    if ("([{".includes(character)) {
      stack.push({ character, line });
      continue;
    }
    if (")]}".includes(character)) {
      const open = stack.pop();
      if (!open || open.character !== pairs[character]) {
        throw new Error(`${file}:${line}: unmatched ${character}`);
      }
    }
  }
  if (inString) throw new Error(`${file}:${line}: unterminated string`);
  if (stack.length) {
    const open = stack.at(-1);
    throw new Error(`${file}:${open.line}: unclosed ${open.character}`);
  }
}

console.log(`python delimiters ok (${files.length} files)`);
