import { DatabaseSync } from "node:sqlite";
import fs from "node:fs";

for (const file of ["app/user_store.py", "app/backend.py"]) {
  const source = fs.readFileSync(new URL(`../${file}`, import.meta.url), "utf8");
  const scripts = [
    ...source.matchAll(/executescript\(\s*"""([\s\S]*?)"""\s*\)/g),
  ].map(match => match[1]);
  if (!scripts.length) throw new Error(`${file}: no SQL schema found`);

  const database = new DatabaseSync(":memory:");
  for (const script of scripts) database.exec(script);
  const tables = new Set(
    database
      .prepare("SELECT name FROM sqlite_master WHERE type='table'")
      .all()
      .map(row => row.name),
  );
  const required =
    file.endsWith("user_store.py")
      ? [
          "users",
          "user_profile",
          "chat_message",
          "user_consent",
          "training_plan",
          "workout_log",
          "weekly_checkin",
        ]
      : ["feedback", "pending_answer", "feedback_status"];
  const missing = required.filter(name => !tables.has(name));
  if (missing.length) {
    throw new Error(`${file}: missing tables ${missing.join(", ")}`);
  }
  console.log(`${file}: SQL schema ok (${scripts.length} script blocks)`);
  database.close();
}

