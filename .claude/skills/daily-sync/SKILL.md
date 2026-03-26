---
name: daily-sync
description: Closes the daily development cycle by generating a structured plan.md entry. Use when user wants to wrap up the day, log today's progress, write a daily plan entry, record what was done, or plan tomorrow's tasks. Synthesizes user's dictation with technical context from the session into a ready-to-paste Markdown block.
allowed-tools: Read, Glob, Grep
argument-hint: "[today's summary and tomorrow's plan]"
---

# Daily Planning & Rhythm Coordinator

Close the daily development cycle by synthesizing the user's dictation with technical context from the current session. Output a structured entry appended to **`docs/plan.md`** (default path).

## How to invoke

The user dictates their progress and next steps — either as the skill argument or as a follow-up message. Example:

> /daily-sync 今天完成了数据集加载的 bug 修复，明天要开始写训练脚本

Or with no argument, prompt the user:

> 请告诉我今天完成了什么，以及明天的计划是什么？

---

## Phase 1: Ingestion & Alignment

1. **Receive dictation** — extract two parts from what the user said:
   - "Today's Completed Tasks" (what was done)
   - "Next Steps / Tomorrow's Plan" (what comes next)

2. **Enrich with session context** — scan the conversation for technical specifics that match the user's dictation:
   - Exact file names or paths modified
   - Function/class names introduced or changed
   - Libraries installed or removed
   - CLI commands run, errors resolved, config values set
   - API endpoints, data schemas, or model configs touched

3. **Strict rule:** Never invent tasks. Only elaborate on items the user explicitly mentioned, using facts already present in the conversation.

---

## Phase 2: Generate the plan.md entry

Use today's date (`currentDate` if available, otherwise ask).

**Write target:** `docs/plan.md` — append the new entry to the end of the file. If the file does not exist, create it. Use `Write` or `Edit` to persist the output directly; do not just print the block unless the user asks to preview first.

Output **only** the Markdown block below — no preamble, no trailing commentary.

Fill in the template:

```markdown
## [YYYY-MM-DD] Daily Sync

### ✅ Completed Today
* **[Task name from user's dictation]**: [Supplement with technical details from session context. E.g., "修复了 `LeRobotDataset.__getitem__` 中的时序索引越界问题，根因是 `delta_timestamps` 未正确处理边界帧。"]
* **[Task name]**: [Details...]

### 🎯 Next Steps (Plan)
* **[Priority 1]**: [User's dictated plan, refined into clear actionable steps if session context allows]
* **[Priority 2]**: [...]

### 📝 Notes / Blockers
* [Only include if user mentioned unresolved issues, or the session ended on an unsolved blocker. Omit this section entirely if nothing to report.]
```

---

## Constraints

- **Language:** Write in the user's dictation language (default Chinese if they dictate in Chinese). Keep all technical terms, file names, function names, and code references in English.
- **Tone:** Professional, objective, factual. No filler phrases.
- **Fidelity:** Do not reorder or reinterpret the user's priorities — preserve their intent exactly.
- **Scope:** One entry per invocation. Do not generate multiple days.
- **Blockers section:** Omit entirely (don't leave it empty) if there are no blockers.
