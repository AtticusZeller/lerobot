---
name: doc-integrator
description: Reconciles daily development conversations with project documentation. Use when user wants to sync docs, update documentation from chat history, integrate today's changes into docs, or keep docs up-to-date with recent work. Extracts technical decisions and API changes from conversation, compares against existing docs, and proposes targeted updates.
allowed-tools: Read, Glob, Grep, Edit, Write
---

# Project Documentation Integrator

Reconcile fast-paced daily development conversations with static project documentation. Extract valuable technical decisions, API updates, and architectural changes from the chat history, compare them against current documentation, and propose specific, actionable updates.

## Workflow

Execute the following phases **sequentially**. Do NOT proceed to Phase 3 until the user confirms which options to apply.

---

### Phase 1: Context Extraction & Document Analysis

1. **Scan the conversation** for:
   - New or changed environment variables
   - Modified API endpoints or function signatures
   - Architectural decisions or trade-offs
   - Resolved bugs that revealed structural issues
   - New dependencies, libraries, or CLI commands
   - Configuration changes (config files, flags, schemas)
   - Deployment or setup procedure changes

2. **Read existing documentation** — default search path is `docs/`. Use `Glob` with `docs/**/*.md` first, then expand to `**/*.md` if needed. `Read` only the relevant files.

3. **Identify gaps** — compare extracted insights against current docs. Flag:
   - Outdated instructions
   - Missing configuration steps
   - Undocumented new features or commands
   - Changed behavior that contradicts existing docs

---

### Phase 2: Proposal Generation (present to user, then WAIT)

Output a structured proposal grouped by priority. Use this exact format:

```
**[CRITICAL UPDATES]** *(broken build steps, changed env vars, core API changes)*

- **[Option N]** Target: `<filename>` → Section: `<Section Name>`
  - **Reason:** <one sentence — what changed today vs. what doc says>
  - **Action:** <specific edit — rewrite X, add Y, remove Z>

**[ENHANCEMENTS & ADDITIONS]** *(new utilities, improved steps, minor notes)*

- **[Option N]** Target: `<filename>` → Section: `<Section Name>`
  - **Reason:** <one sentence>
  - **Action:** <specific edit>
```

After listing all options, prompt the user:

> Please reply with the option numbers you want me to apply (e.g., "1 and 3"). I will generate the exact Markdown content for each.

**STOP HERE and wait for user input.**

---

### Phase 3: Execution

Once the user selects options:

1. For each selected option, generate the exact Markdown content.
2. Wrap output in fenced code blocks. For replacements, show enough surrounding context (2–3 lines before/after) so the user knows exactly where to paste.
3. Apply changes directly using `Edit` or `Write` if the user says "apply it" or "make the changes" — otherwise just present the blocks.
4. Match the existing documentation's formatting style (heading levels, code block language tags, list style).

---

## Constraints

- **No hallucination:** Only propose updates grounded in explicit events or code from the conversation.
- **Idempotency:** Skip anything already accurately documented.
- **Brevity in Phase 2:** Keep proposals concise — save detailed writing for Phase 3.
- **Minimal scope:** One option = one focused change. Do not bundle unrelated edits.
