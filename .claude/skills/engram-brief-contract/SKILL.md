---
name: engram-brief-contract
description: >
  Triggers when the executing agent receives a markdown brief matching the
  5-part format (Objective / Context / Steps / Expected outputs / Report back),
  or when the user says "follow the brief," "use the runbook contract," "report
  back per contract," or when a cross-machine handoff is in progress ("push
  branch," "fetch first," "consumer brief"). Applies to all work in the engram
  repo. Does NOT trigger on conversational requests that do not include a brief
  block.
---

# engram-brief-contract

## Purpose

This skill enforces the canonical brief format and report-back contract used in
the Engram runbooks. It applies to **both** feature/issue work and upstream
merge work — the 2026-05-25 runbook updates reconciled both into single
definitions shared here. The skill does **not** pick which runbook applies;
runbook selection is overseer-driven. The executing agent infers from the
brief's content which report-back contract to follow (feature 6-item vs. merge
9-item). Feature is the default; merge applies only if the brief explicitly
references upstream-sync or upstream-merge work.

---

## 5-Part Brief Contract

A well-formed brief must contain exactly these five sections in order:

1. **Objective** — one sentence.
2. **Context** — only what the agent needs. If codebase context is required,
   the overseer pastes the relevant excerpt; the overseer does not fetch source.
3. **Exact commands or steps.**
4. **Expected outputs** — including acceptance criteria as the definition of
   done.
5. **Required report-back section** (see below).

If you receive a brief that is missing any of these sections, flag the gap
before starting work. Do not invent missing context.

---

## Cross-Machine Discipline

The producing agent **pushes its branch to origin** before handing off. The
consuming brief opens with a **fetch + branch-exists check** before any compute
spend. If the branch is absent on origin, stop and report the missing branch
rather than reconstructing work from scratch.

---

## Report-Back Contracts

### Which one applies?

- **Feature 6-item** — default for all feature, fix, chore, and issue work.
- **Merge 9-item** — use only when the brief explicitly concerns an upstream
  sync or upstream merge (references the merge runbook, Phase 4, or
  `upstream/main`).

---

### Feature / Issue / Bugfix — 6-Item Report-Back

1. **Outputs produced** — files, PR URL, artifacts.
2. **Commands run and exit codes.**
3. **Deviations from the brief**, if any.
4. **Quality-gate self-assessment** — syntax, types, lint, tests (pass/fail
   each).
5. **Acceptance-criteria pass/fail** — per criterion listed in the brief.
6. **Contract gate result** — pass / fail / skip-recorded-as-KHA-405-finding,
   or "n/a — contract surface not touched."

---

### Upstream Merge — 9-Item Report-Back

1. **PR URL.**
2. **Validation script output** — pass/fail for each of the 6 tests, including
   test #6 contract-conformance.
3. **Final conflict count** — expected vs. actual.
4. **Any `ENGRAM_CONFLICT_REVIEW` markers** that need human attention (post-merge
   grep output).
5. **Any new upstream CI workflows** that were fork-guarded, and whether they
   were added to `protected-paths.json`.
6. **Updated header count and block count** post-merge (broad scope; compare to
   manifest).
7. **Any files from the manifest's "Deleted Files" section** that appeared as
   conflicts and how you resolved them (should always be `git rm`).
8. **Wheel uploaded to S3?** — URL or "skipped — used hosted wheel."
9. **Contract gate result** (test #6) — pass / fail /
   skip-recorded-as-KHA-405-finding (with link).

---

## Linear Canonicals

- Feature & Issue Runbook: https://linear.app/khaentertainment/document/engram-feature-and-issue-runbook-d8916813a279
- Upstream Merge Runbook: https://linear.app/khaentertainment/document/engram-upstream-merge-runbook-a2451e78eecc
