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

This skill enforces the brief format and report-back contract from the Engram
Feature & Issue Runbook. It covers feature, fix, chore and issue work.

Engram no longer merges `upstream/main`, so the upstream-merge contract that
this skill used to carry is gone. If a brief asks for an upstream merge or an
upstream sync, stop and flag it to the overseer before doing anything: that
work is out of policy (see "Rules That Come First" in `AGENTS.md`).

---

## 5-Part Brief Contract

A well-formed brief must contain exactly these five sections in order:

1. **Objective**: one sentence.
2. **Context**: only what the agent needs. If codebase context is required,
   the overseer pastes the relevant excerpt; the overseer does not fetch source.
3. **Exact commands or steps.**
4. **Expected outputs**, including acceptance criteria as the definition of
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

"Origin" means `Clarit-AI/Engram`. Nothing is pushed to `sgl-project/sglang`.

---

## 6-Item Report-Back

1. **Outputs produced**: files, PR URL, artifacts.
2. **Commands run and exit codes.**
3. **Deviations from the brief**, if any.
4. **Quality-gate self-assessment**: syntax, types, lint, tests (pass/fail
   each). A gate that could not run in the environment is reported as "not
   run", never as a pass.
5. **Acceptance-criteria pass/fail**, per criterion listed in the brief.
6. **Contract gate result**: pass / fail / skip-recorded-as-KHA-405-finding,
   or "n/a: contract surface not touched."

### Evidence rule

Any statement in a report-back that something works, passes or is fixed on a
GPU cites its row in the release-test harness's `results.tsv`: commit SHA,
model, and log folder. A result with no row is reported as "not tested", and a
result with no commit SHA is reported as "unversioned".

---

## Linear Canonical

- Feature & Issue Runbook: https://linear.app/khaentertainment/document/engram-feature-and-issue-runbook-d8916813a279
