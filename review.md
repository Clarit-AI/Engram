# Engram review rules

Engram is a fork of SGLang that adds persistent snapshot and restore of Mamba/SSM state. These rules apply on top of the global review prompt. Where they disagree, this file wins.

## 1. Upstream and branch hygiene

- Engram is pinned to SGLang release tags. Flag any PR that merges or syncs upstream `main`, or pulls in upstream commits beyond a deliberate cherry-pick.
- Flag any change that targets, pushes to or opens a PR against `sgl-project/sglang`. Every `gh` call in scripts and commands must name `--repo Clarit-AI/Engram`.
- Flag a new or changed dependency that has no upper bound: no version at all, or only a lower bound such as `>=1.2`. An exact pin (`==`) or a bounded range (`>=0.14.1,<0.15`) passes. An unbounded package has already broken a fresh install once.
- Flag anything that re-adds a schedule to the upstream-sync workflow or re-adds merge-upstream instructions.

## 2. Fork-marking convention

Any change to upstream-derived code must carry the fork markers:

- An `ENGRAM_MODIFIED` header comment on every upstream file the fork modifies. Fork-only files do not need one, though some carry it.
- Balanced `BEGIN ENGRAM` / `END ENGRAM` delimiters around each edit inside an otherwise-upstream file, or an `ENGRAM_CHANGED:` comment on a single changed line.
- New fork files registered in `.engram/manifest.md`.

Flag fork edits that are unmarked, unbalanced `BEGIN`/`END` pairs, and changes to a protected path that the PR does not explain. The protected paths are the globs in `.engram/policy/protected-paths.json`, plus any other file the fork has added or changed relative to upstream; `scripts/policy/check_protected_paths.py` applies both. That JSON file is the only list. This file keeps no copy of it.

## 3. Snapshot and restore is the highest-risk surface

For any change touching the scheduler, the prefill or extend path, Mamba2 metadata or mixers, the memory or mamba pool, the snapshot policy, or the snapshot save and restore handlers, say whether it could break:

- **Save:** a snapshot must hold the state at the point it claims, and its recorded token list must match that state.
- **Restore:** restored state must load into the right request and slot.
- **Consumption:** restored SSM state must reach the next forward pass. This has failed before, when restored state was never wired into the chunk-scan `initial_state` (KHA-390). Save and restore each looked correct on their own.

Known open problems as of 2026-11-14, on `main` at `7e34b5c6cf`. Check any change against them. An item comes off this list when its fix has merged and a passing row in `results.tsv` names the merge commit. A fix that changes no behaviour, such as message text, comes off when it merges. If this date is months old, read `results.tsv` before trusting the list.

- A snapshot saved after a turn can list only the prompt's tokens, even when it was taken late in the reply. Flag changes to how a snapshot's token list or length is set.
- The every-turn snapshot policy suppresses snapshots for a minimum interval, which can leave the prefill snapshot as the saved one. `/save_snapshot` copies the existing warm snapshot without refreshing it.
- A turn that produces one token cannot be saved.
- Generation after a restore can run past end-of-sequence, and restore does not carry the request's sampling settings.
- Server error text must name the real flag, `--enable-snapshot-persistence`.

When a change plausibly affects snapshot behaviour, say so directly. Do not accept the repo's existing snapshot tests as proof, because they have passed while behaviour was wrong. Recommend validating with the independent equivalence check in `Clarit-AI/engram-harness` (snapshot then restore must match an uninterrupted run), with the result recorded as a row in `results.tsv` naming commit, SGLang tag, model and GPU.

## 4. Status claims

Flag docs, changelogs, release notes or comments that claim a result (a pass, a speed, zero leaks) without a matching row in `results.tsv`. Flag a release tag proposed without one.

## Extra fix-prompt bullets

Add these to the agent fix prompt for every Engram review:

- Preserve the fork markers: keep `ENGRAM_MODIFIED` headers and balanced `BEGIN`/`END` delimiters intact, and update `.engram/manifest.md` if you add or move a fork file.
- If a fix touches the snapshot or restore surface, validate it with the engram-harness equivalence check and record the result row before calling it done. Do not rely on the repo's snapshot tests alone.
- Never push to or target `sgl-project/sglang`. Name `--repo Clarit-AI/Engram` on every `gh` call.
