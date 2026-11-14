# Engram Git Hooks

Mechanical enforcement layer for Engram's development runbook. Hooks live here
and are shared via version control. They are **opt-in per checkout**: until you
run the activation command, nothing in this folder runs.

## Activation (one-time per checkout)

```bash
git config core.hooksPath .engram/hooks
```

Git uses one hook folder at a time. Activating this folder replaces any other
hook folder, including `.husky`. The `pre-push` hook here carries the upstream
guard that `.husky/pre-push` used to provide, so nothing is lost by switching.

Check the current state with:

```bash
git config --get core.hooksPath
```

---

## Hooks

### `pre-push`

Runs automatically on `git push`. Enforces four gates in order:

1. **No push to upstream.** Refuses any push to a remote named `upstream` or
   whose URL is `sgl-project/sglang`. Engram pins to SGLang release tags and
   does not push to the upstream repo.
2. **No direct push to `main`.** Aborts with a message directing you to open a
   PR.
3. **Lint gate.** Runs `ruff check --select=F401,F821` over `python/`, `test/`,
   `benchmark/`, and `examples/`. Aborts on first failure.
4. **CPU-only test gate.** Runs `pytest -q test/srt/cpu/`. This is the
   canonical CPU-runnable kernel test suite sourced from `.github/workflows/pr-test-xeon.yml`
   and the `suite_xeon` definition in `test/srt/run_suite.py`. Aborts on failure.

Gates 3 and 4 need a Python environment with `ruff`, `pytest` and an installed
`sglang`. The hook uses `python` by default. Point it at another interpreter
with `ENGRAM_PYTHON`:

```bash
ENGRAM_PYTHON=/path/to/venv/bin/python git push
```

If that interpreter has no `ruff` or `pytest`, the hook stops and says the gate
could not run. It does not report a lint failure it never checked.

> **Do NOT call `pre-commit` from this hook.** The pre-commit binary in the
> engram venv has a stale shebang pointing to the deleted `sglang-mamba` path
> (tracked in KHA-400). Lint and tests are invoked directly instead.

### `start-feature.sh`

Helper script (not a Git-triggered hook) for starting feature or fix branches
off a fresh `origin/main`.

```bash
.engram/hooks/start-feature.sh feat/my-feature
.engram/hooks/start-feature.sh fix/kha-123-crash
.engram/hooks/start-feature.sh chore/update-deps
```

**What it does:**
- Validates the slug matches `(feat|fix|chore)/[a-z0-9-]+`
- Refuses to run if the branch already exists locally or on `origin` (prevents
  accidental clobber)
- Fetches `origin main`, creates the branch off `origin/main`, and prints the
  base commit SHA

---

## Skipping the Lint and Test Gates

```bash
ENGRAM_SKIP_GATE=1 git push
```

Skips gates 3 and 4 and prints a warning. Gates 1 and 2 still run, so a push to
upstream or to `main` is still refused. Use it when:

- The machine has no environment that can run the gates
- You're pushing a work-in-progress branch with known lint failures you'll fix
  before merging

The fork's hardware CI is switched off (every job carries
`if: github.repository == 'sgl-project/sglang'`), so no CI run repeats these
gates. A branch pushed with `ENGRAM_SKIP_GATE=1` has not been linted or tested
until someone runs `/engram-gate` on it.

## `--no-verify`

```bash
git push --no-verify
```

Skips the pre-push hook entirely, including the upstream and `main` refusals.
Prefer `ENGRAM_SKIP_GATE=1`, which keeps them.

---

## Runbook Cross-Reference

These hooks implement the **mechanical enforcement layer** described in the
[Engram Feature & Issue Runbook](https://linear.app/khaentertainment/document/engram-feature-and-issue-runbook-d8916813a279).
