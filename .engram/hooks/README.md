# Engram Git Hooks

Mechanical enforcement layer for Engram's development runbooks. Hooks live here
and are shared via version control; they are **opt-in per checkout**.

## Activation (one-time per checkout)

```bash
git config core.hooksPath .engram/hooks
```

That's it. All contributors on the same checkout then share the same guards.

---

## Hooks

### `pre-push`

Runs automatically on `git push`. Enforces three gates in order:

1. **No direct push to `main`** — aborts immediately with a message directing
   you to open a PR. See the three-layer enforcement section in either runbook.
2. **Lint gate** — runs `ruff check --select=F401,F821` over `python/`, `test/`,
   `benchmark/`, and `examples/`. Aborts on first failure.
3. **CPU-only test gate** — runs `pytest -q test/srt/cpu/`. This is the
   canonical CPU-runnable kernel test suite sourced from `.github/workflows/pr-test-xeon.yml`
   and the `suite_xeon` definition in `test/srt/run_suite.py`. Aborts on failure.

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

## `--no-verify` Escape Hatch

```bash
git push --no-verify
```

Skips the pre-push hook entirely. Use only when:
- You're pushing a work-in-progress branch with known lint failures you'll fix
  before merging
- CI already covers the same gates and local test infra is unavailable

Do **not** use `--no-verify` on anything that goes to main.

---

## Runbook Cross-References

These hooks implement the **mechanical enforcement layer** described in:

- [Engram Feature & Issue Runbook — Three-Layer Enforcement](https://linear.app/khaentertainment/document/engram-feature-and-issue-runbook-d8916813a279)
- [Engram Upstream Merge Runbook — Three-Layer Enforcement](https://linear.app/khaentertainment/document/engram-upstream-merge-runbook-a2451e78eecc)
