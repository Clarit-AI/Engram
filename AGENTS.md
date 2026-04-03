# AGENTS.md — Engram

**Engram** (formerly **sglang-mamba**) — fork of [SGLang](https://github.com/sgl-project/sglang) adding persistent snapshot/statefulness support for Mamba-family models.

> **Agent note:** Legacy references to `sglang-mamba` may still appear in paths,
> test artifacts, and history. They refer to this same project.

## Session Start

Before changing code or docs:

```bash
# Surface project state and recent history
memory_search("sglang mamba")
memory_search("sglang mamba backlog issues")

# Read local instructions
sed -n '1,220p' CLAUDE.md
```

## Linear And GitHub Workflow

- Use Core Memory MCP for Linear and GitHub project tracking.
- Do not use the native Linear CLI or native Linear MCP for this repo.
- Prefer Core Memory MCP for project-state writes such as issue updates and comments.

For local `gh` usage that is still needed for repository operations, verify the
active account first:

```bash
gh auth status
gh auth switch --hostname github.com --user Clarit-AI
```

This repository's `origin` is `Clarit-AI/Engram`. If `gh` is active on
`KHAEntertainment`, switch to `Clarit-AI` before creating or editing PRs.

## Canonical Docs

- User-friendly API guide: `docs/stateful_mamba/api_guide.md`
- Canonical technical spec: `docs/stateful_mamba/http_api_spec.md`
- Historical docs only: `docs/stateful_mamba/.archive/`

Anything under `docs/stateful_mamba/.archive/` is historical reference only and
must not drive implementation or product decisions.
