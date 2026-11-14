# CLAUDE.md: Engram

[`AGENTS.md`](AGENTS.md) is the canonical instruction file for this repo. The
line below imports it, so its rules load with this file. Read it first.

@AGENTS.md

## Claude-Specific Notes

- **Skills.** Project skills live in `.claude/skills/` and are invoked via the
  `Skill` tool. `AGENTS.md` lists the two Engram ones.
- **Slash commands.** The `/engram-*` commands live in `.claude/commands/` and
  fire only when typed. `AGENTS.md` lists them.
- **Local notes.** If `.claude/local-notes.md` exists, it may contain local
  maintainer preferences or environment details. Treat it as local-only
  context, not public project policy.
- **Memory.** Auto-memory and any other memory system hold decisions and
  preferences only. Release status lives in the harness's `results.tsv`
  (rule 3 in `AGENTS.md`).
