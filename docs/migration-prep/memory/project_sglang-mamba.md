---
name: sglang-mamba project overview
description: Core facts about the stateful Mamba SGLang fork — repo, model, infrastructure, server flags
type: project
---

## Repository

- **GitHub**: https://github.com/clarit-ai/engram
- **Primary branch**: `main`
- **Local clone** (on GCloud instance): the working directory is typically `/home/bbrenner/sglang-mamba` when cloned fresh; the GCloud instance is at `<INSTANCE_NAME>` / `<ZONE>` / `<PROJECT_ID>` (see CLAUDE.md for placeholder values)
- **Install**: `pip install -e python/ [--break-system-packages]` — `.venv` was empty as of 2026-03-24; system-wide install was used

## Model

- **Primary**: `granite-4.0-h-tiny` (`GraniteMoeHybridForCausalLM`, 40-layer Mamba/attention hybrid, hidden_size=1536, 3 safetensors shards)
- **Path on GCloud**: `/home/jeanclawdai/models/granite-4.0-h-tiny`
- **Fallback models**: Nemotron-4B (`NVIDIA-Nemotron-3-Nano-4B-BF16`), Granite-Q4 GGUF
- `config.sh` in `test/phases/` is the single source of truth for MODEL_PATH, MODEL_NAME, SERVER_PORT, etc. — edit one file to swap models

## Server flags (confirmed from server_args.py)

- `--enable-snapshot-persistence` (NOT `--enable-mamba-snapshots` — that flag does NOT exist)
- `--snapshot-dir <path>`
- `--mamba-scheduler-strategy [no_buffer | extra_buffer]`
- `--disable-radix-cache`
- Default port: 30000 (configurable via `$SERVER_PORT` in test scripts)

## Web UI (chatbot)

- Next.js app at `localhost:3000` (Vercel AI Chatbot template)
- `SGLANG_BASE_URL=http://localhost:30000/v1`, `SGLANG_MODEL_ID=default`
- **CRITICAL**: always sends full Postgres conversation history with every request (client-side re-injection via `getMessagesByChatId`). This is NOT server-side state recall — it cannot prove Mamba state persistence. HITL tests that involve the web UI are smoke checks only, not state proofs.

## Key source files

- `python/sglang/srt/mem_cache/memory_pool.py` — HybridReqToTokenPool, MambaPool, HybridLinearKVPool
- `python/sglang/srt/mem_cache/mamba_radix_cache.py` — MambaRadixCache, dual LRU, tombstone nodes, COW
- `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` — ForwardMetadata, Mamba2Metadata
- `python/sglang/srt/snapshot/mamba_snapshot.py` — MambaSnapshotMetadata, save/load
- `python/sglang/srt/managers/scheduler.py` — handle_save_snapshot, handle_restore_snapshot
- `python/sglang/srt/managers/io_struct.py` — RestoreSnapshotReqOutput, EmbeddingReqInput
- `test/phases/` — phase execution documents (phases 0–8), config.sh, codemap.md, results/

**Why:** This is a long-running project spanning multiple sessions. Losing these facts wastes orientation time at the start of each session.
**How to apply:** Use when starting a new session on sglang-mamba work to orient quickly without re-reading files.
