# New VM Setup — Engram Server Phase Testing

Use this doc when spinning up a fresh VM to resume server phases (1/4/6/7/8).

---

## Hardware Requirement

**GPU must be sm75 or higher** (A100, A10G, T4, RTX 3090+).

V100 (sm70) is **incompatible** — FLA Mamba2 kernels and FlashInfer both require sm75+. This is a hard block; no workaround exists for hybrid Mamba models.

---

## 1. Clone & Install

```bash
git clone https://github.com/clarit-ai/engram.git
cd sglang-mamba
git checkout main

pip install -e "python[all]"

# Verify GPU
python -c "import torch; cap = torch.cuda.get_device_capability(); print(cap); assert cap[0]*10+cap[1] >= 75, 'Need sm75+'"
```

---

## 2. Get the Model

### Option A — hf-mount (no pre-download, lazy FUSE mount)

```bash
# Install hf-mount (see https://github.com/huggingface/hf-mount)
pip install hf-mount

# Mount the granite model (check exact repo ID on HuggingFace)
mkdir -p /mnt/models/granite-4.0-h-tiny
hf-mount ibm-granite/granite-4.0-tiny-mamba2-instruct /mnt/models/granite-4.0-h-tiny

export MODEL_PATH=/mnt/models/granite-4.0-h-tiny
```

### Option B — Direct download

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ibm-granite/granite-4.0-tiny-mamba2-instruct',
                  local_dir='/home/ubuntu/models/granite-4.0-h-tiny')
"
export MODEL_PATH=/home/ubuntu/models/granite-4.0-h-tiny
```

### Option C — Transfer from old VM (if still accessible)

```bash
# From old VM:
gcloud compute scp --recurse \
    /home/jeanclawdai/models/granite-4.0-h-tiny \
    NEW_INSTANCE:/home/ubuntu/models/

export MODEL_PATH=/home/ubuntu/models/granite-4.0-h-tiny
```

---

## 3. Update config.sh

```bash
# Edit MODEL_PATH in test/phases/config.sh to match your model location:
sed -i "s|/home/jeanclawdai/models/granite-4.0-h-tiny|$MODEL_PATH|g" \
    test/phases/config.sh

# Verify
source test/phases/config.sh
```

---

## 4. Confirm Unit Tests Still Pass (no server needed)

```bash
python -m pytest \
    test/registered/radix_cache/test_mamba_unittest.py::TestMamba::test_mamba_radix_cache_1 \
    test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py \
    test/registered/radix_cache/test_mamba_radix_cache_gauntlet.py \
    -v
# Expected: 16 passed
```

---

## 5. Run Server Phases (in order)

Each phase document is self-contained. Source config.sh first, then follow the doc.

```bash
source test/phases/config.sh

# Phase 1 — stateless baseline (no radix cache)
# Follow: test/phases/phase-01-stateless-inference-baseline.md

# Phase 4 — live server with radix cache (no_buffer strategy)
# Follow: test/phases/phase-04-live-server-integration-no-buffer.md

# Phase 7 — snapshot system end-to-end (validates gap fixes from PRs #4 #6)
# Follow: test/phases/phase-07-snapshot-system.md

# Phase 6 — extra_buffer strategy
# Follow: test/phases/phase-06-extra-buffer-strategy.md

# Phase 8 — gauntlet stress tests
# Follow: test/phases/phase-08-gauntlet-stress-tests.md
```

**Stop at first failure and diagnose before continuing.**

---

## 6. Phase Status at Last Checkpoint (2026-03-28)

| Phase | Requires Server | Status |
|-------|----------------|--------|
| 0 — Environment verification | No | **PASS** |
| 1 — Stateless inference baseline | Yes | INCOMPLETE |
| 2 — MambaPool unit tests | No | **PASS** |
| 3 — MambaRadixCache gauntlet | No | **PASS** |
| 4 — Live server, no_buffer | Yes | INCOMPLETE |
| 5 — Mamba2Metadata integrity | No | **PASS** |
| 6 — extra_buffer strategy | Yes | INCOMPLETE |
| 7 — Snapshot system e2e | Yes | INCOMPLETE |
| 8 — Gauntlet stress tests | Yes | INCOMPLETE |

Results for completed phases: `test/phases/results/`

---

## 7. Key Snapshot/Restore Context (Phase 7)

Three gap fixes landed in PRs #4 and #6 before server testing was interrupted:

- **Gap 1**: Restored requests sync `fill_ids`/`origin_input_ids` with restored Mamba state
- **Gap 2**: `create_new_request` restore correctly creates fresh request preserving conversation namespace
- **Gap 3**: Startup restore preloads latest snapshots into WARM tier (was only logging them)

Phase 7 validates all three. It is the most important server phase.

---

## 8. Nemotron Fallback

If granite OOMs on the new GPU (unlikely with sm75+ since it's the sm70 kernel support, not VRAM, that was the issue):

```bash
# In config.sh or as env override:
MODEL_PATH=$NEMOTRON_MODEL_PATH MODEL_NAME=$NEMOTRON_MODEL_NAME source test/phases/config.sh
```

---

## 9. Files to Copy Off the Old VM Before Shutdown

Everything in the repo itself is safe — it's all committed and pushed to `clarit-ai/engram`. The items below are **local-only** (gitignored or outside the repo) and will be lost if the VM is deleted.

### High priority (context you'd otherwise reconstruct from scratch)

| Source path on old VM | What it is | Copy to |
|----------------------|------------|---------|
| `~/.remember/remember.md` | Session handoff note (last state / next steps) | Any safe location |
| `~/.remember/now.md` | Rolling session memory buffer (recent work log) | Any safe location |
| `~/.remember/today-2026-03-28.md` | Compressed daily summary | Any safe location |
| `~/.claude/projects/-home-bbrenner-sglang-mamba/memory/*.md` | All persistent memory files (project context, feedback, user prefs) | `~/.claude/projects/.../memory/` on new VM |
| `~/.claude/settings.json` | Claude Code plugin config (`enabledPlugins`, `skipDangerousModePermissionPrompt`) | `~/.claude/settings.json` on new VM |

### Batch copy command (run from old VM)

```bash
# Replace DESTINATION with your local machine or GCS bucket path
DESTINATION="your-local-machine:~/sglang-mamba-backup/"

# Memory files (Claude context)
scp ~/.remember/remember.md \
    ~/.remember/now.md \
    ~/.remember/today-2026-03-28.md \
    $DESTINATION

# Project-scoped memory
scp -r ~/.claude/projects/-home-bbrenner-sglang-mamba/memory/ \
    $DESTINATION

# Claude settings
scp ~/.claude/settings.json $DESTINATION
```

### To restore on a new VM

```bash
# After cloning the repo and installing Claude Code:
mkdir -p ~/.remember
cp remember.md now.md today-*.md ~/.remember/

mkdir -p "~/.claude/projects/-home-USER-sglang-mamba/memory/"
# NOTE: adjust the path — it mirrors the absolute path of the clone on the new VM
# e.g., if cloned to /home/ubuntu/sglang-mamba the dir is:
# ~/.claude/projects/-home-ubuntu-sglang-mamba/memory/
cp memory/*.md "~/.claude/projects/-home-ubuntu-sglang-mamba/memory/"

cp settings.json ~/.claude/settings.json
```

### Not needed (all committed to git)

- `test/phases/results/*.md` — phase reports ✓ committed
- `test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py` ✓ committed
- `test/registered/radix_cache/test_mamba_radix_cache_gauntlet.py` ✓ committed
- All source code changes ✓ committed and pushed to main
