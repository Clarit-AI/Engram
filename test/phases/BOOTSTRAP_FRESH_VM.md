# Bootstrap — Fresh VM from Zero

Use this when there is no disk to migrate and the environment must be rebuilt completely.
Estimated time to first passing unit test: ~20 minutes (excluding model download).

---

## 0. Hardware check (do this first, non-negotiable)

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

Must show compute capability **7.5 or higher** (T4=7.5, A100=8.0, A10G=8.6).
If it shows 7.0 (V100), stop — get a different instance. No workaround exists.

---

## 1. System packages

```bash
# Ubuntu 22.04 / Debian 12 assumed
sudo apt-get update -qq
sudo apt-get install -y python3 python3-pip python3-dev git curl wget \
    build-essential libfuse2   # libfuse2 needed for hf-mount
```

---

## 2. CUDA

This project requires **CUDA 12.x** (torch was built against 12.8 on the original VM).
If the VM was provisioned with a GPU image, CUDA is likely already present.

```bash
nvcc --version   # should show release 12.x
# If missing, install CUDA 12.8 from https://developer.nvidia.com/cuda-downloads
# Choose: Linux → x86_64 → Ubuntu → 22.04 → deb(network)
```

---

## 3. Clone the repo

```bash
git clone https://github.com/clarit-ai/engram.git
cd sglang-mamba
# main branch — all test fixes and docs are here
```

---

## 4. Python dependencies

```bash
# Requires Python 3.10+; 3.11 was used on original VM
python3 --version

pip install --upgrade pip

# Core install — this pulls in torch, flashinfer, triton, transformers, etc.
pip install -e "python[all]" --break-system-packages

# Verify torch sees CUDA and correct GPU
python3 -c "
import torch
cap = torch.cuda.get_device_capability()
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute: {cap[0]}.{cap[1]}')
print(f'CUDA:    {torch.version.cuda}')
assert cap[0]*10+cap[1] >= 75, 'NEED sm75+, got sm{}{}'.format(*cap)
print('OK')
"
```

---

## 5. Get the model (pick one method)

### Fastest: hf-mount (lazy FUSE — no full download)

```bash
pip install hf-mount

mkdir -p ~/models/granite-4.0-h-tiny

# Mount the IBM Granite Mamba hybrid model
sudo hf-mount ibm-granite/granite-4.0-tiny-mamba2-instruct \
    ~/models/granite-4.0-h-tiny

export MODEL_PATH=~/models/granite-4.0-h-tiny
```

> **Note**: verify the exact HuggingFace repo ID at https://huggingface.co/ibm-granite
> if `granite-4.0-tiny-mamba2-instruct` 404s, search for `granite-4.0-h-tiny` or
> `granite-mamba` — IBM may rename between releases.

### Alternative: direct download

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'ibm-granite/granite-4.0-tiny-mamba2-instruct',
    local_dir='$HOME/models/granite-4.0-h-tiny'
)
"
export MODEL_PATH=~/models/granite-4.0-h-tiny
```

---

## 6. Update config.sh

```bash
cd ~/sglang-mamba
sed -i "s|/home/jeanclawdai/models/granite-4.0-h-tiny|$MODEL_PATH|g" \
    test/phases/config.sh

source test/phases/config.sh   # should print model path without errors
```

---

## 7. Smoke test — unit phases (no server, ~90 seconds)

```bash
cd ~/sglang-mamba
python3 -m pytest \
    test/registered/radix_cache/test_mamba_unittest.py::TestMamba::test_mamba_radix_cache_1 \
    test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py \
    test/registered/radix_cache/test_mamba_radix_cache_gauntlet.py \
    -v
# Expected: 16 passed
```

If any of these fail, the install or repo state is broken — do not proceed to server phases.

---

## 8. Restore Claude context (optional but saves orientation time)

If you have the backup files from the old VM (see `NEW_VM_SETUP.md` §9):

```bash
# Adjust paths: the memory dir mirrors the absolute clone path
MEMORY_DIR="$HOME/.claude/projects/$(echo $HOME/sglang-mamba | tr '/' '-' | sed 's/^-//')/memory"
mkdir -p "$MEMORY_DIR" ~/.remember

cp backup/memory/*.md "$MEMORY_DIR/"
cp backup/remember.md backup/now.md backup/today-*.md ~/.remember/
cp backup/settings.json ~/.claude/settings.json
```

If backup is unavailable, Claude can reconstruct context from the git log and
`test/phases/results/` reports — just read `NEW_VM_SETUP.md` at session start.

---

## 9. Install Claude Code CLI

```bash
# If not already installed
npm install -g @anthropic-ai/claude-code
# or via pip (check current install method at https://docs.anthropic.com/claude-code)
```

---

## 10. Start server phase testing

```bash
cd ~/sglang-mamba
source test/phases/config.sh

# Run in order — stop at first failure and diagnose:
# Phase 1 → test/phases/phase-01-stateless-inference-baseline.md
# Phase 4 → test/phases/phase-04-live-server-integration-no-buffer.md
# Phase 7 → test/phases/phase-07-snapshot-system.md   ← most critical
# Phase 6 → test/phases/phase-06-extra-buffer-strategy.md
# Phase 8 → test/phases/phase-08-gauntlet-stress-tests.md
```

Each phase doc is self-contained. Read the **Environment Setup** section of each before
running anything.

---

## Reference versions (from original working VM)

| Package | Version |
|---------|---------|
| Python | 3.11.2 |
| torch | 2.9.1 |
| torch CUDA | 12.8 |
| CUDA toolkit | 11.8 (system) / 12.8 (torch-bundled) |
| flashinfer-python | 0.6.3 |
| triton | 3.5.1 |
| Claude Code | 2.1.86 |

If `pip install -e "python[all]"` pulls different versions and tests fail,
pin to these as a fallback.
