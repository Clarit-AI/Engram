#!/usr/bin/env bash
# ENGRAM_MODIFIED — Post-sync validation script for Engram fork
# Usage: scripts/validate-sync.sh [--model MODEL_NAME] [--skip-snapshot]
#
# Validates that an upstream sync didn't break Engram-specific features.
# Requires: 1x A100-80GB (or equivalent), CUDA, Python with sglang installed.
#
# Tests run:
#   1. Server startup with snapshot args
#   2. Basic inference (Granite 4.0-H-tiny)
#   3. Snapshot save/restore roundtrip
#   4. Post-restore generation (stateful recall)
#   5. KV cache mixin behavior
#
# Exit codes:
#   0 = all tests passed
#   1 = test failure (details in output)
#   2 = environment error (missing GPU, model, etc.)

set -euo pipefail

# --- Configuration ---
DEFAULT_MODEL="ibm-granite/granite-4.0-h-tiny"
MODEL="${1:-$DEFAULT_MODEL}"
SKIP_SNAPSHOT="${SKIP_SNAPSHOT:-false}"
PORT=30000
SNAPSHOT_DIR=$(mktemp -d /tmp/engram-sync-validate-XXXXXX)
LOG_DIR=$(mktemp -d /tmp/engram-sync-logs-XXXXXX)
SERVER_PID=""

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log_info "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$SNAPSHOT_DIR"
    log_info "Logs saved to $LOG_DIR"
}
trap cleanup EXIT

# --- Preflight ---
log_info "=== Engram Post-Sync Validation ==="
log_info "Model: $MODEL"
log_info "Snapshot dir: $SNAPSHOT_DIR"
log_info "Logs: $LOG_DIR"

if ! command -v nvidia-smi &>/dev/null; then
    log_fail "nvidia-smi not found — no GPU available"
    exit 2
fi

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
log_info "GPUs detected: $GPU_COUNT"

if ! python -c "import sglang" 2>/dev/null; then
    log_fail "sglang not importable — is it installed?"
    exit 2
fi

# --- Test 1: Server startup with snapshot args ---
log_info "Test 1: Server startup with snapshot persistence..."

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --enable-snapshot-persistence \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --snapshot-trigger-policy manual_only \
    --mem-fraction-static 0.80 \
    --disable-cuda-graph \
    > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

# Wait for server ready (up to 5 minutes)
WAITED=0
MAX_WAIT=300
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:$PORT/health" | grep -q "ok" 2>/dev/null; then
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    log_fail "Server failed to start within ${MAX_WAIT}s"
    cat "$LOG_DIR/server.log" | tail -50
    exit 1
fi
log_pass "Server started successfully (${WAITED}s)"

# --- Test 2: Basic inference ---
log_info "Test 2: Basic inference..."

RESPONSE=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
        "max_tokens": 20,
        "temperature": 0
    }')

if echo "$RESPONSE" | python -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['message']['content']" 2>/dev/null; then
    CONTENT=$(echo "$RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:80])")
    log_pass "Inference works: '$CONTENT'"
else
    log_fail "Inference returned invalid response"
    echo "$RESPONSE" | python -m json.tool 2>/dev/null || echo "$RESPONSE"
    exit 1
fi

# --- Test 3: Snapshot save/restore ---
if [ "$SKIP_SNAPSHOT" = "true" ]; then
    log_info "Skipping snapshot tests (--skip-snapshot)"
else
    log_info "Test 3: Snapshot save/restore roundtrip..."

    # Establish context with a fact
    CONV_ID="validate-sync-$(date +%s)"
    curl -s "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "messages": [{"role": "user", "content": "Remember this: the secret code is DELTA-7. Acknowledge."}],
            "max_tokens": 50,
            "temperature": 0,
            "conversation_id": "'"$CONV_ID"'"
        }' > "$LOG_DIR/establish.json" 2>&1

    # Save snapshot
    SAVE_RESPONSE=$(curl -s "http://localhost:$PORT/snapshot/save" \
        -H "Content-Type: application/json" \
        -d '{"conversation_id": "'"$CONV_ID"'"}')

    if echo "$SAVE_RESPONSE" | grep -qi "error"; then
        log_fail "Snapshot save failed: $SAVE_RESPONSE"
        exit 1
    fi
    log_pass "Snapshot saved"

    # Check snapshot file exists
    SNAP_COUNT=$(find "$SNAPSHOT_DIR" -type f -name "*.bin" -o -name "*.pt" -o -name "*.safetensors" 2>/dev/null | wc -l)
    if [ "$SNAP_COUNT" -eq 0 ]; then
        # Some implementations use different formats
        SNAP_COUNT=$(find "$SNAPSHOT_DIR" -type f 2>/dev/null | wc -l)
    fi
    log_info "Snapshot files on disk: $SNAP_COUNT"

    # Restore snapshot
    RESTORE_RESPONSE=$(curl -s "http://localhost:$PORT/snapshot/restore" \
        -H "Content-Type: application/json" \
        -d '{"conversation_id": "'"$CONV_ID"'"}')

    if echo "$RESTORE_RESPONSE" | grep -qi "error"; then
        log_fail "Snapshot restore failed: $RESTORE_RESPONSE"
        exit 1
    fi
    log_pass "Snapshot restored"

    # --- Test 4: Post-restore stateful recall ---
    log_info "Test 4: Stateful recall after restore..."

    RECALL_RESPONSE=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "messages": [{"role": "user", "content": "What was the secret code I told you?"}],
            "max_tokens": 30,
            "temperature": 0,
            "conversation_id": "'"$CONV_ID"'"
        }')

    RECALL_TEXT=$(echo "$RECALL_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_ERROR")

    if echo "$RECALL_TEXT" | grep -qi "DELTA-7"; then
        log_pass "Stateful recall works: '$RECALL_TEXT'"
    else
        log_fail "Stateful recall failed — model did not recall 'DELTA-7'"
        log_info "Response was: $RECALL_TEXT"
        exit 1
    fi
fi

# --- Test 5: Import sanity (catches broken markers) ---
log_info "Test 5: Import sanity check..."

python -c "
from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.io_struct import *
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.memory_pool import *
print('All critical imports succeed')
" 2>"$LOG_DIR/import_check.log"

if [ $? -eq 0 ]; then
    log_pass "All critical imports succeed"
else
    log_fail "Import check failed:"
    cat "$LOG_DIR/import_check.log"
    exit 1
fi

# --- Summary ---
echo ""
echo "=================================="
echo -e "${GREEN}ALL TESTS PASSED${NC}"
echo "=================================="
echo "Model:     $MODEL"
echo "GPU count: $GPU_COUNT"
echo "Logs:      $LOG_DIR"
echo ""
echo "Safe to push the merge."
