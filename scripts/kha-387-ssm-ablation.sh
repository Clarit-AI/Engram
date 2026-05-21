#!/usr/bin/env bash
# ENGRAM_DIAGNOSTIC — KHA-387 SSM ablation runner
#
# Launches a fresh server, runs the 5-condition ablation in
# scripts/kha-387-ssm-ablation.py, captures the report + server log.
#
# Usage:
#   scripts/kha-387-ssm-ablation.sh                       # Codestral default
#   scripts/kha-387-ssm-ablation.sh --model /path/to/m    # override model
#   scripts/kha-387-ssm-ablation.sh --port 30200          # override port
#
# Exit codes:
#   0  — harness completed; verdict in $RUN_DIR/report.md
#   2  — environment error (missing GPU, model unreachable, server failed)
#   3  — harness exited non-zero

set -euo pipefail

# --- Configuration ---
MODEL="${MODEL:-/workspace/models/mamba-codestral-7b}"
SERVED_NAME="${SERVED_NAME:-mamba-codestral-7b}"
PORT="${PORT:-30200}"
RUN_DIR="${RUN_DIR:-/workspace/runs/kha-387-ssm-ablation-$(date -u +%Y%m%dT%H%M%SZ)}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --served-name) SERVED_NAME="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$RUN_DIR"
SNAP_DIR="$RUN_DIR/snapshots"
mkdir -p "$SNAP_DIR"
SERVER_LOG="$RUN_DIR/server.log"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${YELLOW}[INFO]${NC}  $1"; }
log_pass()  { echo -e "${GREEN}[PASS]${NC}  $1"; }
log_fail()  { echo -e "${RED}[FAIL]${NC}  $1"; }

log_info "KHA-387 SSM ablation"
log_info "Model:        $MODEL"
log_info "Served-name:  $SERVED_NAME"
log_info "Port:         $PORT"
log_info "Run dir:      $RUN_DIR"
log_info "Snapshot dir: $SNAP_DIR"

# --- Preflight ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log_fail "nvidia-smi not found"; exit 2
fi
if ! python -c "import sglang" 2>/dev/null; then
    log_fail "sglang not importable"; exit 2
fi
if ! python -c "import safetensors" 2>/dev/null; then
    log_fail "safetensors not importable"; exit 2
fi

# --- Launch server ---
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log_info "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

log_info "Launching server..."
TORCHDYNAMO_DISABLE=1 SGLANG_ENABLE_SPEC_V2=false \
    python -m sglang.launch_server \
        --model-path "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --trust-remote-code \
        --served-model-name "$SERVED_NAME" \
        --enable-snapshot-persistence \
        --snapshot-dir "$SNAP_DIR" \
        --snapshot-trigger-policy every_turn \
        --mamba-scheduler-strategy no_buffer \
        --disable-radix-cache \
        --disable-cuda-graph \
        --disable-piecewise-cuda-graph \
        > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
log_info "Server PID: $SERVER_PID"

# --- Wait for /health ---
WAITED=0
MAX_WAIT=600
while [[ $WAITED -lt $MAX_WAIT ]]; do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log_fail "Server died during startup; see $SERVER_LOG"
        tail -50 "$SERVER_LOG"
        exit 2
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
if [[ $WAITED -ge $MAX_WAIT ]]; then
    log_fail "Server did not become healthy within ${MAX_WAIT}s; see $SERVER_LOG"
    tail -50 "$SERVER_LOG"
    exit 2
fi
log_pass "Server healthy after ${WAITED}s"

# --- Run ablation ---
log_info "Running ablation cases..."
if ! python "$(dirname "$0")/kha-387-ssm-ablation.py" \
        --server-url "http://localhost:$PORT" \
        --served-name "$SERVED_NAME" \
        --snapshot-dir "$SNAP_DIR" \
        --run-dir "$RUN_DIR"; then
    log_fail "Ablation harness exited non-zero"
    exit 3
fi

log_pass "Ablation complete"
log_info "Report:    $RUN_DIR/report.md"
log_info "Results:   $RUN_DIR/results.json"
log_info "Per-case:  $RUN_DIR/log.txt"
log_info "Server:    $SERVER_LOG"
