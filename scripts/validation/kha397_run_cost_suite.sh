#!/usr/bin/env bash
# KHA-397 Addendum 2 — cost-suite orchestrator.
#
# Runs warm-suite (1 server lifetime) + N cold-one (N server lifetimes)
# at each target history length, then summarizes.
#
# Server is launched WITHOUT --disable-radix-cache so warm restore can
# benefit from radix-tree cache hits.

set -euo pipefail

MODEL=${MODEL:-/workspace/models/granite-4.0-h-tiny}
SNAPSHOT_DIR=${SNAPSHOT_DIR:-/workspace/snapshots}
OUT_DIR=${OUT_DIR:-/tmp/kha397-cost}
LOG_DIR=${LOG_DIR:-/tmp/kha397-cost/logs}
HOST=127.0.0.1
PORT=30002
ITERATIONS=${ITERATIONS:-5}
LENGTHS=${LENGTHS:-"512 4096"}
PYTHON=${PYTHON:-python}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COST_SCRIPT="$SCRIPT_DIR/kha397_prefill_cost.py"

mkdir -p "$OUT_DIR" "$LOG_DIR"

launch_server() {
    # Echo ONLY the logfile path; informational messages go to stderr so
    # this function can be used in $(launch_server ...) cleanly.
    local tag=$1
    local logfile="$LOG_DIR/server-${tag}.log"
    SGLANG_ENABLE_SPEC_V2=false \
        python -m sglang.launch_server \
            --model-path "$MODEL" \
            --host "$HOST" --port "$PORT" \
            --enable-snapshot-persistence \
            --snapshot-dir "$SNAPSHOT_DIR" \
            --mamba-scheduler-strategy no_buffer \
            --trust-remote-code \
            >"$logfile" 2>&1 &
    SERVER_PID=$!
    echo "[orchestrator] launched server tag=$tag pid=$SERVER_PID log=$logfile" >&2
    echo "$logfile"
}

wait_ready() {
    local i=0
    until curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; do
        sleep 3
        i=$((i+1))
        if [ $i -gt 60 ]; then
            echo "[orchestrator] FAIL: server not ready after 180s"
            return 1
        fi
    done
    echo "[orchestrator] server ready"
}

kill_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
    sleep 4
    # Kill any leftover GPU-holding worker
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [ -n "$gpu_pids" ]; then
        for p in $gpu_pids; do kill "$p" 2>/dev/null || true; done
        sleep 4
    fi
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | awk '{print $1}')
    echo "[orchestrator] gpu_used=${used}MiB after kill"
}

for L in $LENGTHS; do
    echo ""
    echo "================ HISTORY LENGTH: $L ================"
    # Phase A: warm + baseline (one server lifetime, creates snapshots)
    WARM_LOG=$(launch_server "warm-$L")
    wait_ready
    "$PYTHON" "$COST_SCRIPT" --mode warm-suite \
        --history-tokens "$L" --iterations "$ITERATIONS" \
        --server-log "$WARM_LOG" --out-dir "$OUT_DIR" \
        --model-path "$MODEL" --host "$HOST" --port "$PORT"
    kill_server

    # Phase B: N cold iterations, each a fresh server lifetime
    for i in $(seq 1 "$ITERATIONS"); do
        COLD_LOG=$(launch_server "cold-$L-$i")
        wait_ready
        "$PYTHON" "$COST_SCRIPT" --mode cold-one \
            --history-tokens "$L" --iter "$i" \
            --server-log "$COLD_LOG" --out-dir "$OUT_DIR" \
            --model-path "$MODEL" --host "$HOST" --port "$PORT"
        kill_server
    done

    # Phase C: summarize this length
    "$PYTHON" "$COST_SCRIPT" --mode summarize \
        --history-tokens "$L" --out-dir "$OUT_DIR"
done

echo ""
echo "================ ALL DONE ================"
ls -la "$OUT_DIR"/summary-*.json
