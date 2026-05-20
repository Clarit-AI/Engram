#!/usr/bin/env bash
# ENGRAM_DIAGNOSTIC — KHA-301 multi-tenant concurrency smoke
#
# Smoke probe: 4 concurrent sessions against a running sglang server.
# Each session is established under a distinct rid, saves a snapshot,
# restores it, asks a recall question, and asserts cross-talk isolation.
#
# NOT a benchmark. No load generator, no percentile capture. The smoke
# only asserts:
#   (a) all four concurrent save+restore cycles return 200, and
#   (b) no session's turn-2 response contains another session's secret.
#
# Per the KHA-360 finding (2026-05-20), behavioural recall after restore
# may fail on hybrid models due to the attention-KV gap. That's expected
# here — this smoke is about allocation and isolation, not behaviour.
#
# Usage:
#   scripts/concurrency/concurrent-smoke.sh                    # default model path
#   scripts/concurrency/concurrent-smoke.sh --model <path>     # override
#   scripts/concurrency/concurrent-smoke.sh --port <n>         # default 30000
#   scripts/concurrency/concurrent-smoke.sh --n <count>        # default 4
#
# Exit codes:
#   0  — smoke passed (all sessions 200, no crosstalk)
#   2  — environment error (server unreachable)
#   3  — at least one session got a non-200 from save/restore
#   4  — crosstalk detected: a session's response contained another's secret

set -uo pipefail

MODEL="${MODEL:-/workspace/models/granite-4.0-h-tiny}"
PORT="${PORT:-30000}"
N="${N:-4}"
RUN_DIR="$(mktemp -d /tmp/kha-301-XXXXXX)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --port)  PORT="$2"; shift 2 ;;
        --n)     N="$2"; shift 2 ;;
        --help|-h)
            grep '^#' "$0" | head -30
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# --- Sanity: server reachable? ---
if ! curl -s -m 5 -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" | grep -q "^200$"; then
    log_fail "Server at localhost:$PORT not reachable. Launch it first per KHA-360 results doc."
    exit 2
fi

log_info "model=$MODEL  n=$N  run_dir=$RUN_DIR"

# --- One session: turn1 (establish), save, restore, turn2 (recall) ---
# Args: $1 = session index (1..N)
# Writes per-session output to $RUN_DIR/sess-$i-*.json
run_session() {
    local i="$1"
    local secret="ZULU-$i"
    local rid="kha-301-smoke-$(date +%s)-$$-$i"

    # Turn 1 — establish the per-session secret
    curl -s -m 30 "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Remember: the secret code is $secret. Acknowledge.\"}],
            \"max_tokens\": 20, \"temperature\": 0, \"rid\": \"$rid\"
        }" > "$RUN_DIR/sess-$i-turn1.json" 2>&1

    # Save snapshot
    local save_code
    save_code=$(curl -s -m 15 -o "$RUN_DIR/sess-$i-save.json" -w "%{http_code}" \
        "http://localhost:$PORT/save_snapshot" \
        -H "Content-Type: application/json" \
        -d "{\"rid\": \"$rid\"}")
    echo "$save_code" > "$RUN_DIR/sess-$i-save.code"

    # Restore snapshot
    local restore_code
    restore_code=$(curl -s -m 15 -o "$RUN_DIR/sess-$i-restore.json" -w "%{http_code}" \
        "http://localhost:$PORT/restore_snapshot" \
        -H "Content-Type: application/json" \
        -d "{\"rid\": \"$rid\"}")
    echo "$restore_code" > "$RUN_DIR/sess-$i-restore.code"

    # Turn 2 — recall (with fact absent from prompt)
    curl -s -m 30 "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"What was the secret code I told you earlier?\"}],
            \"max_tokens\": 40, \"temperature\": 0, \"rid\": \"$rid\"
        }" > "$RUN_DIR/sess-$i-turn2.json"
}

# --- Run all N sessions in parallel ---
log_info "spawning $N concurrent sessions ..."
for i in $(seq 1 "$N"); do
    run_session "$i" &
done
wait
log_info "all sessions complete"

# --- Aggregate results ---
EXIT=0

# (a) All save+restore must be 200
NON_200=0
for i in $(seq 1 "$N"); do
    local_save=$(cat "$RUN_DIR/sess-$i-save.code")
    local_restore=$(cat "$RUN_DIR/sess-$i-restore.code")
    if [[ "$local_save" != "200" || "$local_restore" != "200" ]]; then
        log_fail "session $i: save=$local_save restore=$local_restore"
        NON_200=1
    else
        log_pass "session $i: save=200 restore=200"
    fi
done
if [[ "$NON_200" == "1" ]]; then
    EXIT=3
fi

# (b) No crosstalk: session i's turn-2 response must not contain another session's secret
echo ""
log_info "crosstalk check across $N sessions:"
CROSSTALK=0
for i in $(seq 1 "$N"); do
    content=$(python -c "import sys,json; d=json.load(open('$RUN_DIR/sess-$i-turn2.json')); print(d['choices'][0]['message']['content'])" 2>/dev/null || echo "")
    for j in $(seq 1 "$N"); do
        if [[ "$i" == "$j" ]]; then continue; fi
        if echo "$content" | grep -q "ZULU-$j"; then
            log_fail "session $i response mentions ZULU-$j (another session's secret)"
            CROSSTALK=1
        fi
    done
    own=$(echo "$content" | grep -o "ZULU-$i" | head -1 || true)
    echo "  session $i (own=ZULU-$i): own-token-seen='$own' content='$(echo "$content" | head -c 120)'"
done

if [[ "$CROSSTALK" == "1" ]]; then
    log_fail "CROSSTALK DETECTED — multi-tenant isolation regression"
    EXIT=4
else
    log_pass "no crosstalk between sessions"
fi

echo ""
echo "Smoke artifacts under: $RUN_DIR"
exit "$EXIT"
