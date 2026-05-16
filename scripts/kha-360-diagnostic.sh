#!/usr/bin/env bash
# ENGRAM_DIAGNOSTIC — KHA-360 hybrid-class stateful recall diagnostic
#
# Disambiguates two confounds in post-registry recall failures on hybrid
# (Mamba SSM + attention KV) models:
#
#   (A) Prompt-surface-form / model-capability confound — the model cannot
#       recall facts in the recall-question surface form even when the fact
#       is in-context.
#   (B) Attention KV restoration gap — the snapshot system restores SSM
#       state but not attention KV; hybrid models need both.
#
# Phase 1 (baseline) must run first. If baseline fails, the chosen target
# model cannot demonstrate recall under the test methodology and (B) cannot
# be evaluated until methodology is redesigned. Per KHA-360 acceptance
# criterion 3.
#
# Bounded at 2-4 H100/H200 hours per KHA-360 estimated effort.
#
# Usage:
#   scripts/kha-360-diagnostic.sh                       # default target (Nemotron-3-Nano-Omni)
#   scripts/kha-360-diagnostic.sh --model <hf_id>       # override target
#   scripts/kha-360-diagnostic.sh --phase baseline      # phase 1 only
#   scripts/kha-360-diagnostic.sh --phase kv-probe      # phase 2 only (assumes baseline passed)
#   scripts/kha-360-diagnostic.sh --control             # also run pure-Mamba2 (Codestral) control
#
# Exit codes:
#   0  — diagnostic completed; see report at $REPORT_PATH for verdict
#   2  — environment error (missing GPU, model unreachable, server failed)
#   3  — baseline phase failed; KV-probe skipped; methodology redesign needed
#   4  — baseline passed but KV probe inconclusive (see report)

set -euo pipefail

# --- Configuration ---
TARGET_MODEL="${TARGET_MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
CONTROL_MODEL="${CONTROL_MODEL:-mistralai/Codestral-Mamba-7B-v0.1}"
PHASE="${PHASE:-all}"
RUN_CONTROL=false
PORT="${PORT:-30000}"
RUN_DIR="$(mktemp -d /tmp/kha-360-XXXXXX)"
REPORT_PATH="$RUN_DIR/report.md"

# Args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) TARGET_MODEL="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --control) RUN_CONTROL=true; shift ;;
        --help|-h)
            grep '^#' "$0" | head -40
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

# --- Colors / logging ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# --- Diagnostic prompts ---
#
# DESIGN NOTE: keep prompts identical across phases. The only variable is
# whether the fact arrives via in-context turn (baseline) or via restored
# snapshot (probe). If recall succeeds in baseline but fails in probe, the
# delta points at the KV restoration gap.
SECRET_PHRASE="the secret code is DELTA-7"
RECALL_QUESTION="What was the secret code I told you earlier?"
EXPECTED_TOKEN="DELTA-7"

# ============================================================================
# PHASE 1 — Baseline diagnostic (no snapshots)
# ============================================================================
# Establish the fact and ask the recall question in a SINGLE prompt, no role
# markers, no snapshot involvement. This isolates (A): can the model recall
# this fact in this surface form when the fact is unambiguously in-context?
#
# Expected pass: response contains "$EXPECTED_TOKEN" verbatim or paraphrased
# unambiguously (e.g. "delta-7", "Delta 7").
# Expected fail: refusal template, hallucinated code, or omission.
#
# Per KHA-360: "If baseline fails, the diagnostic methodology is on a model
# incapable of demonstrating recall and the whole investigation needs a
# different target."
phase_baseline() {
    local model="$1"
    log_info "[baseline] model=$model"

    local response
    response=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model\",
            \"messages\": [{
                \"role\": \"user\",
                \"content\": \"Earlier I told you: $SECRET_PHRASE. $RECALL_QUESTION\"
            }],
            \"max_tokens\": 40,
            \"temperature\": 0
        }")

    local content
    content=$(echo "$response" | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")

    echo "[baseline:$model] response: $content" | tee -a "$REPORT_PATH"

    # Verdict: case-insensitive match on the canonical token
    if echo "$content" | grep -qi "DELTA[- ]\?7"; then
        log_pass "Baseline recall succeeded for $model"
        echo "BASELINE_PASS=1" >> "$RUN_DIR/state.env"
        return 0
    else
        log_fail "Baseline recall FAILED for $model — surface-form confound dominates; KV probe inconclusive"
        echo "BASELINE_PASS=0" >> "$RUN_DIR/state.env"
        return 3
    fi
}

# ============================================================================
# PHASE 2 — KV-restoration probe (snapshot save/restore + recall in NEW turn)
# ============================================================================
# Only run if baseline passed for this model. Establish fact via turn 1, save
# snapshot, restore snapshot, ask recall question in a SEPARATE turn (no fact
# in current prompt). For pure-Mamba2 models the SSM state alone should be
# sufficient (control case). For hybrid models the SSM state is restored but
# attention KV is not — if recall fails on hybrid but passes on control, (B)
# is confirmed.
#
# Expected pass (control): response contains "$EXPECTED_TOKEN".
# Expected fail (hybrid): refusal / hallucination → confirms KV gap.
# Anomalies (hybrid passes, or control fails) → see report, escalate.
phase_kv_probe() {
    local model="$1"
    local label="$2"  # "target" or "control"
    log_info "[kv-probe:$label] model=$model"

    local rid="kha-360-$label-$(date +%s)-$$"

    # Turn 1 — establish the fact (no recall question yet)
    curl -s "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Remember this: $SECRET_PHRASE. Acknowledge.\"}],
            \"max_tokens\": 30, \"temperature\": 0, \"rid\": \"$rid\"
        }" > "$RUN_DIR/$label-establish.json"

    # Save snapshot
    curl -s "http://localhost:$PORT/save_snapshot" \
        -H "Content-Type: application/json" \
        -d "{\"rid\": \"$rid\"}" > "$RUN_DIR/$label-save.json"

    # Restore snapshot
    curl -s "http://localhost:$PORT/restore_snapshot" \
        -H "Content-Type: application/json" \
        -d "{\"rid\": \"$rid\"}" > "$RUN_DIR/$label-restore.json"

    # Turn 2 — ask recall question only (fact NOT in current prompt)
    local response
    response=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$RECALL_QUESTION\"}],
            \"max_tokens\": 40, \"temperature\": 0, \"rid\": \"$rid\"
        }")

    local content
    content=$(echo "$response" | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
    echo "[kv-probe:$label:$model] response: $content" | tee -a "$REPORT_PATH"

    if echo "$content" | grep -qi "DELTA[- ]\?7"; then
        log_pass "KV probe recall succeeded for $label ($model)"
        echo "KV_${label}_PASS=1" >> "$RUN_DIR/state.env"
    else
        log_fail "KV probe recall FAILED for $label ($model)"
        echo "KV_${label}_PASS=0" >> "$RUN_DIR/state.env"
    fi
}

# ============================================================================
# Report generation
# ============================================================================
# Interpretation table (encoded in report for the GPU operator to fill in):
#
#   baseline pass | control pass | target pass | verdict
#   --------------+--------------+-------------+----------------------------------
#         0       |      *       |     *       | exit 3 — methodology redesign
#         1       |      1       |     1       | SSM-only restore is sufficient on
#                 |              |             | hybrid; original 4/5 PASS was
#                 |              |             | surface-form noise. Close KHA-360.
#         1       |      1       |     0       | KV gap CONFIRMED. File impl ticket
#                 |              |             | for attention-KV snapshot support.
#         1       |      0       |     *       | Control failed — anomalous;
#                 |              |             | exit 4. Investigate snapshot path
#                 |              |             | before drawing hybrid conclusion.
write_report_header() {
    cat > "$REPORT_PATH" <<EOF
# KHA-360 Diagnostic Report — $(date -u +%Y-%m-%dT%H:%M:%SZ)

Target model: $TARGET_MODEL
Control model: $CONTROL_MODEL (run=$RUN_CONTROL)
Run dir: $RUN_DIR

## Verdict matrix

| baseline | control kv | target kv | verdict |
|----------|------------|-----------|---------|
| FAIL     | n/a        | n/a       | Methodology redesign — model cannot recall in this surface form |
| PASS     | PASS       | PASS      | SSM-only restore sufficient on hybrid — original failure was surface form |
| PASS     | PASS       | FAIL      | **KV restoration gap CONFIRMED** — file attention-KV snapshot impl ticket |
| PASS     | FAIL       | any       | Control regression — investigate snapshot path before hybrid conclusion |

## Raw observations

EOF
}

# ============================================================================
# Main
# ============================================================================
write_report_header
echo "BASELINE_PASS=" > "$RUN_DIR/state.env"

case "$PHASE" in
    baseline)
        phase_baseline "$TARGET_MODEL" || true
        $RUN_CONTROL && phase_baseline "$CONTROL_MODEL" || true
        ;;
    kv-probe)
        phase_kv_probe "$TARGET_MODEL" "target"
        $RUN_CONTROL && phase_kv_probe "$CONTROL_MODEL" "control"
        ;;
    all|*)
        if ! phase_baseline "$TARGET_MODEL"; then
            log_fail "Baseline failed on target — skipping KV probe (per KHA-360 design)"
            echo -e "\n## Result\n\nBaseline failed. KV probe skipped. Methodology redesign needed before further work." >> "$REPORT_PATH"
            cat "$REPORT_PATH"
            exit 3
        fi
        phase_kv_probe "$TARGET_MODEL" "target"
        if $RUN_CONTROL; then
            if phase_baseline "$CONTROL_MODEL"; then
                phase_kv_probe "$CONTROL_MODEL" "control"
            fi
        fi
        ;;
esac

echo -e "\n## State\n" >> "$REPORT_PATH"
cat "$RUN_DIR/state.env" >> "$REPORT_PATH"
echo -e "\nReport written to: $REPORT_PATH"
cat "$REPORT_PATH"
