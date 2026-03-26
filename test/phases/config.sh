#!/usr/bin/env bash
# Central configuration for all SGLang-Mamba test phases.
# To test a different model: edit MODEL_PATH and MODEL_NAME here only.
# All phase documents source this file.

# =============================================================================
# MULTI-MODEL STRATEGY
# =============================================================================
# Testing priority:
#   1. granite-4.0-h-tiny  — primary (try first; full FP16 hybrid Mamba)
#   2. Nemotron-4B          — fallback if granite OOMs on V100 16GB
#   3. granite-q4          — quantized comparison pass (fp16 vs quantized Mamba)
#
# Granite Q4 server args (e.g., --dtype q4) are TBD — verify when testing.
# =============================================================================

# --- Primary model (granite-4.0-h-tiny) ---
export MODEL_PATH=${MODEL_PATH:-/home/jeanclawdai/models/granite-4.0-h-tiny}
export MODEL_NAME=${MODEL_NAME:-granite-4.0-h-tiny}

# --- Fallback model (Nemotron 4B FP16 — use if granite OOMs) ---
export NEMOTRON_MODEL_PATH=${NEMOTRON_MODEL_PATH:-/home/jeanclawdai/models/NVIDIA-Nemotron-3-Nano-4B-BF16}
export NEMOTRON_MODEL_NAME=${NEMOTRON_MODEL_NAME:-nemotron-4b}

# --- Quantized comparison model (Granite 4B Q4_K_M GGUF) ---
export GRANITE_Q4_MODEL_PATH=${GRANITE_Q4_MODEL_PATH:-/home/jeanclawdai/models/granite-4.0-h-tiny-gguf/granite-4.0-h-tiny-Q4_K_M.gguf}
export GRANITE_Q4_MODEL_NAME=${GRANITE_Q4_MODEL_NAME:-granite-4b-q4}

# --- Current default (start with primary; change to switch model) ---
# To use Nemotron: MODEL_PATH=$NEMOTRON_MODEL_PATH MODEL_NAME=$NEMOTRON_MODEL_NAME
# To use Q4 GGUF:  MODEL_PATH=$GRANITE_Q4_MODEL_PATH MODEL_NAME=$GRANITE_Q4_MODEL_NAME

export SERVER_PORT=30000
export SERVER_URL=http://localhost:${SERVER_PORT}
export SNAPSHOT_DIR=/tmp/mamba_snapshots
export RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"

mkdir -p "$RESULTS_DIR" "$SNAPSHOT_DIR"

echo "Model:   $MODEL_PATH"
echo "Results: $RESULTS_DIR"