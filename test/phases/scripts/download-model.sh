#!/bin/bash
# Quick model download for testing - only essential files

MODEL_ID="${1:-nvidia/Nemotron-Cascade-2-30B-A3B}"
OUTPUT_DIR="${2:-/workspace/models/Nemotron-Cascade-2-30B-A3B}"

echo "=== Quick Model Download ==="
echo "Model: $MODEL_ID"
echo "Output: $OUTPUT_DIR"
echo ""
echo "This downloads only essential files (not full model):"
echo "  - config.json"
echo "  - model safetensors (sharded)"
echo "  - tokenizer files"
echo ""

mkdir -p "$OUTPUT_DIR"

# Install huggingface-hub if needed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    pip install -q huggingface_hub
fi

# Export variables explicitly so Python can read them
export MODEL_ID
export OUTPUT_DIR

python -c "
import os
import sys

model_id = os.environ.get('MODEL_ID')
output_dir = os.environ.get('OUTPUT_DIR')

print(f'Downloading {model_id} to {output_dir}...')
print('This may take 10-30 minutes for a 30B model...')

# Import after setting env vars
from huggingface_hub import snapshot_download

# Download only essential files (exclude large trainer checkpoints, etc.)
allow_patterns = [
    '*.json',
    '*.txt',
    '*.md',
    'model-*.safetensors',
    'tokenizer*',
    'special_tokens_map.json',
    'tokenizer_config.json',
    'vocab.json',
    'merges.txt',
]

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    print(f'\n✅ Download complete: {output_dir}')
    print(f'\nTo use with SGLang:')
    print(f'  python -m sglang.launch_server --model-path {output_dir}')
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"
