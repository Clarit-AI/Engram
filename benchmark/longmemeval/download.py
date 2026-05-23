"""Download the LongMemEval dataset from HuggingFace Hub.

Dataset
-------
- Repo:     xiaowu0162/LongMemEval
- License:  MIT
- Size:     ~2 GB
- Content:  500 questions spanning 5 memory-ability types:
            episodic, semantic, temporal, spatial, factual.
            Context windows range from 115K to 1.5M tokens.

Usage
-----
# Real download (requires huggingface-hub):
    python -m benchmark.longmemeval.download

# Dry-run (generates 5 synthetic questions, no network needed):
    python -m benchmark.longmemeval.download --dry-run

Output is written to:
    benchmark/longmemeval/data/longmemeval/

Files produced
--------------
longmemeval_s.jsonl   Single-session split (~115K tokens per context)
longmemeval_m.jsonl   Multi-session split (~500K–1.5M tokens per context)
test_dry_run.jsonl    5 synthetic questions (--dry-run mode)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent
_DATA_DIR = _HERE / "data" / "longmemeval"
_HF_REPO = "xiaowu0162/LongMemEval"

# ---------------------------------------------------------------------------
# Synthetic dry-run data
# ---------------------------------------------------------------------------

_SYNTHETIC_QUESTIONS = [
    {
        "question_id": "syn_001",
        "memory_type": "episodic",
        "memory_text": (
            "On Monday, Alice told Bob that she had visited Paris last summer. "
            "She described the Eiffel Tower as magnificent and mentioned eating "
            "a croissant at a small café near the Louvre. Bob said he had never "
            "been to Paris but wanted to visit someday. They agreed to plan a trip "
            "together next spring."
        ),
        "question": "What city did Alice visit last summer?",
        "answer": "Paris",
    },
    {
        "question_id": "syn_002",
        "memory_type": "factual",
        "memory_text": (
            "The capital of France is Paris. The city is home to approximately "
            "2.1 million people within its city limits and more than 12 million "
            "in the greater metropolitan area. Paris is known for the Eiffel Tower, "
            "the Louvre Museum, and Notre-Dame Cathedral."
        ),
        "question": "What is the approximate population of Paris within its city limits?",
        "answer": "2.1 million",
    },
    {
        "question_id": "syn_003",
        "memory_type": "temporal",
        "memory_text": (
            "The project kicked off on January 15, 2024. By February 28 the team "
            "had completed the first milestone: a working prototype. The second "
            "milestone, user testing, wrapped up on April 10. The final release "
            "was shipped on June 1, 2024."
        ),
        "question": "When was the final release shipped?",
        "answer": "June 1, 2024",
    },
    {
        "question_id": "syn_004",
        "memory_type": "spatial",
        "memory_text": (
            "The office building has four floors. The reception is on the ground "
            "floor (floor 1). Engineering occupies floor 2. Design and Product are "
            "on floor 3. The executive suite is on floor 4, the top floor."
        ),
        "question": "Which floor does the Engineering team occupy?",
        "answer": "Floor 2",
    },
    {
        "question_id": "syn_005",
        "memory_type": "semantic",
        "memory_text": (
            "Mamba is a state-space model (SSM) architecture that processes "
            "sequences in linear time. Unlike transformers, which use quadratic "
            "attention, Mamba maintains a hidden state that is updated recurrently. "
            "This makes it especially efficient for very long contexts."
        ),
        "question": "What is the time complexity of Mamba's sequence processing?",
        "answer": "Linear time",
    },
]


def _write_dry_run(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "test_dry_run.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for q in _SYNTHETIC_QUESTIONS:
            f.write(json.dumps(q) + "\n")
    print(f"Dry-run data written to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Real download via huggingface_hub
# ---------------------------------------------------------------------------

def _download_hf(output_dir: Path, token: Optional[str] = None) -> None:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for downloading LongMemEval.\n"
            "Install it with: pip install huggingface-hub\n"
            "Or use --dry-run for synthetic data."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover available files in the repo
    try:
        repo_files = list(list_repo_files(_HF_REPO, repo_type="dataset", token=token))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not list repo files: {exc}", file=sys.stderr)
        repo_files = [
            "longmemeval_s.jsonl",
            "longmemeval_m.jsonl",
        ]

    jsonl_files = [f for f in repo_files if f.endswith(".jsonl")]
    if not jsonl_files:
        jsonl_files = ["longmemeval_s.jsonl", "longmemeval_m.jsonl"]

    for filename in jsonl_files:
        print(f"Downloading {filename} …")
        try:
            local_path = hf_hub_download(
                repo_id=_HF_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=str(output_dir),
                token=token,
            )
            print(f"  Saved to: {local_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] Failed to download {filename}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download or generate LongMemEval benchmark data."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate 5 synthetic questions instead of downloading from HuggingFace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DATA_DIR,
        help=f"Directory to write dataset files (default: {_DATA_DIR}).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace access token (or set HF_TOKEN env var).",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        _write_dry_run(args.output_dir)
        return 0

    print(f"Downloading {_HF_REPO} (~2 GB) to {args.output_dir} …")
    print("License: MIT  |  500 questions  |  5 memory types  |  115K–1.5M tokens")
    _download_hf(args.output_dir, token=args.token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
