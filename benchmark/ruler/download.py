"""
RULER dataset provisioning.

RULER generates synthetic data on-the-fly — there is no fixed corpus to
download.  This module provides:

1. Offline generation via generate_dataset() — works with no network access.
2. Optional HuggingFace Hub metadata fetch (hsiehjackson/RULER) for config
   reference.

License: MIT (RULER scripts and generated data are MIT-licensed).
Approx. data size: Generated on-the-fly; typically <100 MB for 4 K–128 K
    context configs at 10 samples per (task, length).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from .tasks import (
    CONTEXT_LENGTHS,
    RULER_TASKS,
    TaskInstance,
    generate_synthetic_task,
)

logger = logging.getLogger(__name__)

_HF_REPO_ID = "hsiehjackson/RULER"
_DEFAULT_OUTPUT = Path(__file__).parent / "data" / "ruler" / "generated_tasks.jsonl"


# ---------------------------------------------------------------------------
# Offline generation
# ---------------------------------------------------------------------------


def generate_dataset(
    tasks: Optional[List[str]] = None,
    context_lengths: Optional[List[int]] = None,
    num_samples: int = 10,
    output_path: Optional[str | Path] = None,
) -> List[TaskInstance]:
    """Generate a RULER dataset fully offline.

    Args:
        tasks: List of RULER task names to generate.  Defaults to all 13.
        context_lengths: List of context-length buckets to generate.
            Defaults to all six standard lengths.
        num_samples: Number of samples per (task, context_length) pair.
        output_path: If given, write generated instances to this JSONL file.

    Returns:
        List of TaskInstance objects.
    """
    if tasks is None:
        tasks = RULER_TASKS
    if context_lengths is None:
        context_lengths = CONTEXT_LENGTHS

    invalid = [t for t in tasks if t not in RULER_TASKS]
    if invalid:
        raise ValueError(f"Unknown task names: {invalid}")

    instances: List[TaskInstance] = []
    total = len(tasks) * len(context_lengths) * num_samples
    logger.info(
        "Generating %d RULER instances (%d tasks × %d ctx_lengths × %d samples)…",
        total,
        len(tasks),
        len(context_lengths),
        num_samples,
    )

    for task_name in tasks:
        for ctx_len in context_lengths:
            for sample_idx in range(num_samples):
                instance = generate_synthetic_task(task_name, ctx_len, sample_idx)
                instances.append(instance)

    logger.info("Generated %d instances.", len(instances))

    if output_path is not None:
        _write_jsonl(instances, Path(output_path))
    elif output_path is None:
        # Write to default location
        _write_jsonl(instances, _DEFAULT_OUTPUT)

    return instances


def _instance_to_dict(instance: TaskInstance) -> dict:
    return {
        "task_id": instance.task_id,
        "task_name": instance.task_name,
        "context_length": instance.context_length,
        "context_text": instance.context_text,
        "question": instance.question,
        "answer": instance.answer,
    }


def _instance_from_dict(d: dict) -> TaskInstance:
    return TaskInstance(
        task_id=d.get("task_id"),
        task_name=d["task_name"],
        context_length=d["context_length"],
        context_text=d["context_text"],
        question=d["question"],
        answer=d["answer"],
    )


def _write_jsonl(instances: List[TaskInstance], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(_instance_to_dict(inst)) + "\n")
    logger.info("Dataset written to %s (%d lines)", path, len(instances))


def load_dataset(path: str | Path) -> List[TaskInstance]:
    """Load a previously generated dataset from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    instances: List[TaskInstance] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(_instance_from_dict(json.loads(line)))
    logger.info("Loaded %d instances from %s", len(instances), path)
    return instances


# ---------------------------------------------------------------------------
# Optional HuggingFace Hub metadata fetch
# ---------------------------------------------------------------------------


def fetch_hf_ruler_config(
    repo_id: str = _HF_REPO_ID,
    local_dir: Optional[str | Path] = None,
) -> Optional[Path]:
    """Attempt to fetch RULER config files from the HuggingFace Hub.

    This is informational only — RULER benchmark tasks are generated
    synthetically and do not require downloading any HF dataset.

    Args:
        repo_id: HuggingFace repository ID (default: hsiehjackson/RULER).
        local_dir: Local directory to cache downloaded files.

    Returns:
        Path to the downloaded directory, or None if the fetch failed.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed. Skipping HF config fetch."
        )
        return None

    if local_dir is None:
        local_dir = Path("/tmp/ruler_hf_config")

    try:
        logger.info("Fetching RULER config from HuggingFace Hub: %s", repo_id)
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            ignore_patterns=["*.parquet", "*.arrow"],  # skip large data files
        )
        logger.info("RULER config downloaded to: %s", path)
        return Path(path)
    except Exception as exc:
        logger.warning("Failed to fetch RULER config from HF Hub: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Dataset info
# ---------------------------------------------------------------------------

RULER_DATASET_INFO = {
    "name": "RULER",
    "source": "https://huggingface.co/datasets/hsiehjackson/RULER",
    "paper": "RULER: What's the Real Context Window Size of Your Long-Context Language Models?",
    "license": "MIT",
    "generation": "Synthetic (on-the-fly, no fixed download corpus)",
    "approx_size_mb": "<100 MB for 4K–128K configs at 10 samples per (task, context_length)",
    "tasks": RULER_TASKS,
    "context_lengths": CONTEXT_LENGTHS,
    "num_tasks": len(RULER_TASKS),
    "note": (
        "RULER generates data synthetically.  Use generate_dataset() for "
        "offline generation.  huggingface_hub is optional and only used to "
        "fetch configuration metadata."
    ),
}
