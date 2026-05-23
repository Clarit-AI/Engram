"""Download or synthesise the GraphWalks dataset.

Dataset
-------
- HuggingFace repo : ``openai/graphwalks``
- Task             : Multi-hop graph-traversal QA
- License          : MIT (OpenAI GraphWalks)
- Estimated size   : ~50 MB (exact size confirmed on first download)
- Save path        : ``<benchmark_dir>/data/graphwalks/``

Usage
-----
# Download from HuggingFace (requires ``datasets`` or ``huggingface_hub``):
python -m benchmark.graphwalks.download

# Dry-run: generate 5 synthetic questions only (no network, no GPU):
python -m benchmark.graphwalks.download --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import string
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Root of the benchmark package.
_BENCHMARK_DIR = Path(__file__).parent
_DATA_DIR = _BENCHMARK_DIR / "data" / "graphwalks"

HF_REPO_ID = "openai/graphwalks"
DRY_RUN_PATH = _DATA_DIR / "test_dry_run.jsonl"


# ---------------------------------------------------------------------------
# Synthetic question generator
# ---------------------------------------------------------------------------


def generate_synthetic_question(
    num_nodes: int = 5,
    num_hops: int = 3,
    seed: int = 0,
) -> Dict:
    """Generate a synthetic path-graph question with a known answer.

    The graph is a simple directed path: A → B → C → … where node names are
    single uppercase letters.  The question asks "starting from node X,
    following the path, which node do you reach after N hops?"

    Parameters
    ----------
    num_nodes:
        Total number of nodes in the path graph.  Must be > num_hops.
    num_hops:
        Number of hops the question asks about.
    seed:
        Random seed (used to vary start node offset for different questions).

    Returns
    -------
    dict with keys: question_id, graph_text, question, answer, num_hops,
    num_nodes.
    """
    if num_nodes <= num_hops:
        raise ValueError(
            f"num_nodes ({num_nodes}) must be greater than num_hops ({num_hops})"
        )
    if num_nodes > 26:
        raise ValueError("num_nodes must be ≤ 26 for single-letter node names")

    rng = random.Random(seed)
    offset = rng.randint(0, num_nodes - num_hops - 1)

    # Build a path graph of exactly num_nodes nodes starting from the offset.
    all_letters = list(string.ascii_uppercase)
    nodes = all_letters[offset: offset + num_nodes]

    # Textual description
    edges_desc = " -> ".join(nodes)
    graph_text = (
        f"Graph description:\n"
        f"This graph is a directed path with {num_nodes} nodes.\n"
        f"Edges: {edges_desc}\n"
        f"Each node connects only to the next node in the sequence."
    )

    start_node = nodes[0]
    answer_node = nodes[num_hops]
    question = (
        f"Starting from node {start_node}, following the directed edges, "
        f"which node do you reach after exactly {num_hops} hop(s)?"
    )

    question_id = f"synthetic_{seed}_hops{num_hops}_nodes{num_nodes}"

    return {
        "question_id": question_id,
        "graph_text": graph_text,
        "question": question,
        "answer": answer_node,
        "num_hops": num_hops,
        "num_nodes": num_nodes,
    }


def generate_dry_run_dataset(
    n: int = 5,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """Generate *n* synthetic questions and write them to *output_path* (JSONL).

    Uses varying seeds and hop counts to produce a diverse mini-dataset.
    """
    output_path = output_path or DRY_RUN_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    hop_cycle = [1, 2, 3, 2, 1]
    for i in range(n):
        num_hops = hop_cycle[i % len(hop_cycle)]
        record = generate_synthetic_question(num_nodes=6, num_hops=num_hops, seed=i)
        records.append(record)

    with output_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    logger.info("Wrote %d synthetic questions to %s", n, output_path)
    return records


# ---------------------------------------------------------------------------
# HuggingFace download
# ---------------------------------------------------------------------------


def download_from_hf(output_dir: Optional[Path] = None) -> Path:
    """Download the GraphWalks dataset from HuggingFace.

    Tries ``datasets`` first (richer API), then falls back to
    ``huggingface_hub.snapshot_download`` for raw files.

    Returns the local directory containing the dataset files.
    """
    output_dir = output_dir or _DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try datasets library first.
    try:
        from datasets import load_dataset  # noqa: PLC0415

        logger.info("Downloading %s via `datasets` library…", HF_REPO_ID)
        ds = load_dataset(HF_REPO_ID, split="test")

        out_path = output_dir / "graphwalks_test.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in ds:
                fh.write(json.dumps(row) + "\n")

        logger.info("Saved %d rows to %s", len(ds), out_path)
        return output_dir

    except ImportError:
        logger.info(
            "`datasets` not available; falling back to huggingface_hub snapshot."
        )

    # Fallback: huggingface_hub.
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        local_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(output_dir / "hf_snapshot"),
        )
        logger.info("Snapshot downloaded to %s", local_dir)
        return Path(local_dir)

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to download {HF_REPO_ID} from HuggingFace. "
            "If you are offline, use --dry-run to generate synthetic data.\n"
            f"Underlying error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download or synthesise the GraphWalks dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate 5 synthetic questions instead of downloading from HF.",
    )
    p.add_argument(
        "--n-synthetic",
        type=int,
        default=5,
        help="Number of synthetic questions to generate in dry-run mode.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save downloaded data (default: benchmark/data/graphwalks/).",
    )
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None

    if args.dry_run:
        path = (out_dir / "test_dry_run.jsonl") if out_dir else DRY_RUN_PATH
        generate_dry_run_dataset(n=args.n_synthetic, output_path=path)
    else:
        download_from_hf(output_dir=out_dir)


if __name__ == "__main__":
    main()
