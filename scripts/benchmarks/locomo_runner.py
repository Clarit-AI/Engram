"""
LoCoMo benchmark runner for Engram stateful recall evaluation (KHA-255).

Modes:
  stateless  — baseline, each question is answered with full conversation context
               but no snapshot save/restore and no rid reuse across questions.
  engram     — exercises rid reuse + save/restore: conversation context is built
               incrementally via snapshot persistence, exercising the KHA-348 fix path.

Usage:
  python scripts/benchmarks/locomo_runner.py \
    --mode {stateless,engram} \
    --model <model-id> \
    --server-url http://localhost:30000 \
    --dataset-path data/locomo10.json \
    --conversations all \
    --output-dir s3://engram/benchmark-results/locomo/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Scoring helpers (mirrors official LoCoMo eval logic exactly)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    import re
    import string
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    num_common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _partial_f1(prediction: str, ground_truth: str) -> float:
    """Category 1: split answer into sub-phrases, average F1 across each."""
    parts = [p.strip() for p in ground_truth.split(",") if p.strip()]
    if not parts:
        return _token_f1(prediction, ground_truth)
    return sum(_token_f1(prediction, p) for p in parts) / len(parts)


def score_qa(prediction: str, qa: dict) -> float:
    """Return [0,1] score for a single QA item using the official LoCoMo rubric."""
    category = qa["category"]
    if category == 5:
        # adversarial: model should say it doesn't know
        lower = prediction.lower()
        return 1.0 if ("no information available" in lower or "not mentioned" in lower) else 0.0
    answer = qa.get("answer", "")
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category == 1:
        return _partial_f1(prediction, answer)
    # categories 2, 3, 4
    return _token_f1(prediction, answer)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(path: str, max_conversations: int | None = None) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(p) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array in dataset file")
    if max_conversations is not None:
        data = data[:max_conversations]
    return data


def flatten_conversation(conv: dict) -> list[dict]:
    """Return all turns in chronological session order as a flat list of dicts."""
    turns = []
    for n in range(1, 200):
        key = f"session_{n}"
        if key not in conv:
            break
        session_turns = conv[key]
        if not isinstance(session_turns, list):
            continue
        date = conv.get(f"{key}_date_time", "")
        for turn in session_turns:
            turns.append({
                "session": n,
                "date": date,
                "speaker": turn.get("speaker", ""),
                "dia_id": turn.get("dia_id", ""),
                "text": turn.get("text", ""),
                "blip_caption": turn.get("blip_caption", ""),
            })
    return turns


def build_context_text(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        line = f"{t['speaker']}: {t['text']}"
        if t.get("blip_caption"):
            line += f" [image: {t['blip_caption']}]"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

def chat_completion(
    server_url: str,
    model: str,
    messages: list[dict],
    rid: str | None = None,
    dry_run: bool = False,
    dry_run_answer: str = "dry-run-answer",
) -> tuple[str, float]:
    """Return (answer_text, latency_seconds)."""
    if dry_run:
        return dry_run_answer, 0.001

    import urllib.request

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0.0,
    }
    if rid is not None:
        payload["rid"] = rid

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    latency = time.perf_counter() - t0
    answer = body["choices"][0]["message"]["content"].strip()
    return answer, latency


def save_snapshot(server_url: str, rid: str, dry_run: bool = False) -> tuple[bool, float]:
    if dry_run:
        return True, 0.001
    import urllib.request
    payload = json.dumps({"rid": rid}).encode()
    req = urllib.request.Request(
        f"{server_url}/save_snapshot",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    latency = time.perf_counter() - t0
    return body.get("success", False), latency


def restore_snapshot(server_url: str, rid: str, dry_run: bool = False) -> tuple[bool, float]:
    if dry_run:
        return True, 0.001
    import urllib.request
    payload = json.dumps({"rid": rid}).encode()
    req = urllib.request.Request(
        f"{server_url}/restore_snapshot",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    latency = time.perf_counter() - t0
    return body.get("success", False), latency


# ---------------------------------------------------------------------------
# Runner modes
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a long conversation history. "
    "Answer questions about the conversation concisely and accurately. "
    "If the answer is not present in the conversation, say 'No information available'."
)

QA_PROMPT_TEMPLATE = "Based on the conversation above, answer the following question:\n{question}"


def run_stateless(
    sample: dict,
    server_url: str,
    model: str,
    dry_run: bool,
) -> list[dict]:
    """One fresh request per QA item. Baseline — no snapshot, no rid reuse."""
    turns = flatten_conversation(sample["conversation"])
    context = build_context_text(turns)
    results = []
    for qa in sample["qa"]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context + "\n\n" + QA_PROMPT_TEMPLATE.format(question=qa["question"])},
        ]
        answer, latency = chat_completion(server_url, model, messages, dry_run=dry_run)
        results.append({
            "dia_id_evidence": qa.get("evidence", []),
            "question": qa["question"],
            "ground_truth": qa.get("answer", qa.get("adversarial_answer", "")),
            "category": qa["category"],
            "prediction": answer,
            "score": score_qa(answer, qa),
            "latency_s": round(latency, 4),
            "save_latency_s": None,
            "restore_latency_s": None,
        })
    return results


def run_engram(
    sample: dict,
    server_url: str,
    model: str,
    dry_run: bool,
) -> list[dict]:
    """
    Engram mode — exercises rid reuse + save/restore (KHA-348 fix path).

    Protocol:
    1. Feed conversation turns incrementally to the model via a persistent rid.
    2. After each session's turns are fed, save snapshot.
    3. For each QA item, restore snapshot and ask the question with the same rid.
    4. Measure save/restore latency separately.
    """
    rid = f"locomo-{sample['sample_id']}-{uuid.uuid4().hex[:8]}"
    turns = flatten_conversation(sample["conversation"])

    # Build the conversation context incrementally
    context_so_far: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Feed all turns to establish context
    for turn in turns:
        line = f"{turn['speaker']}: {turn['text']}"
        if turn.get("blip_caption"):
            line += f" [image: {turn['blip_caption']}]"
        context_so_far.append({"role": "user", "content": line})
        # Fire-and-forget to build state; we only care about the final answer
        chat_completion(server_url, model, context_so_far, rid=rid, dry_run=dry_run,
                        dry_run_answer="[context feeding]")
        context_so_far.append({"role": "assistant", "content": "[context feeding]"})

    # Save snapshot after full context is loaded
    save_ok, save_latency = save_snapshot(server_url, rid, dry_run=dry_run)
    if not save_ok and not dry_run:
        raise RuntimeError(f"save_snapshot failed for rid={rid}")

    results = []
    for qa in sample["qa"]:
        restore_ok, restore_latency = restore_snapshot(server_url, rid, dry_run=dry_run)
        if not restore_ok and not dry_run:
            raise RuntimeError(f"restore_snapshot failed for rid={rid}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": QA_PROMPT_TEMPLATE.format(question=qa["question"])},
        ]
        answer, latency = chat_completion(server_url, model, messages, rid=rid, dry_run=dry_run)
        results.append({
            "dia_id_evidence": qa.get("evidence", []),
            "question": qa["question"],
            "ground_truth": qa.get("answer", qa.get("adversarial_answer", "")),
            "category": qa["category"],
            "prediction": answer,
            "score": score_qa(answer, qa),
            "latency_s": round(latency, 4),
            "save_latency_s": round(save_latency, 4),
            "restore_latency_s": round(restore_latency, 4),
        })

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(results_by_conv: dict[str, list[dict]], output_dir: str, mode: str, model: str) -> tuple[str, str]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"locomo_{mode}_{model.replace('/', '-')}_{timestamp}"

    all_results = []
    for conv_id, qa_results in results_by_conv.items():
        all_results.append({"conv_id": conv_id, "qa_results": qa_results})

    # Aggregate stats
    flat_qa = [r for v in results_by_conv.values() for r in v]
    by_cat: dict[int, list[float]] = defaultdict(list)
    for r in flat_qa:
        by_cat[r["category"]].append(r["score"])
    overall = sum(r["score"] for r in flat_qa) / len(flat_qa) if flat_qa else 0.0
    cat_means = {cat: sum(scores) / len(scores) for cat, scores in by_cat.items()}

    save_lats = [r["save_latency_s"] for r in flat_qa if r["save_latency_s"] is not None]
    restore_lats = [r["restore_latency_s"] for r in flat_qa if r["restore_latency_s"] is not None]

    summary = {
        "mode": mode,
        "model": model,
        "timestamp": timestamp,
        "total_questions": len(flat_qa),
        "overall_f1": round(overall, 4),
        "f1_by_category": {str(k): round(v, 4) for k, v in sorted(cat_means.items())},
        "mean_save_latency_s": round(sum(save_lats) / len(save_lats), 4) if save_lats else None,
        "mean_restore_latency_s": round(sum(restore_lats) / len(restore_lats), 4) if restore_lats else None,
        "conversations": all_results,
    }

    output_dir = output_dir.rstrip("/")
    json_path = f"{output_dir}/{stem}.json"
    csv_path = f"{output_dir}/{stem}_summary.csv"

    if output_dir.startswith("s3://"):
        import subprocess
        json_tmp = f"/tmp/{stem}.json"
        csv_tmp = f"/tmp/{stem}_summary.csv"
        with open(json_tmp, "w") as f:
            json.dump(summary, f, indent=2)
        subprocess.run(["aws", "s3", "cp", json_tmp, json_path], check=True)
        _write_csv(flat_qa, csv_tmp)
        subprocess.run(["aws", "s3", "cp", csv_tmp, csv_path], check=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        _write_csv(flat_qa, csv_path)

    return json_path, csv_path


def _write_csv(flat_qa: list[dict], path: str) -> None:
    fields = ["question", "category", "ground_truth", "prediction", "score",
              "latency_s", "save_latency_s", "restore_latency_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(flat_qa)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoCoMo benchmark runner for Engram (KHA-255)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["stateless", "engram"],
                   help="stateless=baseline, engram=snapshot save/restore")
    p.add_argument("--model", required=True,
                   help="Model ID passed to /v1/chat/completions")
    p.add_argument("--server-url", default="http://localhost:30000",
                   help="Base URL of the Engram/SGLang server")
    p.add_argument("--dataset-path", required=True,
                   help="Path to locomo10.json (local or s3://)")
    p.add_argument("--conversations", default="all",
                   help="Number of conversations to run, or 'all'")
    p.add_argument("--output-dir", required=True,
                   help="Output directory (local path or s3://bucket/prefix)")
    p.add_argument("--dry-run", action="store_true",
                   help="Mock server responses for unit testing without a live model")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    max_conv = None if args.conversations == "all" else int(args.conversations)
    dataset = load_dataset(args.dataset_path, max_conversations=max_conv)
    print(f"Loaded {len(dataset)} conversations from {args.dataset_path}", flush=True)

    run_fn = run_engram if args.mode == "engram" else run_stateless

    results_by_conv: dict[str, list[dict]] = {}
    total_start = time.perf_counter()

    for i, sample in enumerate(dataset):
        conv_id = sample["sample_id"]
        print(f"[{i+1}/{len(dataset)}] {conv_id} ...", flush=True)
        results_by_conv[conv_id] = run_fn(
            sample=sample,
            server_url=args.server_url,
            model=args.model,
            dry_run=args.dry_run,
        )

    wall_clock = time.perf_counter() - total_start
    print(f"Total wall-clock: {wall_clock:.1f}s", flush=True)

    json_path, csv_path = write_outputs(results_by_conv, args.output_dir, args.mode, args.model)
    print(f"Results: {json_path}")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    main()
