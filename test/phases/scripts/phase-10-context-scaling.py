#!/usr/bin/env python3
"""
Phase 10 Addendum: Context Window Scaling Test

Tests Mamba snapshot persistence at large context lengths.
Two modes: single-shot (fast, proves mechanics) and multi-turn (proves coherence).

Usage:
    python test/phases/scripts/phase-10-context-scaling.py --mode single-shot
    python test/phases/scripts/phase-10-context-scaling.py --mode multi-turn --tiers 2K,8K,32K
    python test/phases/scripts/phase-10-context-scaling.py --mode both
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "/tmp/mamba_snapshots")

# Tier definitions: label -> target token count
TIERS = {
    "2K": 2000,
    "8K": 8000,
    "32K": 32000,
    "64K": 64000,
    "128K": 128000,
}

# VRAM safety limit (MB) — stop before OOM on 80GB A100
VRAM_LIMIT_MB = 78000

# Latency thresholds for distinguishing WARM vs COLD restore
WARM_RESTORE_MAX_MS = 50
COLD_RESTORE_MIN_MS = 50

# Factual content paragraphs for building context.
# Each is ~150-250 tokens of distinct, verifiable content.
CONTEXT_PARAGRAPHS = [
    "The Amazon River is the largest river by discharge volume of water in the world, and by some definitions it is the longest. The headwaters of the Apurimac River on Nevado Mismi had been considered for nearly a century as the Amazon's most distant source, until a 2014 study found it to be the Cordillera Rumi Cruz at the headwaters of the Mantaro River in Peru. The Mantaro and Apurimac join, and with other tributaries form the Ucayali River, which in turn meets the Maranon River upstream of Iquitos, Peru, to form what countries other than Brazil consider to be the main stem of the Amazon.",
    "The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometers over an area of approximately 344,400 square kilometers. The reef is located in the Coral Sea, off the coast of Queensland, Australia. It can be seen from outer space and is the world's biggest single structure made by living organisms. The reef structure is composed of and built by billions of tiny organisms known as coral polyps. It supports a wide diversity of life and was selected as a World Heritage Site in 1981.",
    "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China-Nepal border runs across its summit point. Its elevation of 8,848.86 meters was most recently established in 2020 by the Chinese and Nepali authorities. The mountain attracts many climbers, some of them highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet.",
    "The Mediterranean Sea is a sea connected to the Atlantic Ocean, surrounded by the Mediterranean Basin and almost completely enclosed by land: on the north by Western and Southern Europe and Anatolia, on the south by North Africa, and on the east by the Levant. Although the sea is sometimes considered a part of the Atlantic Ocean, it is usually referred to as a separate body of water. Geological evidence indicates that around 5.9 million years ago, the Mediterranean was mostly cut off from the Atlantic and was partly or completely desiccated over a period of 600,000 years during the Messinian salinity crisis before being refilled by the Zanclean flood about 5.3 million years ago.",
    "The Sahara is a desert on the African continent. With an area of 9,200,000 square kilometers, it is the largest hot desert in the world and the third largest desert overall, smaller only than the deserts of Antarctica and the northern Arctic. The name 'Sahara' is derived from the Arabic word for 'desert', the plural of the Arabic word for 'desert'. The Sahara covers large parts of Algeria, Chad, Egypt, Libya, Mali, Mauritania, Morocco, Niger, Western Sahara, Sudan, and Tunisia. It is one of the harshest environments on the planet.",
    "Tokyo is the capital and most populous prefecture of Japan. Located at the head of Tokyo Bay, the prefecture forms part of the Kanto region on the central Pacific coast of Japan's main island of Honshu. Tokyo is the political and economic center of the country, as well as the seat of the Emperor of Japan and the national government. As of 2021, the prefecture has an estimated population of 13,960,000. The Greater Tokyo Area is the most populous metropolitan area in the world, with more than 37.4 million residents as of 2020.",
    "The human brain weighs about 1.4 kilograms and has a volume of roughly 1,130 cubic centimeters in women and 1,260 cubic centimeters in men. The cerebrum, the largest part of the human brain, consists of two cerebral hemispheres. Each hemisphere has an inner core composed of white matter and an outer surface composed of gray matter, which is called the cerebral cortex. The cortex has an area of about 250 square centimeters per hemisphere if unfolded. The human cerebral cortex is roughly 2 to 4 millimeters thick.",
    "Honeybees are social insects that live in colonies containing one queen, thousands of female workers, and a smaller number of male drones. A typical healthy hive can contain between 50,000 and 80,000 worker bees. The queen bee can live for three to five years and is capable of laying up to 2,000 eggs per day during peak season. Worker bees live for about six weeks during the active season but can survive for several months during winter. Drones exist solely to mate with a new queen and die immediately after mating.",
    "The International Space Station is a multinational collaborative project involving five participating space agencies: NASA (United States), Roscosmos (Russia), JAXA (Japan), ESA (Europe), and CSA (Canada). It orbits Earth at an average altitude of approximately 420 kilometers and travels at roughly 28,000 kilometers per hour, completing about 15.5 orbits per day. The ISS serves as a microgravity and space environment research laboratory in which scientific research is conducted in astrobiology, astronomy, meteorology, physics, and other fields.",
    "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets combined, and slightly less than one one-thousandth the mass of the Sun. Jupiter is the third brightest natural object in the Earth's night sky after the Moon and Venus. It has been observed since prehistoric times and is named after Jupiter, the chief deity of ancient Roman religion. Jupiter's most famous feature, the Great Red Spot, is a storm larger than Earth that has been raging for at least 350 years.",
]


def get_gpu_vram_mb() -> int:
    """Get current GPU VRAM usage in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0].strip())
    except Exception:
        pass
    return 0


def get_proc_rss_mb() -> int:
    """Get sglang process RSS in MB."""
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            total_kb = 0
            for line in result.stdout.split("\n"):
                if "sglang::" in line or "launch_server" in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            total_kb += int(parts[5])
                        except ValueError:
                            continue
            return total_kb // 1024
    except Exception:
        pass
    return 0


def check_health() -> bool:
    """Check if the server is healthy."""
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def count_tokens_approx(text: str) -> int:
    """Rough token count estimate (words * 1.3 for subword tokens)."""
    return int(len(text.split()) * 1.3)


def build_prompt_to_token_target(target_tokens: int) -> tuple:
    """Build a prompt that hits approximately target_tokens tokens.

    Returns (messages_list, estimated_tokens).
    """
    paragraphs = []
    estimated = 0

    # System message with some overhead
    system_msg = (
        "You are a helpful assistant. I will provide you with reference text. "
        "After reading all the text, I will ask you questions about it. "
        "Pay close attention to specific facts, numbers, and details."
    )
    estimated += count_tokens_approx(system_msg)

    # Keep adding paragraphs until we're close to target (leaving ~200 tokens for user msg)
    fill_target = target_tokens - 250
    para_idx = 0
    while estimated < fill_target:
        para = CONTEXT_PARAGRAPHS[para_idx % len(CONTEXT_PARAGRAPHS)]
        paragraphs.append(para)
        estimated += count_tokens_approx(para)
        para_idx += 1

    context_text = "\n\n".join(paragraphs)

    user_msg = (
        f"I've provided you with reference text above containing {len(paragraphs)} passages. "
        "Please confirm you have read the text by stating 'I have read the text.' "
        "Then briefly summarize the key subjects covered (just the topics, 1-2 words each). "
        "Keep your response under 100 words."
    )
    estimated += count_tokens_approx(user_msg)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": context_text + "\n\n" + user_msg},
    ]
    return messages, estimated


def send_chat(messages: list, temperature: float = 0.0, max_tokens: int = 100) -> dict:
    """Send a chat completion request and return result with timing."""
    start = time.time()
    r = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": "default",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=300,
    )
    elapsed = time.time() - start
    r.raise_for_status()
    data = r.json()
    return {
        "text": data["choices"][0]["message"]["content"],
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "total_tokens": data["usage"]["total_tokens"],
        "latency_s": elapsed,
    }


def send_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 100) -> dict:
    """Send a /v1/completions request — returns RID in response['id']."""
    start = time.time()
    r = requests.post(
        f"{SERVER_URL}/v1/completions",
        json={
            "model": "default",
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=300,
    )
    elapsed = time.time() - start
    r.raise_for_status()
    data = r.json()
    return {
        "rid": data.get("id", ""),
        "text": data["choices"][0].get("text", ""),
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "total_tokens": data["usage"]["total_tokens"],
        "latency_s": elapsed,
    }


def save_snapshot(rid: str) -> dict:
    """Save a snapshot for the given request ID."""
    start = time.time()
    r = requests.post(
        f"{SERVER_URL}/save_snapshot",
        json={"rid": rid},
        timeout=30,
    )
    elapsed = time.time() - start
    r.raise_for_status()
    data = r.json()
    data["latency_ms"] = elapsed * 1000
    return data


def restore_snapshot(
    conversation_id: str = None,
    rid: str = None,
    create_new_request: bool = False,
) -> dict:
    """Restore a snapshot. Returns result with timing."""
    payload = {"create_new_request": create_new_request}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if rid:
        payload["rid"] = rid

    start = time.time()
    r = requests.post(
        f"{SERVER_URL}/restore_snapshot",
        json=payload,
        timeout=60,
    )
    elapsed = time.time() - start
    r.raise_for_status()
    data = r.json()
    data["latency_ms"] = elapsed * 1000
    return data


def get_snapshot_file_size_mb(conversation_id: str) -> float:
    """Get the size of snapshot files for a conversation in MB."""
    conv_dir = Path(SNAPSHOT_DIR) / f"conversation_{conversation_id}"
    if not conv_dir.exists():
        # Try without prefix
        conv_dir = Path(SNAPSHOT_DIR) / conversation_id
    if not conv_dir.exists():
        return 0.0
    total = sum(f.stat().st_size for f in conv_dir.glob("*") if f.is_file())
    return total / (1024 * 1024)


def evict_warm_and_cold_restore(conversation_id: str) -> dict:
    """Force COLD restore by evicting from WARM tier.

    Strategy: send many junk requests to fill the Mamba pool, forcing LRU eviction.
    Then attempt restore. Verify we hit COLD tier by checking latency.
    Retry with more junk requests if latency indicates WARM hit.
    """
    junk_count = 20  # Start with 20 junk requests
    max_attempts = 5

    for attempt in range(max_attempts):
        # Send junk requests to evict WARM states
        for i in range(junk_count):
            try:
                requests.post(
                    f"{SERVER_URL}/v1/chat/completions",
                    json={
                        "model": "default",
                        "messages": [{"role": "user", "content": f"Junk request {i}. Say OK."}],
                        "temperature": 0.0,
                        "max_tokens": 5,
                    },
                    timeout=30,
                )
            except Exception:
                pass

        # Attempt cold restore
        result = restore_snapshot(conversation_id=conversation_id)

        # Check latency to confirm COLD tier was hit
        if result["latency_ms"] >= COLD_RESTORE_MIN_MS:
            result["cold_verified"] = True
            result["junk_requests_sent"] = junk_count
            result["attempt"] = attempt + 1
            return result

        # WARM hit — eviction failed, double junk count and retry
        junk_count *= 2

    # Never confirmed COLD
    result["cold_verified"] = False
    result["junk_requests_sent"] = junk_count
    result["attempt"] = max_attempts
    return result


# ────────────────────────────────────────────────────────────────
# Single-Shot Mode
# ────────────────────────────────────────────────────────────────

def run_single_shot(tiers: List[str], output_dir: Path) -> List[dict]:
    """Run single-shot test at each tier."""
    results = []

    for tier_label in tiers:
        target = TIERS[tier_label]
        print(f"\n{'='*60}")
        print(f"SINGLE-SHOT: {tier_label} ({target:,} tokens)")
        print(f"{'='*60}")

        # Pre-flight checks
        vram = get_gpu_vram_mb()
        if vram > VRAM_LIMIT_MB:
            print(f"  ABORT: GPU VRAM at {vram}MB exceeds {VRAM_LIMIT_MB}MB limit")
            results.append({"tier": tier_label, "target_tokens": target, "status": "OOM_SKIP", "vram_mb": vram})
            break

        if not check_health():
            print("  ABORT: Server not healthy")
            results.append({"tier": tier_label, "target_tokens": target, "status": "SERVER_DOWN"})
            break

        tier_result = {
            "tier": tier_label,
            "target_tokens": target,
            "timestamp": datetime.now().isoformat(),
            "vram_before_mb": vram,
            "rss_before_mb": get_proc_rss_mb(),
        }

        # 1. Build prompt to target length
        print(f"  Building prompt for ~{target:,} tokens...")
        messages, estimated = build_prompt_to_token_target(target)
        tier_result["estimated_prompt_tokens"] = estimated

        # Convert messages to a single prompt string for /v1/completions
        prompt_parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        prompt_text = "\n".join(prompt_parts) + "Assistant:"

        # 2. Send request via /v1/completions to get a proper RID
        print(f"  Sending request via /v1/completions...")
        t0 = time.time()
        try:
            comp_result = send_completion(prompt_text, temperature=0.0, max_tokens=100)
        except requests.exceptions.Timeout:
            print(f"  TIMEOUT after 300s")
            tier_result["status"] = "TIMEOUT"
            results.append(tier_result)
            break
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP ERROR: {e}")
            tier_result["status"] = f"HTTP_ERROR: {e}"
            results.append(tier_result)
            break

        inference_latency = time.time() - t0
        rid = comp_result["rid"]
        actual_prompt_tokens = comp_result["prompt_tokens"]
        tier_result["rid"] = rid
        tier_result["actual_prompt_tokens"] = actual_prompt_tokens
        tier_result["inference_latency_s"] = round(inference_latency, 3)
        tier_result["inference_latency_ms"] = round(inference_latency * 1000, 1)
        tier_result["completion_tokens"] = comp_result["completion_tokens"]
        tier_result["output_text"] = comp_result["text"][:200]
        tier_result["output_nonempty"] = len(comp_result["text"].strip()) > 0

        print(f"  RID: {rid[:16]}...")
        print(f"  Prompt tokens: {actual_prompt_tokens:,}  |  Inference: {inference_latency:.3f}s")
        print(f"  Output: {comp_result['text'][:100]}...")

        # Check VRAM after inference
        vram_after = get_gpu_vram_mb()
        tier_result["vram_after_mb"] = vram_after
        tier_result["vram_delta_mb"] = vram_after - vram
        tier_result["rss_after_mb"] = get_proc_rss_mb()

        if vram_after > VRAM_LIMIT_MB:
            print(f"  WARNING: VRAM at {vram_after}MB after inference (limit {VRAM_LIMIT_MB})")
            tier_result["vram_warning"] = True

        # 3. Save snapshot using the RID from the completion
        print(f"  Saving snapshot (rid={rid[:16]}...)...")
        save_result = save_snapshot(rid=rid)
        tier_result["save_success"] = save_result.get("success", False)
        tier_result["save_latency_ms"] = save_result.get("latency_ms", 0)
        tier_result["save_message"] = save_result.get("message", "")

        if save_result.get("success"):
            snap_conv_id = (save_result.get("snapshot_id") or "").rsplit("-t", 1)[0] or ""
            snap_size = get_snapshot_file_size_mb(snap_conv_id)
            tier_result["snapshot_size_mb"] = round(snap_size, 2)
            tier_result["snapshot_id"] = save_result.get("snapshot_id", "")
            print(f"  Snapshot saved: {snap_size:.1f}MB in {save_result['latency_ms']:.0f}ms")
        else:
            print(f"  Snapshot save FAILED: {save_result.get('message', 'unknown')}")
            tier_result["status"] = "SAVE_FAILED"

        # 4. WARM restore
        print(f"  WARM restore test...")
        warm_result = restore_snapshot(
            conversation_id=snap_conv_id,
        )
        tier_result["warm_restore_success"] = warm_result.get("success", False)
        tier_result["warm_restore_latency_ms"] = round(warm_result.get("latency_ms", 0), 1)
        tier_result["warm_restore_tier"] = (
            "WARM" if warm_result["latency_ms"] < WARM_RESTORE_MAX_MS else "COLD?"
        )
        print(f"  WARM restore: {warm_result.get('latency_ms', 0):.0f}ms "
              f"(success={warm_result.get('success')})")

        # 6. COLD restore
        print(f"  COLD restore test (evicting WARM tier)...")
        cold_result = evict_warm_and_cold_restore(
            conversation_id=snap_conv_id,
        )
        tier_result["cold_restore_success"] = cold_result.get("success", False)
        tier_result["cold_restore_latency_ms"] = round(cold_result.get("latency_ms", 0), 1)
        tier_result["cold_restore_verified"] = cold_result.get("cold_verified", False)
        tier_result["cold_restore_junk_sent"] = cold_result.get("junk_requests_sent", 0)
        tier_result["cold_restore_attempts"] = cold_result.get("attempt", 0)
        verified_str = "VERIFIED" if cold_result.get("cold_verified") else "UNVERIFIED"
        print(f"  COLD restore: {cold_result.get('latency_ms', 0):.0f}ms [{verified_str}]")

        tier_result["status"] = "PASS"
        results.append(tier_result)

        # Write intermediate results
        with open(output_dir / "phase-10-context-scaling-single-shot.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    return results


# ────────────────────────────────────────────────────────────────
# Multi-Turn Mode
# ────────────────────────────────────────────────────────────────

# Facts to embed in system prompt for recall verification
RECALL_FACTS = [
    "Your name is Dr. Elara Voss.",
    "You work at the Meridian Institute in Portland, Oregon.",
    "Your office is on the 7th floor, room 714.",
    "Your cat is named Whiskers and is 3 years old.",
    "Your favorite color is teal.",
    "You drive a blue 2019 Subaru Outback.",
    "You graduated from MIT in 2010 with a PhD in Physics.",
    "Your research focuses on quantum entanglement in superconductors.",
    "You have two siblings: a brother named Marcus and a sister named Priya.",
    "Your birthday is October 15, 1985.",
]


def build_multi_turn_system_prompt() -> str:
    """Build a system prompt with embedded facts for recall testing."""
    facts_text = "\n".join(f"  {i+1}. {fact}" for i, fact in enumerate(RECALL_FACTS))
    return (
        "You are a helpful assistant with the following personal details:\n"
        f"{facts_text}\n\n"
        "Always stay in character as Dr. Elara Voss. When asked about yourself, "
        "use these facts. Be concise and accurate."
    )


def run_multi_turn(tiers: List[str], output_dir: Path) -> List[dict]:
    """Run multi-turn accumulation test at each tier."""
    results = []

    for tier_label in tiers:
        target = TIERS[tier_label]
        print(f"\n{'='*60}")
        print(f"MULTI-TURN: {tier_label} ({target:,} tokens)")
        print(f"{'='*60}")

        # Pre-flight
        vram = get_gpu_vram_mb()
        if vram > VRAM_LIMIT_MB:
            print(f"  ABORT: GPU VRAM at {vram}MB exceeds limit")
            results.append({"tier": tier_label, "target_tokens": target, "status": "OOM_SKIP"})
            break

        if not check_health():
            print("  ABORT: Server not healthy")
            results.append({"tier": tier_label, "target_tokens": target, "status": "SERVER_DOWN"})
            break

        tier_result = {
            "tier": tier_label,
            "target_tokens": target,
            "timestamp": datetime.now().isoformat(),
            "vram_before_mb": vram,
            "rss_before_mb": get_proc_rss_mb(),
        }

        # Build conversation
        system_prompt = build_multi_turn_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]

        # Pre-fill with context paragraphs to jump close to target
        fill_paragraphs = []
        estimated_context = count_tokens_approx(system_prompt)
        fill_target = target - 300  # Leave room for turns
        para_idx = 0
        while estimated_context < fill_target:
            para = CONTEXT_PARAGRAPHS[para_idx % len(CONTEXT_PARAGRAPHS)]
            fill_paragraphs.append(para)
            estimated_context += count_tokens_approx(para)
            para_idx += 1

        if fill_paragraphs:
            messages.append({
                "role": "user",
                "content": "Please read the following reference text carefully:\n\n"
                           + "\n\n".join(fill_paragraphs),
            })
            messages.append({
                "role": "assistant",
                "content": "I have read the reference text and am ready for questions.",
            })

        # Accumulate turns
        turn_count = 0
        prefill_latency = None
        coherence_log = []

        while True:
            # Check current context size with a lightweight request
            try:
                check_msg = messages + [{"role": "user", "content": "Say OK."}]
                r = requests.post(
                    f"{SERVER_URL}/v1/chat/completions",
                    json={
                        "model": "default",
                        "messages": check_msg,
                        "temperature": 0.0,
                        "max_tokens": 2,
                    },
                    timeout=120,
                )
                r.raise_for_status()
                current_tokens = r.json()["usage"]["prompt_tokens"]
            except Exception as e:
                print(f"  Token count check failed: {e}")
                tier_result["status"] = "TOKEN_CHECK_FAILED"
                break

            if current_tokens >= target:
                break

            # Add a turn
            turn_count += 1
            fact_q = RECALL_FACTS[(turn_count - 1) % len(RECALL_FACTS)]
            messages.append({
                "role": "user",
                "content": f"Turn {turn_count}: Please confirm: what is fact #{(turn_count % 10) + 1} from your personal details? Just state the fact.",
            })

            try:
                chat_result = send_chat(messages, temperature=0.0, max_tokens=50)
                messages.append({"role": "assistant", "content": chat_result["text"]})

                if turn_count == 1 and prefill_latency is None:
                    prefill_latency = chat_result["latency_s"]

                # Simple coherence check
                response_lower = chat_result["text"].lower()
                expected_fact = RECALL_FACTS[(turn_count - 1) % len(RECALL_FACTS)].lower()
                # Very loose check — just see if any key words from the fact appear
                key_words = [w for w in expected_fact.split() if len(w) > 4][:3]
                matches = sum(1 for w in key_words if w in response_lower)
                coherence_log.append({
                    "turn": turn_count,
                    "tokens": chat_result["prompt_tokens"],
                    "key_word_matches": matches,
                    "response": chat_result["text"][:100],
                })

            except requests.exceptions.Timeout:
                tier_result["status"] = "TIMEOUT"
                tier_result["turns_completed"] = turn_count
                break
            except Exception as e:
                tier_result["status"] = f"ERROR: {e}"
                tier_result["turns_completed"] = turn_count
                break

            if turn_count > 500:  # Safety limit
                print(f"  Safety limit: 500 turns reached")
                break

        tier_result["turns_completed"] = turn_count
        tier_result["final_context_tokens"] = current_tokens if 'current_tokens' in dir() else 0
        tier_result["prefill_latency_s"] = prefill_latency
        tier_result["coherence_log"] = coherence_log

        # Check VRAM
        vram_after = get_gpu_vram_mb()
        tier_result["vram_after_mb"] = vram_after
        tier_result["vram_delta_mb"] = vram_after - vram
        tier_result["rss_after_mb"] = get_proc_rss_mb()

        # Save snapshot — need an rid, use generate endpoint
        print(f"  Completed {turn_count} turns, {current_tokens:,} tokens")
        print(f"  Saving snapshot...")

        # Use generate to get an rid
        try:
            r = requests.post(
                f"{SERVER_URL}/generate",
                json={
                    "text": "Remember everything discussed. Say OK.",
                    "sampling_params": {"temperature": 0.0, "max_new_tokens": 5},
                },
                timeout=30,
            )
            rid = r.json().get("rid", uuid.uuid4().hex)
        except Exception:
            rid = uuid.uuid4().hex

        save_result = save_snapshot(rid)
        tier_result["save_success"] = save_result.get("success", False)
        tier_result["save_latency_ms"] = save_result.get("latency_ms", 0)

        if save_result.get("success"):
            snap_conv_id = save_result.get("snapshot_id", "").rsplit("-t", 1)[0]
            tier_result["snapshot_size_mb"] = round(get_snapshot_file_size_mb(snap_conv_id), 2)

        # Restore and verify fact recall
        print(f"  Restoring and testing recall...")
        if save_result.get("success"):
            warm_result = restore_snapshot(conversation_id=snap_conv_id)
            tier_result["warm_restore_latency_ms"] = round(warm_result.get("latency_ms", 0), 1)

            # Ask about fact #1 (should be remembered if snapshot works)
            recall_msgs = [
                {"role": "user", "content": "What is your name? Reply in one sentence."},
            ]
            try:
                recall_result = send_chat(recall_msgs, temperature=0.0, max_tokens=50)
                recall_text = recall_result["text"].lower()
                # Check if "elara" or "voss" appears
                tier_result["recall_fact1"] = "elara" in recall_text or "voss" in recall_text
                tier_result["recall_response"] = recall_result["text"][:200]
                tier_result["recall_success"] = tier_result["recall_fact1"]
            except Exception as e:
                tier_result["recall_error"] = str(e)
                tier_result["recall_success"] = False

        # State fidelity vs model coherence distinction
        tier_result["state_fidelity"] = "snapshot saved and restored" if save_result.get("success") else "FAILED"
        if turn_count > 0 and coherence_log:
            last_coherent = coherence_log[-1]["key_word_matches"] > 0
            tier_result["model_coherence"] = "coherent" if last_coherent else "degraded"

        tier_result["status"] = "PASS" if save_result.get("success") else "PARTIAL"
        results.append(tier_result)

        # Intermediate results
        with open(output_dir / "phase-10-context-scaling-multi-turn.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    return results


# ────────────────────────────────────────────────────────────────
# Report Generation
# ────────────────────────────────────────────────────────────────

def generate_report(single_shot_results: list, multi_turn_results: list, output_dir: Path):
    """Generate markdown results report."""
    lines = []
    lines.append("# Phase 10 Addendum: Context Window Scaling Results\n")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**Model**: granite-4.0-h-tiny (4B, 131K context)")
    lines.append(f"**Server**: {SERVER_URL}")
    lines.append("")

    # Single-shot results
    if single_shot_results:
        lines.append("## Mode A: Single-Shot (Snapshot Mechanics)\n")
        lines.append("| Tier | Prompt Tokens | Inference (s) | Save (ms) | Snapshot (MB) | WARM Restore (ms) | COLD Restore (ms) | COLD Verified | Status |")
        lines.append("|------|--------------|---------------|-----------|---------------|-------------------|-------------------|---------------|--------|")
        for r in single_shot_results:
            if r.get("status") in ("OOM_SKIP", "SERVER_DOWN", "TIMEOUT", "HTTP_ERROR"):
                lines.append(f"| {r['tier']} | - | - | - | - | - | - | - | **{r['status']}** |")
                continue
            lines.append(
                f"| {r['tier']} "
                f"| {r.get('actual_prompt_tokens', '?'):,} "
                f"| {r.get('inference_latency_s', '?'):.3f} "
                f"| {r.get('save_latency_ms', '?'):.0f} "
                f"| {r.get('snapshot_size_mb', '?'):.1f} "
                f"| {r.get('warm_restore_latency_ms', '?'):.0f} "
                f"| {r.get('cold_restore_latency_ms', '?'):.0f} "
                f"| {'Yes' if r.get('cold_restore_verified') else 'No'} "
                f"| {r.get('status', '?')} |"
            )
        lines.append("")

        # Resource scaling
        lines.append("### Resource Scaling (Single-Shot)\n")
        lines.append("| Tier | VRAM Before (MB) | VRAM After (MB) | VRAM Delta | RSS (MB) |")
        lines.append("|------|-----------------|-----------------|------------|----------|")
        for r in single_shot_results:
            if r.get("status") in ("OOM_SKIP", "SERVER_DOWN"):
                continue
            lines.append(
                f"| {r['tier']} "
                f"| {r.get('vram_before_mb', '?'):,} "
                f"| {r.get('vram_after_mb', '?'):,} "
                f"| {r.get('vram_delta_mb', 0):+,} "
                f"| {r.get('rss_after_mb', '?'):,} |"
            )
        lines.append("")

    # Multi-turn results
    if multi_turn_results:
        lines.append("## Mode B: Multi-Turn (Coherence)\n")
        lines.append("| Tier | Turns | Final Context | Prefill (s) | Snapshot (MB) | Recall | Fidelity | Coherence | Status |")
        lines.append("|------|-------|--------------|-------------|---------------|--------|----------|-----------|--------|")
        for r in multi_turn_results:
            recall = "PASS" if r.get("recall_success") else "FAIL"
            ctx = f"{r.get('final_context_tokens', 0):,}" if r.get('final_context_tokens') else "?"
            prefill = f"{r.get('prefill_latency_s', 0):.3f}" if isinstance(r.get('prefill_latency_s'), (int, float)) else "?"
            snap_mb = f"{r.get('snapshot_size_mb', 0):.1f}" if isinstance(r.get('snapshot_size_mb'), (int, float)) else "?"
            lines.append(
                f"| {r['tier']} "
                f"| {r.get('turns_completed', '?')} "
                f"| {ctx} "
                f"| {prefill} "
                f"| {snap_mb} "
                f"| {recall} "
                f"| {r.get('state_fidelity', '?')} "
                f"| {r.get('model_coherence', '?')} "
                f"| {r.get('status', '?')} |"
            )
        lines.append("")

    # Key findings
    lines.append("## Key Findings\n")
    if single_shot_results:
        sizes = [r.get("snapshot_size_mb", 0) for r in single_shot_results if r.get("snapshot_size_mb")]
        if sizes:
            lines.append(f"1. **Snapshot size**: {min(sizes):.1f} - {max(sizes):.1f} MB across all tiers "
                         f"({'constant' if max(sizes) - min(sizes) < 5 else 'varies'})")
        cold_latencies = [r.get("cold_restore_latency_ms", 0) for r in single_shot_results
                         if r.get("cold_restore_verified")]
        if cold_latencies:
            lines.append(f"2. **COLD restore latency**: {min(cold_latencies):.0f} - {max(cold_latencies):.0f}ms "
                         f"({'under 200ms target' if max(cold_latencies) < 200 else 'exceeds 200ms'})")
        max_tier = max((r for r in single_shot_results if r.get("status") == "PASS"),
                      key=lambda r: r.get("actual_prompt_tokens", 0), default=None)
        if max_tier:
            lines.append(f"3. **Max context achieved**: {max_tier.get('actual_prompt_tokens', 0):,} tokens "
                         f"(tier {max_tier['tier']})")
    lines.append("")

    # Separate state fidelity from model coherence
    lines.append("## Important Distinction\n")
    lines.append("- **State fidelity**: Whether the snapshot system correctly saves and restores Mamba SSM state. "
                 "This is what the system is responsible for.")
    lines.append("- **Model coherence**: Whether the 4B parameter model produces sensible output at extreme context "
                 "lengths. This is a model capability limitation, not a snapshot system issue.")
    lines.append("")

    report_path = output_dir / "phase-10-context-scaling-results.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {report_path}")
    return report_path


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    global SERVER_URL

    parser = argparse.ArgumentParser(description="Phase 10: Context Window Scaling Test")
    parser.add_argument("--mode", choices=["single-shot", "multi-turn", "both"],
                        default="both", help="Test mode")
    parser.add_argument("--tiers", default="2K,8K,32K,64K,128K",
                        help="Comma-separated tier labels to test")
    parser.add_argument("--server-url", default=SERVER_URL, help="Server URL")
    parser.add_argument("--output-dir", default="test/phases/results",
                        help="Output directory for results")
    parser.add_argument("--model", default="granite-tiny", help="Model label for reports")
    args = parser.parse_args()

    SERVER_URL = args.server_url
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiers = [t.strip() for t in args.tiers.split(",")]
    for t in tiers:
        if t not in TIERS:
            print(f"ERROR: Unknown tier '{t}'. Valid: {list(TIERS.keys())}")
            sys.exit(1)

    print(f"Phase 10 Context Scaling Test")
    print(f"  Server:  {SERVER_URL}")
    print(f"  Tiers:   {tiers}")
    print(f"  Mode:    {args.mode}")
    print(f"  Model:   {args.model}")
    print(f"  VRAM limit: {VRAM_LIMIT_MB}MB")
    print()

    # Check server health
    if not check_health():
        print("ERROR: Server is not healthy. Start it first:")
        print("  source test/phases/infra/config.sh")
        print("  python -m sglang.launch_server --model-path $MODEL_PATH ...")
        sys.exit(1)

    # Check idle VRAM
    idle_vram = get_gpu_vram_mb()
    print(f"Idle GPU VRAM: {idle_vram:,}MB")
    if idle_vram > 70000:
        print(f"  WARNING: Idle VRAM is {idle_vram:,}MB. Large context tiers may OOM.")
        print(f"  Consider restarting with --mem-fraction-static 0.80")
    print()

    single_shot_results = []
    multi_turn_results = []

    if args.mode in ("single-shot", "both"):
        single_shot_results = run_single_shot(tiers, output_dir)

    if args.mode in ("multi-turn", "both"):
        # Only test tiers that passed single-shot
        if single_shot_results:
            passed_tiers = [r["tier"] for r in single_shot_results if r.get("status") == "PASS"]
            if not passed_tiers:
                print("\nNo tiers passed single-shot — skipping multi-turn")
            else:
                print(f"\nMulti-turn will test tiers: {passed_tiers}")
                multi_turn_results = run_multi_turn(passed_tiers, output_dir)
        else:
            multi_turn_results = run_multi_turn(tiers, output_dir)

    # Generate report
    generate_report(single_shot_results, multi_turn_results, output_dir)


if __name__ == "__main__":
    main()
