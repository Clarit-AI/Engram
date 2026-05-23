"""
KHA-390 regression — Mamba SSM consumption ablation.

After /restore_snapshot succeeds, a subsequent plain /generate must consume the
restored SSM state: different SSM seeds must produce distinct output.

Fix confirmed on-main (ecb999e65): [none] and [gaussian] tamper conditions produce
output distinct from the no-restore baseline; distinguishes_conditions passes.
The [zeros] case is a known mathematical edge case: h0=0 recurrence equals starting
fresh, so zeroed restore == no-restore is expected behaviour, not the original bug.

Helpers and constants are lifted from scripts/kha-387-ssm-ablation.py, which
proved the bug empirically on pure-Mamba2 Codestral.
"""

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import requests
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ── configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.environ.get("KHA390_MODEL", "/workspace/models/mamba-codestral-7b")
DEFAULT_PORT = int(os.environ.get("KHA390_PORT", "30201"))
DEFAULT_SERVED_NAME = os.environ.get("KHA390_SERVED_NAME", "mamba-codestral-7b")
HEALTH_TIMEOUT_S = int(os.environ.get("KHA390_HEALTH_TIMEOUT", "600"))

# ── constants (lifted from scripts/kha-387-ssm-ablation.py) ─────────────────

# Two-segment Codestral prompt: a "seed" that primes a recognisable code
# structure, and a "tail" that without SSM state produces unrelated content.
# Modelled after KHA-385 v2's python_constant case (restore-and-generate recall
# of a sentinel).
SEED_PROMPT = (
    "import unittest\n"
    "# regression marker: ZEPHYR_KHA387:271828\n"
    "# this constant is the canonical sentinel used by all KHA-387 tests below\n"
    "# stored_pair returns the sentinel as a tuple of (1, 2)\n"
    "def stored_pair():\n"
    "    # canonical sentinel string defined above is\n"
)
# Continuation — only the immediate tail.  Without state the model falls back to
# its prior; with consumed state it recalls the sentinel.
TAIL_PROMPT = "    return "

SENTINEL = "ZEPHYR_KHA387:271828"
MAX_TOKENS = 96
SAMPLING: Dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "n": 1,
    "max_new_tokens": MAX_TOKENS,
}

# ── HTTP helpers (lifted from scripts/kha-387-ssm-ablation.py) ──────────────


def _http_post(url: str, payload: dict, timeout: float = 120.0) -> Tuple[int, dict]:
    r = requests.post(url, json=payload, timeout=timeout)
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text}
    return r.status_code, body


def _wait_for_health(server_url: str, max_wait_s: int = 600) -> None:
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{server_url}/health", timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(
        f"Server at {server_url} did not become healthy in {max_wait_s}s"
    )


# ── snapshot path helpers (lifted from scripts/kha-387-ssm-ablation.py) ─────


def _turn_state_path(snap_root: Path, conv_id: str, turn: int) -> Path:
    return snap_root / f"conversation_{conv_id}" / f"turn_{turn}_state.safetensors"


# ── tensor tamper helpers (lifted from scripts/kha-387-ssm-ablation.py) ─────


def _read_snapshot_tensors(state_path: Path) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    with safe_open(state_path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _write_snapshot_tensors(state_path: Path, tensors: Dict[str, torch.Tensor]) -> None:
    # Atomic overwrite — same approach as the original script.
    tmp = state_path.with_suffix(state_path.suffix + ".kha390tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, state_path)


def _zero_tensors(orig: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in orig.items()}


def _randomize_tensors(orig: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Gaussian with per-tensor stats matched to the original so we don't trip
    # the load-time NaN/Inf guards (validate_state_tensors only flags exact
    # all-zero as WARNING; random with matched stats passes cleanly).
    out: Dict[str, torch.Tensor] = {}
    for k, v in orig.items():
        std = max(float(v.float().std().item()), 1e-3)
        mean = float(v.float().mean().item())
        sample = torch.randn_like(v.float()) * std + mean
        sample = sample.clamp(min=mean - 6 * std, max=mean + 6 * std)
        out[k] = sample.to(v.dtype)
    return out


# ── per-request helpers ──────────────────────────────────────────────────────


def _emit(
    server_url: str, served_name: str, text: str, rid: str, conv_id: str
) -> Tuple[int, dict]:
    payload = {
        "text": text,
        "sampling_params": SAMPLING,
        "rid": rid,
        "conversation_id": conv_id,
        "model": served_name,
    }
    return _http_post(f"{server_url}/generate", payload)


def _seed_snapshot(
    server_url: str,
    served_name: str,
    snap_root: Path,
) -> Tuple[str, str, Path]:
    """Seed /generate + /save_snapshot.  Returns (conv_id, rid_seed, state_path)."""
    conv_id = f"kha390-{uuid.uuid4().hex[:12]}"
    rid_seed = f"{conv_id}-seed"

    code, body = _emit(server_url, served_name, SEED_PROMPT, rid_seed, conv_id)
    assert code == 200, f"seed /generate failed HTTP {code}: {body}"

    save_payload: Dict[str, Any] = {
        "rid": rid_seed,
        "conversation_id": conv_id,
        "snapshot_id": f"{conv_id}-t1",
        "turn_number": 1,
    }
    code, body = _http_post(f"{server_url}/save_snapshot", save_payload)
    assert code == 200 and body.get(
        "success"
    ), f"/save_snapshot failed HTTP {code}: {body}"

    state_path = _turn_state_path(snap_root, conv_id, 1)
    assert state_path.exists(), f"Expected safetensors at {state_path} after save"

    return conv_id, rid_seed, state_path


def _tamper_snapshot_on_disk(state_path: Path, mode: str) -> None:
    """Overwrite state_path with zeros or Gaussian noise in-place."""
    assert mode in ("zeros", "gaussian"), f"Unknown tamper mode: {mode!r}"
    orig = _read_snapshot_tensors(state_path)
    mutated = _zero_tensors(orig) if mode == "zeros" else _randomize_tensors(orig)
    _write_snapshot_tensors(state_path, mutated)


def _restore(server_url: str, conv_id: str, rid_tail: str) -> None:
    """POST /restore_snapshot with turn_number=1 to force the disk path (per KHA-387)."""
    payload: Dict[str, Any] = {
        "rid": rid_tail,
        "conversation_id": conv_id,
        "turn_number": 1,
    }
    code, body = _http_post(f"{server_url}/restore_snapshot", payload)
    assert code == 200 and body.get(
        "success"
    ), f"/restore_snapshot failed HTTP {code}: {body}"


def _continuation_ids(
    server_url: str, served_name: str, conv_id: str, rid_tail: str
) -> Tuple[int, ...]:
    """Run TAIL_PROMPT and return the first 32 output token IDs as a tuple."""
    code, body = _emit(server_url, served_name, TAIL_PROMPT, rid_tail, conv_id)
    assert code == 200, f"tail /generate failed HTTP {code}: {body}"
    ids = (
        (body.get("meta_info") or {}).get("output_ids") or body.get("output_ids") or []
    )
    if not isinstance(ids, list):
        ids = []
    return tuple(ids[:32])


# ── server fixture (scope=module — one launch per test module) ───────────────


class _ServerHandle:
    def __init__(self, server_url: str, served_name: str, snap_root: Path) -> None:
        self.server_url = server_url
        self.served_name = served_name
        self.snap_root = snap_root


@pytest.fixture(scope="module")
def launch_codestral_server(tmp_path_factory: pytest.TempPathFactory) -> _ServerHandle:
    """
    Launch a Codestral server with the exact flags from scripts/kha-387-ssm-ablation.sh:
      --disable-radix-cache --mamba-scheduler-strategy no_buffer
      --disable-cuda-graph --disable-piecewise-cuda-graph
    Yields a _ServerHandle; tears down on module exit.
    """
    snap_root: Path = tmp_path_factory.mktemp("kha390_snaps")
    port = DEFAULT_PORT
    server_url = f"http://localhost:{port}"

    env = os.environ.copy()
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["SGLANG_ENABLE_SPEC_V2"] = "false"

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        DEFAULT_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--trust-remote-code",
        "--served-model-name",
        DEFAULT_SERVED_NAME,
        "--enable-snapshot-persistence",
        "--snapshot-dir",
        str(snap_root),
        "--snapshot-trigger-policy",
        "every_turn",
        "--mamba-scheduler-strategy",
        "no_buffer",
        "--disable-radix-cache",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
    ]

    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        _wait_for_health(server_url, max_wait_s=HEALTH_TIMEOUT_S)
        yield _ServerHandle(
            server_url=server_url, served_name=DEFAULT_SERVED_NAME, snap_root=snap_root
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ── tests ────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.snapshot
@pytest.mark.parametrize(
    "tamper",
    [
        "none",
        pytest.param(
            "zeros",
            marks=pytest.mark.xfail(
                reason=(
                    "zeroed restored state == no restored state; the h0=0 recurrence "
                    "equals starting fresh, so restore_zeros legitimately equals the "
                    "no-restore baseline (KHA-390) — expected math, not the bug"
                ),
                strict=True,
            ),
        ),
        "gaussian",
    ],
)
def test_restore_consumption_changes_output(
    launch_codestral_server: _ServerHandle, tamper: str
) -> None:
    """
    After /restore_snapshot, /generate must produce output distinct from the
    no-restore baseline.  Currently fails: restored SSM state is ignored (KHA-390).
    """
    srv = launch_codestral_server

    # Baseline: seed + save, continuation WITHOUT restore.
    conv_base, _, _state_base = _seed_snapshot(
        srv.server_url, srv.served_name, srv.snap_root
    )
    rid_base_tail = f"{conv_base}-tail"
    baseline_ids = _continuation_ids(
        srv.server_url, srv.served_name, conv_base, rid_base_tail
    )

    # Restored path: fresh seed + save, optional tamper, restore, then continuation.
    conv_r, _, state_path_r = _seed_snapshot(
        srv.server_url, srv.served_name, srv.snap_root
    )
    if tamper != "none":
        _tamper_snapshot_on_disk(state_path_r, tamper)
    rid_r_tail = f"{conv_r}-tail"
    _restore(srv.server_url, conv_r, rid_r_tail)
    restored_ids = _continuation_ids(
        srv.server_url, srv.served_name, conv_r, rid_r_tail
    )

    assert restored_ids != baseline_ids, (
        f"tamper={tamper!r}: restored output is byte-identical to no-restore baseline — "
        f"SSM state not consumed (KHA-390). ids={restored_ids!r}"
    )


@pytest.mark.gpu
@pytest.mark.snapshot
def test_restore_consumption_distinguishes_conditions(
    launch_codestral_server: _ServerHandle,
) -> None:
    """
    With SSM state consumed, different tamper conditions must yield distinct output.
    Currently all three are identical — confirming the state is not reaching the
    forward pass (KHA-390).
    """
    srv = launch_codestral_server

    outputs: Dict[str, Tuple[int, ...]] = {}
    for tamper in ("none", "zeros", "gaussian"):
        conv_id, _, state_path = _seed_snapshot(
            srv.server_url, srv.served_name, srv.snap_root
        )
        if tamper != "none":
            _tamper_snapshot_on_disk(state_path, tamper)
        rid_tail = f"{conv_id}-tail"
        _restore(srv.server_url, conv_id, rid_tail)
        outputs[tamper] = _continuation_ids(
            srv.server_url, srv.served_name, conv_id, rid_tail
        )

    assert len(set(outputs.values())) == 3, (
        f"All three tamper conditions produced non-distinct output (KHA-390). "
        f"none={outputs['none']!r} zeros={outputs['zeros']!r} gaussian={outputs['gaussian']!r}"
    )
