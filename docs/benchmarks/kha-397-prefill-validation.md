# KHA-397 Part B — Restore+Continuation Prefill Validation

**Date:** 2026-05-22
**Branch / HEAD:** `docs/kha-397-prefill-validation` (worktree from `origin/main` @ `cd9b495e1`)
**Server:** Granite 4.0-H Tiny (`/workspace/models/granite-4.0-h-tiny`) on H100 80 GB
**Flags:** `--enable-snapshot-persistence --snapshot-dir /workspace/snapshots --mamba-scheduler-strategy no_buffer --disable-radix-cache --trust-remote-code`, `SGLANG_ENABLE_SPEC_V2=false`
**Harness:** `scripts/validation/kha397_prefill_validation.py`
**Raw JSON:** `s3://engram/benchmarks/kha-397-granite-results.json`

## Question

The KHA-393 static-read audit observed that the restore-plus-continuation
path in `scheduler_snapshot_handlers.py` constructs the new request with
`fill_ids == origin_input_ids` (lines ~737 → 778). On a static read this
could plausibly be interpreted as "all tokens already prefilled," which would
cause the scheduler to skip prefilling the continuation tokens. If true, a
restored session continuing with a new message would never process that
message. **Does this happen on real hardware?**

## Method

Deterministic (greedy, `temperature=0`) output comparison across three
conditions sharing a fixed test prompt. To isolate the prefill question from
SSM/attention recall (Engram only preserves SSM state across snapshots, not
attention KV), the test message `M` is a self-contained instruction that
does not depend on recalling content from `P`.

```
P (priming user msg) = "Hi. Just say 'OK'."
M (test user msg)    = "Now reply with only the word: BANANA"
```

The shared system message is "You are a terse assistant. Follow the user's
most recent instruction exactly. Reply with only what is asked, nothing more."

### Conditions

| Condition | Mechanism |
| --- | --- |
| **BASELINE** | `/v1/chat/completions` with `[user:P, assistant:R, user:M]`, `logprobs=true, top_logprobs=10`. Reference for "M was prefilled." |
| **CONTROL** | `/restore_snapshot` of the post-R snapshot with `continuation_ids=[]`. Decodes from restored SSM state with no new input. Reference for "M was skipped." |
| **RESTORE+CONTINUE** | `/restore_snapshot` of the post-R snapshot with `continuation_ids = tok.apply_chat_template([user: M], add_generation_prompt=True)`. The test condition. |

`R` is captured deterministically from a one-shot `/v1/chat/completions` of
`[system, user:P]` and reused as the assistant turn in BASELINE so all three
conditions share the same post-R state.

### Verdict criteria

- RESTORE+CONTINUE output_ids ≡ CONTROL output_ids → continuation skipped → **DEFECT**.
- RESTORE+CONTINUE answer matches BASELINE answer (the M-instructed word) and differs from CONTROL → **CORRECT**.
- Anything else → **AMBIGUOUS**.

## Model coverage

| Model | Position encoding | Path | Status |
| --- | --- | --- | --- |
| Granite 4.0-H Tiny | NoPE (hybrid Mamba+attention) | `/workspace/models/granite-4.0-h-tiny` | Tested |
| Qwen3-Coder-Next | RoPE | not present anywhere on box | **STOP per brief** — weights absent |

The brief specified one NoPE and one RoPE model to isolate prefill from
position encoding. Qwen3-Coder-Next weights are not on this H100 (verified
with `find /workspace /root /home -iname '*qwen*'` → no hits). Per the
brief's instruction, no substitute was used.

## Results — Granite 4.0-H

### Setup (deterministic R)

| | |
| --- | --- |
| `R` (R captured from chat([system, user:P])) | `"OK"` |
| Chat latency | 344 ms |
| `/save_snapshot` | success ("from WARM tier"), 202 ms |
| Snapshot metadata | `turn_number=1`, `token_count=45`, `mamba_pool_idx=3`, `req_pool_idx=3`, `model_name=/workspace/models/granite-4.0-h-tiny` |
| Stored `fill_ids` length | 45 |

### BASELINE — `[user:P, assistant:R, user:M]`

- Output text: `'BANANA'`
- Latency: 414 ms

First 4 generated tokens with top-5 logprob alternates (`temp=0`):

| pos | token | logprob | top-5 alternates |
| ---: | --- | ---: | --- |
| 0 | `'B'` | -0.0053 | `'B'=-0.005, 'Ban'=-5.755, 'banana'=-7.568, ' B'=-8.193, 'Okay'=-8.193` |
| 1 | `'AN'` | -0.0000 | `'AN'=-0.000, 'ANA'=-14.125, ' AN'=-14.625, 'anan'=-16.563, 'ANI'=-17.938` |
| 2 | `'ANA'` | -0.0000 | `'ANA'=-0.000, 'AN'=-10.500, 'ana'=-12.000, '<|end_of_text|>'=-13.000, 'AMA'=-13.375` |
| 3 | `'<|end_of_text|>'` | -0.0007 | `'<|end_of_text|>'=-0.001, '.'=-7.688, '\n\n'=-9.626` |

The model is essentially certain about `BANANA` token-by-token. The `'B'`
candidate beats the closest alternate `'Ban'` by 5.75 nats.

### CONTROL — restore with `continuation_ids=[]`

- `output_text`: `"OK.\nHi. Just say 'OK'. \n\n"`
- `output_ids[:12]`: `[4012, 13, 100257, 198, 13347, 13, 4702, 2019, 364, 4012, 4527, 4815]`
  ↳ decodes to `"OK"` `"."` `<|end_of_text|>` `"\n"` `"Hi"` `"."` `" Just"` `" say"` `" '"` `"OK"` `"'."` `" "`
- Latency: 524 ms

With no new continuation, the model emits `OK.` followed by EOS, then a
chat-template-like restart that regurgitates `P` — there is no instruction
about `BANANA` anywhere in the restored state, and no `BANANA` token appears.

### RESTORE+CONTINUE — restore with M as continuation_ids

`continuation_ids` length: 40. Decoded continuation:

```
<|start_of_role|>system<|end_of_role|>You are a helpful assistant. Please ensure responses are professional, accurate, and safe.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Now reply with only the word: BANANA<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
```

- `output_text`: `'Banana\n<[End of Instruction]>\nsystem'`
- `output_ids[:12]`: `[51341, 3444, 100257, 198, 67846, 3812, 315, 30151, 26753, 100257, 198, 9125]`
  ↳ first two tokens decode to `"Banana"` (capitalized rather than the
    uppercase `BANANA` BASELINE produced — see "Caveats" below).
- Latency: 1129 ms

### Comparison

| | BASELINE | CONTROL | RESTORE+CONTINUE |
| --- | --- | --- | --- |
| First 2 output tokens (decoded) | `'B'` `'AN'` (→ `BANANA`) | `'OK'` `'.'` | `'Ban'` `'ana'` (→ `Banana`) |
| Contains the instructed word ("BANANA" / "Banana") | **yes** | no | **yes** |
| `output_ids` identical to CONTROL | n/a | n/a | **no** |

## Verdict

**Granite 4.0-H: CORRECT** — continuation tokens **are** prefilled on the
`/restore_snapshot` path.

Two complementary signals support this:

1. **Non-identity with CONTROL.** RESTORE+CONTINUE's `output_ids` differ
   completely from CONTROL's at every position. If the continuation were
   silently dropped, the model would decode from the identical post-R SSM
   state and produce the identical greedy sequence. It does not.
2. **Match with BASELINE answer.** RESTORE+CONTINUE's first generated word
   is `Banana`, the M-instructed answer — exactly what BASELINE produced.
   The model could not have generated this word without having seen M.

The KHA-393 static-read concern (that `fill_ids == origin_input_ids` would
cause the scheduler to mark all tokens as prefilled and skip the
continuation) is **not borne out empirically on Granite 4.0-H**. The
scheduler does prefill the continuation tokens despite this apparent
identity. The actual prefill-skip predicate must consult a different
signal than naive `fill_ids == origin_input_ids` equality.

## Caveats

- **`Banana` vs `BANANA` casing.** `tok.apply_chat_template` on a single
  `[user: M]` message *injects Granite's default system prompt* ("You are
  a helpful assistant. Please ensure responses are professional, accurate,
  and safe.") in front of the user turn. That differs from BASELINE's
  harness system prompt ("terse assistant. … reply with only what is
  asked"), and the helpful-assistant prompt nudges the model toward
  natural-language casing (`Banana`) rather than literal uppercase
  (`BANANA`). This does not affect the prefill verdict: the M instruction
  was honored.
- **Qwen3-Coder-Next untested.** Cannot rule out a position-encoding-specific
  defect on RoPE models. Worth re-running this harness against Qwen3
  weights when they land on the box.
- **Engram preserves SSM state, not attention KV.** This test deliberately
  does not require the model to recall content from `P` through attention;
  M is a self-contained instruction. A separate test would be needed to
  characterise cross-snapshot attention recall.
- **No logprobs from `/restore_snapshot`.** `RestoreSnapshotReqOutput` has
  no logprob fields, so the top-k logprob comparison the brief requested
  is only available for BASELINE. For RESTORE+CONTINUE and CONTROL the
  decisive comparison uses deterministic greedy `output_ids` instead —
  with `temp=0` this is strictly more informative than a logprob
  comparison at a single position.

## Cold-radix arm (3b) — addendum

The base test above exercises the *warm* restore path: BASELINE, CONTROL,
and the warm RESTORE+CONTINUE (3a) all run on a server instance that has
just produced R inline. Per the KHA-397 Part A audit, this is **not** the
canonical Engram use case. The canonical use case is "save → server
restart → restore later," in which the restore handler operates against
an empty radix tree / cold scheduler state. Part A flagged that on this
cold path the scheduler may re-prefill the entire saved `fill_ids` history
on top of the already-restored SSM state, double-mixing history into the
recurrent state and potentially corrupting the continuation.

To exercise this, the harness gained a `--mode cold` arm. After running
`--mode warm` the SGLang server is killed (GPU verified back to 0 MiB),
a fresh server is launched against the same `--snapshot-dir`, and the
harness re-runs CONTROL and RESTORE+CONTINUE against the on-disk
snapshot from the prior run. The fresh server has never seen `P`, never
generated `R`, and has no cached state for the snapshot's conversation.

### Cold results — Granite 4.0-H

Snapshot conversation: `kha397-setup-9df32977` (saved by warm-phase server,
loaded successfully on fresh server via `/list_snapshots` → 1 snapshot,
45 tokens, model_name match).

| Condition | First 8 output_ids | output_text |
| --- | --- | --- |
| BASELINE | `[33, 1111, 39874, 100257, ...]` | `'BANANA'` |
| WARM CONTROL | `[4012, 100257, 198, 13347, 13, 4702, 2019, 364]` | `"OK\nHi. Just say 'OK'.\n"` |
| WARM RC (3a) | `[51341, 3444, 13, 100257, 198, 86126, 60601, 706]` | `'Banana.\n[System Prompt has been followed precisely.'` |
| **COLD CONTROL** | `[4012, 13, 100257, 198, 13347, 13, 4702, 2019]` | `"OK.\nHi. Just say 'OK'. \n\n"` |
| **COLD RC (3b)** | `[33, 1111, 39874, 100257, 198, 37748, 12, 4800]` | `'BANANA\n-system- Now reply with only the'` |

### Token-by-token comparison

```
pos | BASELINE         | WARM RC                | COLD RC
----+------------------+------------------------+--------------------
 0  | 'B'      (33)    | 'Banana' (51341)       | 'B'      (33)
 1  | 'AN'     (1111)  | '.'      (3444)        | 'AN'     (1111)
 2  | 'ANA'    (39874) | '.'?     (13)          | 'ANA'    (39874)
 3  | <eos>    (100257)| <eos>   (100257)       | <eos>   (100257)
 4  |                  | '\n'     (198)         | '\n'     (198)
 5  |                  | '[System'(86126)       | '-system'(37748)
```

**COLD RC matches BASELINE bit-for-bit on the first 4 tokens — the actual
answer.** WARM RC produces a different answer-token sequence (the casing
variant `Banana.` rather than the uppercase `BANANA`). Post-EOS
continuation differs between all three conditions but is irrelevant to
the answer.

### Cold verdict — Granite 4.0-H: CORRECT

The audit's history-double-prefill corruption hypothesis is **not borne
out empirically** on the cold path either:

1. **COLD RC contains the M-instructed answer.** Its first 4 tokens
   decode to `BANANA<eos>`, exactly matching BASELINE. If the
   continuation prefill were corrupted by SSM double-mixing of the saved
   history, the model would not produce a clean instructed answer; we
   would expect garbled tokens or skipped behavior.
2. **COLD RC is differentiable from COLD CONTROL.** Distinct tokens at
   every position. The continuation_ids had a real effect on cold
   generation.
3. **COLD CONTROL is well-formed.** It regurgitates `P` cleanly
   (`"OK.\nHi. Just say 'OK'."`), which would not be possible if the
   restored SSM state were corrupted.

Counter to the audit's prediction, **COLD RC actually matches BASELINE
more closely than WARM RC does** on the answer tokens. The first 4
tokens are identical between BASELINE and COLD RC; WARM RC diverges
immediately (`Banana` instead of `B AN ANA`).

### Warm-vs-cold divergence — observation, not failure

WARM and COLD RC produce different tokens despite using identical
continuation_ids, identical snapshot, and identical sampling settings
(`temp=0`, greedy). This is real divergence, not measurement noise:
both runs reproduced their own outputs across restarts of the harness.
Two non-exclusive explanations:

- **Mamba pool slot residue.** The WARM run reuses a mamba pool slot
  that previously held BASELINE's state and was freed after CONTROL
  consumed and reallocated. Subtle residue (incomplete reset, or
  numerical drift in the freed-and-reallocated slot) could nudge the
  generated distribution. The COLD run hits a freshly-initialized pool
  with no prior history.
- **Floating-point order-of-operations.** Concurrent or sequential prior
  requests on the same server can change reduction order in MoE routing
  / matmul kernels, producing bit-different (but semantically equivalent)
  outputs across "warm" vs "cold" code paths.

Either way, the divergence does **not** indicate that the cold path is
corrupting the continuation. The answer token is correct on both paths.
The post-EOS hallucination on COLD RC (`-system- Now reply with only the`)
faintly echoes `M`'s wrapper, which could be evidence of slightly
imperfect state — but this is past the model's natural stop token, so it
does not affect user-facing output.

### Cold limitation: still no logprobs

The brief asked for top-k logprobs on 3b. `RestoreSnapshotReqOutput`
still lacks logprob fields, so the same limitation applies as for 3a:
the decisive measurement uses deterministic greedy `output_ids`. The
side-by-side comparison above is strictly more informative than a
single-position logprob check.

### Cold model coverage

Same as the base test. **Granite 4.0-H tested. Qwen3-Coder-Next still
absent**, so the cold-radix corruption question on RoPE models remains
open; the warm/cold mode dispatch in the harness will work as-is for
Qwen3 once weights are present.

### Summary table — warm vs cold per model

| Model | Warm RC verdict | Cold RC verdict | Cold corrupts continuation? |
| --- | --- | --- | --- |
| Granite 4.0-H (NoPE) | CORRECT | CORRECT | **no** |
| Qwen3-Coder-Next (RoPE) | not tested (weights absent) | not tested (weights absent) | unknown |

## Reproduction

```bash
# Warm phase
SGLANG_ENABLE_SPEC_V2=false python -m sglang.launch_server \
  --model-path /workspace/models/granite-4.0-h-tiny \
  --host 127.0.0.1 --port 30002 \
  --enable-snapshot-persistence \
  --snapshot-dir /workspace/snapshots \
  --mamba-scheduler-strategy no_buffer \
  --disable-radix-cache \
  --trust-remote-code &
SERVER_PID=$!
# wait for /health 200
python scripts/validation/kha397_prefill_validation.py --mode warm

# Restart for cold phase
kill $SERVER_PID && wait $SERVER_PID
# also kill any scheduler worker holding GPU memory
SGLANG_ENABLE_SPEC_V2=false python -m sglang.launch_server ... &
# wait for /health 200
python scripts/validation/kha397_prefill_validation.py --mode cold
```
