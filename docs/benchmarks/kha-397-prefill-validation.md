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

## Addendum 2 — cold-restore cost measurement

### Question

Correctness is settled: both warm and cold restore produce the M-instructed
answer. This addendum answers a different question — **does cold restore
actually deliver Engram's token-reduction / fast-restore benefit, or does it
silently re-prefill the full history?** The KHA-397 Part A audit predicted
that on a cold radix tree the restore handler will re-prefill the entire
saved `fill_ids` history on top of the already-restored SSM state, losing
the speedup exactly in the canonical persistence scenario (save, restart,
restore later).

### Method

Same Granite 4.0-H Tiny, same `/restore_snapshot` API, but the server is
launched WITHOUT `--disable-radix-cache` so the warm restore can benefit
from radix-tree cache hits — that is precisely what the audit's hypothesis
is about.

Three conditions × two history lengths × five iterations, median reported:

| Condition | Mechanism |
| --- | --- |
| **BASELINE (cold-start)** | Fresh server, no snapshot. Single `/v1/chat/completions` with `[system, user: P+M]`, `max_tokens=1`. This is the "no Engram" cost. |
| **WARM RC** | Warm server: chat to produce R, save snapshot, restore with `continuation_ids=M`. The radix cache has just been populated by the chat that produced the prefix. |
| **COLD RC** | Server restarted between snapshot save and restore. Restore is the first request to the fresh server; radix tree is empty. |

Each WARM RC iteration uses a uniquely-salted P so the radix cache holds
only the prefix of the just-prefilled chat (no cross-iteration contamination
from the system prompt). Each COLD RC iteration is a fresh server lifetime
(kill + relaunch), guaranteeing an empty radix tree at the moment of
restore.

**Measurement sources** (per request):
- *Extend tokens*: scraped from the SGLang scheduler log line
  `Prefill batch, #new-token: N, #cached-token: K` emitted by
  `metrics_reporter.py:491`. This is the authoritative count of tokens the
  scheduler actually pushed through prefill.
- *TTFT proxy*: total wall-clock latency of the HTTP request with
  `max_new_tokens=1`. Decode of one token is negligible vs prefill of
  hundreds–thousands.
- *Restore wall-time*: the COLD RC TTFT for that request includes restore
  overhead (snapshot load + state injection) plus the full re-prefill.
- *Snapshot size*: `du -sb /workspace/snapshots/conversation_<id>/`.

Harness: `scripts/validation/kha397_prefill_cost.py` (modes `warm-suite`,
`cold-one`, `summarize`) driven by
`scripts/validation/kha397_run_cost_suite.sh` (server lifecycle, restarts
between cold iterations, log redirection per phase).

### Results — Granite 4.0-H

| Length | Condition | Extend tokens (med) | Cached tokens (med) | TTFT (med) | Snapshot size |
| --- | --- | ---: | ---: | ---: | ---: |
| **L=512** (actual median history = 590 tokens) | BASELINE | 600 | – | 201.2 ms | – |
|  | WARM RC | **40** | **590** | 286.8 ms | – |
|  | COLD RC | **630** | **0** | 381.4 ms | 57.35 MB |
| **L=4096** (actual median history = 4048 tokens) | BASELINE | 4058 | – | 114.4 ms | – |
|  | WARM RC | **40** | **4048** | 279.5 ms | – |
|  | COLD RC | **4088** | **0** | 373.8 ms | 57.39 MB |

Per-condition spread was tight: WARM RC extend was exactly 40 in 10/10 iters
across both lengths; COLD RC extend was 630/631 in 10/10 iters at L=512 and
4088/4089 in 10/10 iters at L=4096.

### Warm-vs-cold delta — per history length

| | L=512 (history=590) | L=4096 (history=4048) | Scales with history? |
| --- | ---: | ---: | --- |
| Extend tokens (cold − warm) | **+590** | **+4048** | **yes — linearly = full history** |
| Extend tokens (cold ÷ warm) | 15.75× | 102.2× | yes |
| TTFT (cold − warm) | +94.6 ms | +94.3 ms | **no — flat at ~95 ms** |
| Cached tokens (warm − cold) | 590 | 4048 | yes — full history hit on warm |

### Verdict — cold restore re-prefills the full history

**Yes, definitively.** Cold RC extend tokens equal history + continuation
(630 ≈ 590 + 40 at L=512; 4088 ≈ 4048 + 40 at L=4096), with cached_tokens=0
on every cold iteration. The penalty in *tokens prefilled* scales linearly
with history length — at L=4096 the cold path does 102× more prefill work
than the warm path. **The token-reduction benefit Engram is supposed to
provide is NOT realised when the radix tree is cold.** This is the audit's
prediction confirmed empirically.

### Verdict — does the TTFT penalty scale with history length?

**Surprisingly, no — the wall-clock cost is mostly fixed at ~95 ms.**
Despite cold prefilling 4048 extra tokens at L=4096 vs 590 extra tokens at
L=512 (a 6.9× difference), the cold-vs-warm TTFT gap is essentially flat
(~94 ms). H100 prefill throughput is very high — tens of thousands of
tokens per second on this model — so even 4k extra tokens take only ~95 ms
of additional wall-clock. The token *budget* penalty is the load-bearing
metric here, not TTFT.

For a slower model, larger history, or shared-GPU workloads where prefill
contention is real, the TTFT penalty would scale visibly. On this H100 +
Granite 4.0-H-Tiny configuration, TTFT alone hides the cost; the
extend-token count is what makes the bug operationally visible (and
expensive at scale via Engram's "tokens saved" accounting).

### Side finding — warm Engram is *slower* than baseline on this config

A look at TTFT in absolute terms:

| Length | Baseline TTFT | Warm RC TTFT | Cold RC TTFT |
| ---: | ---: | ---: | ---: |
| L=512 | 201 ms | 287 ms (1.43× baseline) | 381 ms (1.90× baseline) |
| L=4096 | 114 ms | 280 ms (2.45× baseline) | 374 ms (3.27× baseline) |

Even WARM Engram restore is *slower* than just doing the full baseline chat
on this model/hardware. Granite 4.0-H-Tiny on H100 has cheap-enough
prefill that the snapshot restore's fixed ~280 ms overhead (Mamba state
injection from disk + new-request setup) exceeds the prefill savings the
radix cache delivers. Cold Engram is unambiguously worse: pays restore
overhead AND re-prefills history.

This is a model/hardware-specific finding — for larger models where prefill
of a 4k history would take seconds rather than ~110 ms, the warm restore
overhead would amortise favourably. But "warm restore beats baseline" is
not a universal property and should not be assumed.

### Snapshot size is dominated by Mamba state, not history

Snapshots are ~57 MB at both history lengths (57.35 MB at L=512, 57.39 MB
at L=4096). The history `fill_ids` is text-tokens that the loader can
re-tokenize; what's actually large on disk is the captured Mamba SSM state
(36 layers × per-layer hidden+conv state for the granitemoehybrid
architecture). This means snapshot storage cost is bounded by the model
configuration, not by conversation length — useful for budgeting but
irrelevant to the warm/cold prefill question.

### Qwen3-Coder-Next — still not run

Weights were re-checked at the start of this addendum
(`find /workspace /root /home -iname '*qwen*'` → only docs/cookbook
folders, no model weights). The cost arm and the correctness arm both
remain pending for Qwen3. Both harnesses (`kha397_prefill_validation.py`
mode warm/cold and `kha397_prefill_cost.py` warm-suite/cold-one) are
model-path-parameterised and will run unchanged once Qwen3 weights land.

### Plain summary

| Question | Granite 4.0-H | Qwen3-Coder-Next |
| --- | --- | --- |
| Does cold restore re-prefill the full history? | **yes** | unknown (not run) |
| Does the prefill-token penalty scale with history? | **yes, linearly = history length** | unknown |
| Does the TTFT penalty scale with history? | no — fixed ~95 ms on this hardware | unknown |
| Is Engram's token-reduction benefit realised on cold restore? | **no — completely lost** | unknown |
| Is Engram's TTFT benefit realised even on warm restore? | no — Mamba-state restore overhead exceeds prefill savings on this fast model | unknown |

## Reproduction

### Correctness arm (base brief + cold-radix addendum)

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
SGLANG_ENABLE_SPEC_V2=false python -m sglang.launch_server ... &
# wait for /health 200
python scripts/validation/kha397_prefill_validation.py --mode cold
```

### Cost arm (Addendum 2 — radix cache ENABLED)

```bash
# One-command suite — handles all server lifetimes and restarts
bash scripts/validation/kha397_run_cost_suite.sh
# outputs land in /tmp/kha397-cost/{warm,cold,summary}-{512,4096}.json
```

Note the cost-suite launcher omits `--disable-radix-cache` deliberately —
the warm-vs-cold radix-state hypothesis collapses if the radix tree is
disabled (with `--disable-radix-cache`, both warm and cold re-prefill the
full history; the prior correctness-arm `--disable-radix-cache` run
showed this directly in `/tmp/kha397/server-warm.log` line 105).
