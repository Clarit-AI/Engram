# Engram HTTP API Spec

Canonical technical draft for Engram's snapshot and statefulness HTTP surface.

This document covers the additional snapshot endpoints implemented in this
repository on top of SGLang's standard serving APIs. This specification and
api_guide.md complement each other for Engram-specific HTTP integration.

The standard SGLang APIs such as `/v1/chat/completions` are documented
elsewhere. This document focuses only on the snapshot/statefulness layer added
here.

## Scope

Implemented endpoints:

- `POST /save_snapshot`
- `POST /list_snapshots`
- `POST /get_snapshot_info`
- `POST /restore_snapshot`
- `POST /delete_snapshot`

All five routes are `ADMIN_OPTIONAL`.

## Server Prerequisites

The snapshot API only applies when the server is started with snapshot
persistence enabled.

Minimum launch shape:

```bash
python -m sglang.launch_server \
  --model-path /workspace/models/granite-4.0-h-small \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-snapshot-persistence \
  --snapshot-dir /workspace/snapshots \
  --trust-remote-code
```

Recommended remote-access shape:

```bash
python -m sglang.launch_server \
  --model-path /workspace/models/granite-4.0-h-small \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-snapshot-persistence \
  --snapshot-dir /workspace/snapshots \
  --admin-api-key "$SGLANG_ADMIN_KEY" \
  --served-model-name granite-4.0-h-small \
  --trust-remote-code
```

Optional Mamba-specific tuning:

```bash
python -m sglang.launch_server \
  --model-path /workspace/models/Mamba-Codestral-7B-v0.1 \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-snapshot-persistence \
  --snapshot-dir /workspace/snapshots \
  --mamba-scheduler-strategy no_buffer \
  --trust-remote-code
```

Notes:

- `--mamba-scheduler-strategy` defaults to `auto`.
- `auto` currently resolves to `no_buffer`.
- `extra_buffer` should only be forced on models and backends that support it.
- Chat-style clients should use the same tokenizer and chat template as the
  server when producing `continuation_ids` for stateful generation.

## Transport

- Base URL: `http://<host>:<port>`
- Method: every snapshot endpoint is `POST`
- Content type: `application/json`
- Response format: JSON

Example base URL:

```text
http://localhost:30000
```

## Authentication

Snapshot routes use the same `ADMIN_OPTIONAL` behavior as other admin-optional
server routes.

Effective auth rules:

- if neither `api_key` nor `admin_api_key` is configured, requests are allowed
  without auth
- if only `api_key` is configured, snapshot routes require that token
- if only `admin_api_key` is configured, snapshot routes require that token
- if both are configured, snapshot routes require `admin_api_key`

Header format:

```http
Authorization: Bearer <token>
```

## Core Concepts

### `rid` vs `conversation_id`

These endpoints use two identifiers that solve different problems:

- `rid` is a live request identifier
- `conversation_id` is the stable snapshot grouping key used on disk and in
  metadata

For restore-only flows, this distinction matters:

- `rid` is what allows the scheduler to inject restored state into a live
  request
- `conversation_id` is what allows the server to locate snapshot files

Using only `conversation_id` in restore-only mode can prove that state exists,
but may not restore that state into a live request.

### Snapshot Selection

Snapshot files are selected by:

- required `conversation_id` for list/info/delete
- optional `turn_number`
- optional `branch_name`

In practice, one of `turn_number` or `branch_name` is effectively required for
`/get_snapshot_info` and `/delete_snapshot`, even though the request dataclasses
type them as optional.

### Stateful Generation

`POST /restore_snapshot` supports two distinct modes:

1. Restore-only:
   `create_new_request=false` or omitted
2. Restore-and-generate:
   `create_new_request=true`

Restore-and-generate requires:

- `continuation_ids`
- `max_new_tokens`

When restore-and-generate succeeds, the response can include:

- `output_ids`
- `output_text`

### Tokenization for `continuation_ids`

For chat-style clients, `continuation_ids` must be encoded with the same
tokenizer and prompt serialization strategy used by the server.

Example:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/workspace/models/granite-4.0-h-small")
messages = [{"role": "user", "content": "What is my favorite color?"}]
formatted = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
continuation_ids = tok.encode(formatted, add_special_tokens=False)
```

## Endpoint Reference

### `POST /save_snapshot`

Save snapshot state for a live request or a request that has already fallen back
to the WARM tier.

#### Request Body

```json
{
  "rid": "req-123",
  "snapshot_id": "optional-custom-id",
  "conversation_id": "chat-123",
  "turn_number": 2,
  "branch_name": "main"
}
```

Fields:

- `rid`: optional string in the request model, but this is the real live
  request lookup key
- `snapshot_id`: optional string
- `conversation_id`: optional string used as the snapshot grouping key
- `turn_number`: optional integer in the request model
- `branch_name`: optional string in the request model

Behavior notes:

- saving a live active request really depends on `rid`
- `conversation_id` alone does not identify a live request
- saving without `turn_number` and `branch_name` only reliably works in the
  WARM-tier fallback path where metadata is reconstructed for you
- the lower storage layer effectively needs one of `turn_number` or
  `branch_name`

#### Success Response

```json
{
  "success": true,
  "snapshot_id": "snap-abc",
  "message": "Snapshot saved"
}
```

#### Failure Response

```json
{
  "success": false,
  "message": "Error saving snapshot: <details>"
}
```

Important:

- handler-level save failures still return HTTP `200`
- clients must inspect `success`, not just the status code

### `POST /list_snapshots`

List snapshot metadata for a conversation.

#### Request Body

```json
{
  "conversation_id": "chat-123"
}
```

Fields:

- `conversation_id`: required string

#### Success Response

```json
{
  "success": true,
  "snapshots": [
    {
      "conversation_id": "chat-123",
      "turn_number": 2,
      "branch_name": null,
      "timestamp": 1712345678.0,
      "token_count": 128,
      "model_name": "/workspace/models/granite-4.0-h-small"
    }
  ]
}
```

Notes:

- each item is a metadata dictionary emitted by the snapshot layer
- the exact metadata shape is implementation-defined and may include additional
  fields such as `mamba_pool_idx`, `req_pool_idx`, `layer_config`, or `fill_ids`

#### Empty Conversation Response

```json
{
  "success": true,
  "snapshots": []
}
```

Important:

- missing or empty conversations do not produce an application error
- the current behavior is HTTP `200` with an empty list

### `POST /get_snapshot_info`

Fetch metadata for a specific snapshot selection inside a conversation.

#### Request Body

```json
{
  "conversation_id": "chat-123",
  "turn_number": 2
}
```

Fields:

- `conversation_id`: required string
- `turn_number`: optional integer in the request model
- `branch_name`: optional string in the request model

Behavior notes:

- the request dataclass allows both `turn_number` and `branch_name` to be
  omitted
- the lower snapshot path logic effectively requires one of them

#### Success Response

```json
{
  "success": true,
  "metadata": {
    "conversation_id": "chat-123",
    "turn_number": 2,
    "branch_name": null,
    "timestamp": 1712345678.0,
    "token_count": 128,
    "model_name": "/workspace/models/granite-4.0-h-small"
  }
}
```

#### Failure Response

```json
{
  "success": false,
  "message": "Snapshot not found: conversation=chat-123, turn=2, branch=None"
}
```

### `POST /restore_snapshot`

Restore Mamba state from a saved snapshot.

This is the most important endpoint for Engram-style stateful clients.

#### Restore-Only Request

```json
{
  "rid": "req-123",
  "conversation_id": "chat-123",
  "turn_number": 2
}
```

#### Restore-And-Generate Request

```json
{
  "conversation_id": "chat-123",
  "create_new_request": true,
  "continuation_ids": [1, 2, 3, 4],
  "max_new_tokens": 80
}
```

Fields:

- `rid`: optional string
- `conversation_id`: optional string
- `turn_number`: optional integer
- `branch_name`: optional string
- `create_new_request`: optional boolean, default `false`
- `continuation_ids`: optional list of ints
- `max_new_tokens`: optional int
- `request_id`: optional correlation id; auto-generated if omitted

Validation rules enforced by the request dataclass:

- at least one of `rid` or `conversation_id` must be provided
- if `create_new_request=false`, `continuation_ids` and `max_new_tokens` must
  not be supplied
- if `create_new_request=true`, both `continuation_ids` and `max_new_tokens`
  are required

#### Restore-Only Behavior

Restore-only is best understood as a live-request operation.

- if a live `rid` is found, the scheduler can inject restored state into that
  request
- if only `conversation_id` is provided, or the `rid` cannot be resolved to a
  live request, the server may report that snapshot state is available for
  future requests instead of performing an in-place restore

#### Restore-And-Generate Behavior

When `create_new_request=true`:

- if `turn_number` and `branch_name` are both omitted, the latest snapshot is
  selected
- the scheduler enforces model compatibility
- `fill_ids` must exist in the snapshot metadata
- the scheduler checks Mamba pool availability and context-length constraints
- the response can include decoded `output_text`

#### Success Response

```json
{
  "success": true,
  "rid": "restored-req-456",
  "mamba_pool_idx": 7,
  "message": "Snapshot restored successfully",
  "token_count": 128,
  "output_ids": [101, 202, 303],
  "output_text": "Your favorite color is blue."
}
```

#### Failure Response

```json
{
  "success": false,
  "rid": null,
  "mamba_pool_idx": null,
  "message": "Snapshot not found for conversation=chat-123",
  "token_count": null,
  "output_ids": null,
  "output_text": null
}
```

Important:

- application-level restore failures are returned as HTTP `500`
- clients should always parse the JSON body before treating a restore failure as
  a generic transport error

### `POST /delete_snapshot`

Delete one snapshot selection within a conversation.

#### Request Body

```json
{
  "conversation_id": "chat-123",
  "turn_number": 2
}
```

Fields:

- `conversation_id`: required string
- `turn_number`: optional integer in the request model
- `branch_name`: optional string in the request model

Behavior notes:

- in practice the lower storage layer expects one of `turn_number` or
  `branch_name`

#### Success Response

```json
{
  "success": true,
  "message": "Snapshot deleted"
}
```

#### Failure Response

```json
{
  "success": false,
  "message": "Snapshot not found"
}
```

## Client Examples

### Python `requests`

```python
import requests

BASE_URL = "http://localhost:30000"
ADMIN_KEY = "replace-me"

session = requests.Session()
session.headers.update(
    {
        "Authorization": f"Bearer {ADMIN_KEY}",
        "Content-Type": "application/json",
    }
)

save_resp = session.post(
    f"{BASE_URL}/save_snapshot",
    json={
        "rid": "req-123",
        "conversation_id": "chat-123",
        "turn_number": 2,
    },
    timeout=30,
)
save_body = save_resp.json()
if not save_body.get("success"):
    raise RuntimeError(save_body)
```

### Python Restore-And-Generate

```python
import requests
from transformers import AutoTokenizer

BASE_URL = "http://localhost:30000"
MODEL_PATH = "/workspace/models/granite-4.0-h-small"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
formatted = tok.apply_chat_template(
    [{"role": "user", "content": "What is my favorite color?"}],
    tokenize=False,
    add_generation_prompt=True,
)
continuation_ids = tok.encode(formatted, add_special_tokens=False)

resp = requests.post(
    f"{BASE_URL}/restore_snapshot",
    json={
        "conversation_id": "chat-123",
        "create_new_request": True,
        "continuation_ids": continuation_ids,
        "max_new_tokens": 64,
    },
    timeout=120,
)
body = resp.json()
if not body.get("success"):
    raise RuntimeError(body)
print(body.get("output_text"))
```

### JavaScript / TypeScript

```ts
const baseUrl = "http://localhost:30000";
const adminKey = process.env.SGLANG_ADMIN_KEY;

const headers: Record<string, string> = {
  "Content-Type": "application/json",
};

if (adminKey) {
  headers["Authorization"] = `Bearer ${adminKey}`;
}

const response = await fetch(`${baseUrl}/list_snapshots`, {
  method: "POST",
  headers,
  body: JSON.stringify({ conversation_id: "chat-123" }),
});

const body = await response.json();
if (!body.success) {
  throw new Error(body.message ?? "snapshot list failed");
}
```

## Recommended Client Workflow

### Pattern A: Standard Chat, Then Snapshot

1. Generate normally through `/v1/chat/completions`.
2. Save a snapshot using the original `rid`.
3. Keep your own `conversation_id`, `turn_number`, and `branch_name` mapping.
4. Use `/list_snapshots` and `/get_snapshot_info` for inspection.
5. Use `/restore_snapshot` in restore-and-generate mode for later turns.

### Pattern B: Restore-And-Generate

1. Save snapshots at important turn boundaries.
2. Tokenize only the new user turn as `continuation_ids`.
3. Call `/restore_snapshot` with `create_new_request=true`.
4. Read `output_text` and the returned `rid`.

## Known Contract Oddities

- `/save_snapshot` can fail at the application layer while still returning HTTP
  `200`.
- `/restore_snapshot` maps application-level failure to HTTP `500`.
- some fields are typed optional in the request dataclasses but are effectively
  required by lower storage-layer behavior
- direct HTTP coverage is strongest for `/save_snapshot` and
  `/restore_snapshot`; `/list_snapshots`, `/get_snapshot_info`, and
  `/delete_snapshot` are more code-confirmed than end-to-end tested

## Validated Against

Primary implementation sources:

- `python/sglang/srt/entrypoints/http_server.py`
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/lang/interpreter.py`
- `python/sglang/snapshot.py`

Tests and validation references:

- `test/registered/radix_cache/test_mamba_snapshot_e2e.py`
- `test/registered/radix_cache/test_mamba_stateful_inference.py`
- `test/sglang/snapshot/test_mamba_snapshot.py`