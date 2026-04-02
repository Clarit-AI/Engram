# Stateful Mamba HTTP API Specification

Code-first HTTP specification for the snapshot endpoints implemented in the
SGLang runtime server.

This document describes the server-facing API exposed by
`python/sglang/srt/entrypoints/http_server.py` and the request/response
dataclasses defined in `python/sglang/srt/managers/io_struct.py`.

It is intentionally separate from
[`api_reference.md`](api_reference.md), which focuses on the frontend Python
snapshot API.

## Scope

These endpoints are specific to Mamba-capable SGLang servers with snapshot
persistence enabled.

Implemented endpoints:

- `POST /save_snapshot`
- `POST /list_snapshots`
- `POST /get_snapshot_info`
- `POST /restore_snapshot`
- `POST /delete_snapshot`

## Server Prerequisites

The snapshot API is only useful when the server is started with snapshot
persistence enabled.

Minimum server flags:

```bash
python -m sglang.launch_server \
  --model-path /workspace/models/granite-4.0-h-small \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-snapshot-persistence \
  --snapshot-dir /workspace/snapshots \
  --trust-remote-code
```

Recommended remote-access configuration:

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
- `auto` currently resolves to `no_buffer` in server args.
- `extra_buffer` should only be forced on models and backends that support it.
- `--chat-template` or `--hf-chat-template-name` may be needed for chat-style
  clients so the client tokenizer matches the server's prompt serialization.

## Base URL And Transport

- Base URL: `http://<host>:<port>`
- Method: all snapshot endpoints are `POST`
- Content type: `application/json`
- Response format: JSON

Example base URL:

```text
http://localhost:30000
```

## Authentication

Snapshot endpoints are marked `ADMIN_OPTIONAL`.

Effective auth behavior:

- if neither `api_key` nor `admin_api_key` is configured, requests are allowed
  without auth
- if only `api_key` is configured, clients must send that bearer token
- if only `admin_api_key` is configured, clients must send that bearer token
- if both are configured, snapshot endpoints require `admin_api_key`

Header format:

```http
Authorization: Bearer <token>
```

Example `curl` header:

```bash
curl -H "Authorization: Bearer $SGLANG_ADMIN_KEY" ...
```

## Request Conventions

### `rid` vs `conversation_id`

Several snapshot operations accept either a request id (`rid`) or a
conversation id (`conversation_id`).

- `rid` is the live request identifier inherited from `BaseReq`
- `conversation_id` is the stable conversation identifier used to group
  snapshots across turns

For `restore_snapshot`, the server validates that at least one of these is
provided.

### Stateful Generation Mode

`POST /restore_snapshot` has two modes:

1. Restore-only mode:
   `create_new_request=false` or omitted. This restores prior state without
   appending a new continuation.
2. Restore-and-generate mode:
   `create_new_request=true`. This requires both:
   - `continuation_ids`
   - `max_new_tokens`

When `continuation_ids` are supplied, the server blocks until generation
completes and returns `output_ids` and `output_text`.

### Tokenization Requirement For `continuation_ids`

For chat-style clients, `continuation_ids` must match the token format used to
create the original request context. In practice, that means using the same
tokenizer and chat template as the server and formatting the next user turn
with `add_generation_prompt=True`.

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

Do not hand-embed literal `User:` / `Assistant:` markers unless that is also how
the original prompt was serialized.

## Response Conventions

### Common Fields

Most endpoints return:

- `success`: boolean
- `message`: optional human-readable status or error string

### HTTP Status Codes

Current route behavior is:

- `200 OK` for successful `save_snapshot`, `list_snapshots`,
  `get_snapshot_info`, and `delete_snapshot`
- `400 Bad Request` for application-level validation failures in
  `list_snapshots`, `get_snapshot_info`, and `delete_snapshot`
- `500 Internal Server Error` for unhandled server-side failures
- `500 Internal Server Error` for `restore_snapshot` when
  `result.success == false`

Important:

- `restore_snapshot` may return a useful JSON body even when the HTTP status is
  `500`
- clients should parse the JSON body before treating non-`200` as an opaque
  transport error

## Endpoint Specification

### `POST /save_snapshot`

Save a snapshot for a live request or conversation.

#### Request Body

```json
{
  "rid": "req-123",
  "snapshot_id": "optional-custom-snapshot-id",
  "conversation_id": "chat-123",
  "turn_number": 2,
  "branch_name": "main"
}
```

Fields:

- `rid`: optional string inherited from `BaseReq`
- `snapshot_id`: optional string
- `conversation_id`: optional string
- `turn_number`: optional integer
- `branch_name`: optional string

#### Success Response

```json
{
  "success": true,
  "snapshot_id": "snap-abc",
  "message": "Snapshot saved"
}
```

#### Error Response

```json
{
  "success": false,
  "message": "Error saving snapshot: <details>"
}
```

### `POST /list_snapshots`

List all snapshots associated with a conversation.

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
      "snapshot_id": "snap-abc",
      "conversation_id": "chat-123",
      "turn_number": 2,
      "branch_name": "main"
    }
  ]
}
```

#### Error Response

```json
{
  "success": false,
  "message": "No snapshots found for conversation"
}
```

`snapshots` is returned as a list of metadata dictionaries. The exact keys are
determined by the server-side metadata implementation.

### `POST /get_snapshot_info`

Fetch metadata for a specific snapshot selection within a conversation.

#### Request Body

```json
{
  "conversation_id": "chat-123",
  "turn_number": 2,
  "branch_name": "main"
}
```

Fields:

- `conversation_id`: required string
- `turn_number`: optional integer
- `branch_name`: optional string

#### Success Response

```json
{
  "success": true,
  "metadata": {
    "snapshot_id": "snap-abc",
    "conversation_id": "chat-123",
    "turn_number": 2,
    "branch_name": "main"
  }
}
```

#### Error Response

```json
{
  "success": false,
  "message": "Snapshot not found"
}
```

### `POST /restore_snapshot`

Restore Mamba state from a previously saved snapshot.

This is the most important endpoint for stateful clients.

#### Restore-Only Request

```json
{
  "conversation_id": "chat-123",
  "turn_number": 2,
  "branch_name": "main"
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
- `request_id`: optional correlation id; auto-generated by the server-side input
  dataclass if omitted

Validation rules enforced by `RestoreSnapshotReqInput`:

- at least one of `rid` or `conversation_id` must be provided
- if `create_new_request=false`, `continuation_ids` and `max_new_tokens` must
  not be supplied
- if `create_new_request=true`, both `continuation_ids` and `max_new_tokens`
  are required

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

Response fields:

- `success`: boolean
- `rid`: request id for the restored or newly created request
- `mamba_pool_idx`: optional internal pool slot index
- `message`: optional status string
- `token_count`: optional integer count associated with restored state
- `output_ids`: optional generated token ids for restore-and-generate mode
- `output_text`: optional decoded generated text for restore-and-generate mode

#### Error Response

Example application-level failure:

```json
{
  "success": false,
  "rid": null,
  "mamba_pool_idx": null,
  "message": "Snapshot not found for conversation_id=chat-123",
  "token_count": null,
  "output_ids": null,
  "output_text": null
}
```

Client guidance:

- always parse the JSON body
- then inspect `success`
- do not rely on `raise_for_status()` alone for debugging restore failures

### `POST /delete_snapshot`

Delete a snapshot selection within a conversation.

#### Request Body

```json
{
  "conversation_id": "chat-123",
  "turn_number": 2,
  "branch_name": "main"
}
```

Fields:

- `conversation_id`: required string
- `turn_number`: optional integer
- `branch_name`: optional string

#### Success Response

```json
{
  "success": true,
  "message": "Snapshot deleted"
}
```

#### Error Response

```json
{
  "success": false,
  "message": "Snapshot not found"
}
```

## Client Configuration Examples

### 1. Python `requests` Client

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
```

Save a snapshot:

```python
save_resp = session.post(
    f"{BASE_URL}/save_snapshot",
    json={"rid": "req-123"},
    timeout=30,
)
save_resp.raise_for_status()
save_body = save_resp.json()
```

Restore and generate:

```python
restore_resp = session.post(
    f"{BASE_URL}/restore_snapshot",
    json={
        "conversation_id": "chat-123",
        "create_new_request": True,
        "continuation_ids": continuation_ids,
        "max_new_tokens": 80,
    },
    timeout=120,
)

restore_body = restore_resp.json()
if not restore_body.get("success"):
    raise RuntimeError(f"restore failed: {restore_body}")
```

### 2. Python Client Using A Chat Template

This is the recommended pattern for chat-style Mamba clients.

```python
import requests
from transformers import AutoTokenizer

BASE_URL = "http://localhost:30000"
MODEL_PATH = "/workspace/models/granite-4.0-h-small"
tok = AutoTokenizer.from_pretrained(MODEL_PATH)

def encode_continuation(messages):
    formatted = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tok.encode(formatted, add_special_tokens=False)

continuation_ids = encode_continuation(
    [{"role": "user", "content": "What is my favorite color?"}]
)

r = requests.post(
    f"{BASE_URL}/restore_snapshot",
    json={
        "conversation_id": "chat-123",
        "create_new_request": True,
        "continuation_ids": continuation_ids,
        "max_new_tokens": 64,
    },
    timeout=120,
)
body = r.json()
print(body["output_text"])
```

### 3. JavaScript / TypeScript Client

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

### 4. Example Client Configuration Object

```yaml
engram:
  base_url: http://mamba-host:30000
  auth:
    bearer_token_env: SGLANG_ADMIN_KEY
  snapshot_api:
    save_path: /save_snapshot
    restore_path: /restore_snapshot
    list_path: /list_snapshots
    info_path: /get_snapshot_info
    delete_path: /delete_snapshot
  timeouts:
    save_seconds: 30
    restore_seconds: 120
  tokenizer:
    model_path: /workspace/models/granite-4.0-h-small
    use_chat_template: true
    add_generation_prompt: true
```

## Recommended Client Workflow

### Pattern A: Full Chat First, Stateful Continuations After

1. Send turn 1 using `/v1/chat/completions`.
2. Save a snapshot with `/save_snapshot` using the original `rid`.
3. For later turns, tokenize only the new turn into `continuation_ids`.
4. Call `/restore_snapshot` with `create_new_request=true`.
5. Read `output_text` from the response.

### Pattern B: Snapshot Catalog Management

1. Save snapshots after important turns.
2. List snapshots by `conversation_id`.
3. Inspect metadata before restore.
4. Delete old branches or stale turns when no longer needed.

## Error Handling Recommendations

- Parse the JSON response body even for `500` responses from
  `/restore_snapshot`.
- Treat `success=false` as the authoritative application-level failure signal.
- Log both HTTP status and response JSON.
- Use longer client timeouts for restore-and-generate flows than for metadata
  operations.

## Current Limitations

- There is no published OpenAPI or Swagger document yet; this spec is derived
  from the implemented code.
- `snapshots` and `metadata` payload shapes are partially implementation-defined
  dictionaries rather than strict public schemas.
- `mamba_pool_idx` is returned today but should be treated as diagnostic data,
  not a stable client contract.

## Source Of Truth

When this document and code diverge, the current source of truth is:

- `python/sglang/srt/entrypoints/http_server.py`
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
