# KHA-387 SSM Ablation Report — 2026-05-21T01:18:50Z

Server: http://localhost:30200  Model name: mamba-codestral-7b
Snapshot dir: /workspace/runs/kha-387-ssm-ablation-20260521T011632Z/snapshots
Run dir: /workspace/runs/kha-387-ssm-ablation-20260521T011632Z

## Per-case continuations

| Case | save_ok | restore_ok | sentinel? | first 16 output ids | text head |
|---|---|---|---|---|---|
| no_restore | True | n/a | no | `[29502, 29513, 781, 29520, 781, 781, 781, 18269, 29574, 29473, 29508, 29502, 29502, 29502, 29501, 29508]` | `0;\n}\n\n\n+++++ 1000-1999/1001/1001.cpp\n#include <iostream>\n\nusing namespace std;\n\n` |
| real_warm | True | ok | no | `[29502, 29513, 781, 29520, 781, 781, 781, 18269, 29574, 29473, 29508, 29502, 29502, 29502, 29501, 29508]` | `0;\n}\n\n\n+++++ 1000-1999/1001/1001.cpp\n#include <iostream>\n\nusing namespace std;\n\n` |
| real_disk | True | ok | no | `[29502, 29513, 781, 29520, 781, 781, 781, 18269, 29574, 29473, 29508, 29502, 29502, 29502, 29501, 29508]` | `0;\n}\n\n\n+++++ 1000-1999/1001/1001.cpp\n#include <iostream>\n\nusing namespace std;\n\n` |
| zero_disk | True | ok | no | `[29502, 29513, 781, 29520, 781, 781, 781, 18269, 29574, 29473, 29508, 29502, 29502, 29502, 29501, 29508]` | `0;\n}\n\n\n+++++ 1000-1999/1001/1001.cpp\n#include <iostream>\n\nusing namespace std;\n\n` |
| random_disk | True | ok | no | `[29502, 29513, 781, 29520, 781, 781, 781, 18269, 29574, 29473, 29508, 29502, 29502, 29502, 29501, 29508]` | `0;\n}\n\n\n+++++ 1000-1999/1001/1001.cpp\n#include <iostream>\n\nusing namespace std;\n\n` |

## Summary

- all_same_prefix: **True**
- real_warm ≡ no_restore: **True**
- real_disk ≡ real_warm: **True**
- zero_disk ≡ random_disk: **True**
- distinct prefixes: **1**

**Verdict:** Case 1 — restored SSM not consumed. All five conditions produced the same continuation prefix. The pending-restore + next-/generate path is not using the injected recurrent state. Bug is in request lifecycle, not in attention-KV scope.

Server-log evidence for restoration events should be cross-checked from
the server.log written alongside this run; this harness records HTTP-
level evidence only.
