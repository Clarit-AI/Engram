# Handoff

## State
PR #71 (upstream-sync-20260515, ~509 commits) validated on H100 + merged. Phase 3b primary signals green: pytest gate 76/77, Phase 9 gauntlet 11/12, validate-sync Tests 1/2/3/5 (Test 4 was model-behavior, not snapshot). Two prebuilt-wheel publishing lags fixed mid-session — `sgl_kernel 0.4.2.post2` and `sgl_deep_gemm 2.5.0+714dd1a` rebuilt + uploaded to `s3://engram/wheels/` (version-pinned + `-latest-` aliases). Env freeze + GCP image `engram-v0-4-0-h100-cu128` (family `engram-h100-cu128`, project `engram-dev-493301`) durable. CI fixes for F1-F4 (model_config.py retired marker, k8s-integration fork guard in pr-test-rust.yml, bare-guard replacement in 9 unguarded AMD/ROCm720 jobs, 6 duplicate-if jobs cleaned) all on the merged branch.

## Next
1. Spin up a fresh GCP instance from image `engram-v0-4-0-h100-cu128` and run the 4-step restore (RAID local SSDs → `/workspace`, clone Engram, `pip install --user -e python`, `hf download granite-4.0-h-tiny`) to prove the image actually restores in ~7-10 min. Until that's verified, the image is unproven.
2. Focused follow-up PR for the four documented Engram-side issues from the Phase 3b comment: validate-sync.sh needs `--mamba-scheduler-strategy extra_buffer` (or the Spec-V2 guard needs to check `speculative_algorithm`), jinja2 floor `>=3.1.0` in pyproject, `post_forward_snapshot_callback` NoneType bug at `scheduler.py:1315`, and possibly the 65 cosmetic-but-dead `|| github.event_name == 'pull_request'` clauses in nightly AMD workflows (functionally inert under their outer fork guards — overseer didn't say to clean these in PR #71).
3. Tear down instance `engram-upstream-merge-2025-05-19` in `us-east4-a` to stop the $4/hr spot burn once #1 above is decided.

## Context
- GCP **does not** support machine-images for ACCELERATOR_OPTIMIZED GEN_3 (H100) instances — use disk snapshot + custom image instead. Don't waste time on `gcloud compute machine-images create`.
- `sglang` is editable-installed pointing at `/workspace/Engram/python` (local SSD, ephemeral). A boot-disk image alone won't make `import sglang` work after restore; the restore sequence has to re-clone Engram to the same path and re-run `pip install -e` to land the Rust gRPC extension at `python/sglang/srt/grpc/_core.cpython-310-x86_64-linux-gnu.so`.
- `sgl-kernel` builds OOM at `MAX_JOBS=26` (230GB RAM but nvcc with `--threads=32` × 26 jobs blows past the kernel). **Use `MAX_JOBS=8`** — proven safe (88GB peak), takes ~2.5h for SM90-only build with `CMAKE_ARGS="-DENABLE_BELOW_SM90=OFF"`. Prior session's CMakeLists patch is obsolete; native flag works post `0fde6153`.
- Granite 4.0-H-tiny has **no valid Spec-V2-on config** — `memory_pool.py:514` guard requires `extra_buffer`, but `server_args.py:2539` says GraniteMoeHybridForCausalLM doesn't support it. Both guards from commit `3eda9801` (2026-03-30). Use `SGLANG_ENABLE_SPEC_V2=false` env override (no source edit) to run anything mamba on Granite.
- PyPI's `sgl-deep-gemm==0.1.0` ships a **CUDA-13-linked `_C.so`** — incompatible with our cu128 baseline. Always pull from `s3://engram/wheels/sgl_deep_gemm-engram-latest-cu128-sm90.whl` on this instance class.
- jinja2 must be `>=3.1.0` for chat templates; system image shipped 3.0.3 which silently broke `/v1/chat/completions` on first attempt.
- Hard rule from this session: don't auto-mark PR ready-for-review on partial success. Test 4 model behavior + impossible Spec-V2-on supplementary check needed overseer judgement — overseer made the call after the comment, I didn't.
