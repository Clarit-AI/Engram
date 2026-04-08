# Pure Mamba2 Model Support in SGLang

## Problem Statement

SGLang has comprehensive Mamba2 infrastructure — `MambaMixer2` SSM layers, `Mamba2AttnBackend`, `MambaPool` for state management, `MambaRadixCache` for prefix caching — but all of it was wired exclusively for **hybrid** models that combine Mamba layers with attention layers. Pure Mamba2 models like `mistralai/Mamba-Codestral-7B-v0.1` (`Mamba2ForCausalLM`) could not load at all.

SGLang upstream tracks this as [sgl-project/sglang#7429](https://github.com/sgl-project/sglang/issues/7429) with `Mamba2ForCausalLM` unchecked and no open PRs working on it.

### Failure Cascade

Four distinct blockers prevented pure Mamba2 models from loading:

1. **`architectures: None` crash** — Pure Mamba models may have `architectures: None` in their HF config, causing `IndexError` at 23 unguarded `architectures[0]` accesses in `model_config.py`
2. **`num_attention_heads` missing** — `_derive_model_shapes()` unconditionally reads `num_attention_heads`, which doesn't exist on pure SSM configs
3. **No model class** — No `Mamba2ForCausalLM` model implementation exists among SGLang's 100+ model files
4. **`mamba2_config` doesn't recognize pure Mamba** — The property only checks for hybrid config types (FalconH1Config, NemotronHConfig, etc.)
5. **Division-by-zero** — KV cache profiling computes `cell_size = 0` when `full_attention_layer_ids = []`, causing a crash

## Architecture Spectrum

| Architecture | Model | Mamba Layers | Attention Layers | Status |
|---|---|---|---|---|
| Dense hybrid | Granite 4.0-H-tiny | ~75% | ~25% | Working |
| MoE hybrid | Nemotron-Cascade-2-30B | Mixed | Mixed | Working |
| Non-Mamba recurrent (GLA) | Qwen3-Coder-Next | DeltaNet | Full | Working |
| **Pure SSM** | **Codestral Mamba 7B** | **100%** | **0%** | **This PR** |

## What Already Existed

The heavy lifting was already done by the hybrid model implementations:

| Component | File | Purpose |
|---|---|---|
| `MambaMixer2` | `layers/attention/mamba/mamba.py` | SSM layer (conv + selective scan) |
| `Mamba2AttnBackend` | `layers/attention/mamba/` | Backend for batched SSM forward |
| `MambaPool` | `mem_cache/memory_pool.py` | GPU state pool for conv/temporal states |
| `HybridReqToTokenPool` | `mem_cache/memory_pool.py` | Request-to-token mapping with mamba support |
| `MambaRadixCache` | `mem_cache/mamba_radix_cache.py` | Dual LRU tree for prefix caching |
| `Mamba2CacheParams` | `configs/mamba_utils.py` | State shape and dtype configuration |

All of these components are **architecture-agnostic** — they don't assume hybrid models. `MambaPool` allocates state for a list of layer IDs without caring whether attention layers exist. `MambaRadixCache` maintains separate `full_lru_list` and `mamba_lru_list` that can operate independently.

## What Was Missing

The gap was purely at the integration layer:

| Gap | Solution |
|---|---|
| No config class providing `full_attention_layer_ids`, `mamba2_cache_params` | `configs/mamba2.py` — `Mamba2Config` with `model_type = "mamba2"` |
| No model class wiring `MambaMixer2` for all layers | `models/mamba2.py` — `Mamba2ForCausalLM` following NemotronH pattern |
| `mamba2_config` property doesn't match pure Mamba | Added `isinstance(config, Mamba2Config)` check |
| `architectures[0]` crashes on empty list | Guards in `__init__`, `_config_draft_model`, `_derive_hybrid_model`, `is_deepseek_nsa` |
| Division-by-zero when `cell_size = 0` | Guard in `profile_max_num_token` |

## Implementation Approach

### Config: `Mamba2Config`

Inherits from `PretrainedConfig` with `model_type = "mamba2"`. This overrides transformers' built-in `Mamba2Config` via Python's class registration mechanism, ensuring `AutoConfig.from_pretrained()` returns our config.

Key properties:
- `full_attention_layer_ids` returns `[]` (no attention)
- `mamba_layer_ids` returns `list(range(num_hidden_layers))` (all layers are SSM)
- `mamba2_cache_params` builds `Mamba2CacheParams` from HF config fields

### Model: `Mamba2ForCausalLM`

Follows the NemotronH pattern with attention/MLP/MoE stripped out:

```text
Mamba2ForCausalLM
  +-- Mamba2Model
  |     +-- embed_tokens (VocabParallelEmbedding)
  |     +-- layers[] (Mamba2DecoderLayer x N)
  |     |     +-- norm (RMSNorm)
  |     |     +-- mixer (MambaMixer2)
  |     +-- final_layernorm (RMSNorm)
  +-- lm_head (ParallelLMHead)
```

Each `Mamba2DecoderLayer` does: `RMSNorm -> MambaMixer2 -> residual`. No attention, no MLP, no conditional branching.

### Weight Mapping

HF Codestral checkpoints use `backbone.*` prefix; SGLang uses `model.*`. The `A_log` parameter is mapped to `A` with the transformation `A = -exp(A_log)` handled by `MambaMixer2`'s existing weight loader.

```text
backbone.embeddings.weight     -> model.embed_tokens.weight
backbone.layers.{i}.mixer.*    -> model.layers.{i}.mixer.*
backbone.layers.{i}.norm.*     -> model.layers.{i}.norm.*
backbone.norm_f.weight         -> model.final_layernorm.weight
lm_head.weight                 -> lm_head.weight
```

### KV Cache Profiling Fix

When `full_attention_layer_ids = []`, the KV `cell_size` computes to 0 (no attention heads, no KV cache needed). This caused a division-by-zero in `profile_max_num_token`. Fix: when `cell_size == 0`, detect this as a special SSM-only case and compute token capacity using a bounded estimate based on the model's context length and available memory after Mamba state allocation, rather than relying on the KV-based calculation. This ensures `handle_max_mamba_cache()` properly constrains request capacity.

## Upstream Contribution Path

This implementation can be contributed upstream to `sgl-project/sglang`:

1. **`configs/mamba2.py`** — Pure Mamba2 config class (upstream has none)
2. **`models/mamba2.py`** — Model class for `Mamba2ForCausalLM` architecture
3. **`model_runner.py`** — `mamba2_config` property extension
4. **`model_config.py`** — `architectures` guard (fixes a latent bug for any model with empty architectures)
5. **`model_runner_kv_cache_mixin.py`** — Division-by-zero guard (fixes a latent bug for any pure-SSM model)

Items 4 and 5 are bug fixes that benefit all users, not just Mamba2.

## Testing

### Validation Gates

| Gate | Test | Status |
|---|---|---|
| 1 | Server starts with Codestral Mamba 7B | Pending (GPU) |
| 2 | Single inference produces coherent output | Pending (GPU) |
| 3 | Snapshot save/load round-trip | Pending (GPU) |
| 4 | Multi-turn state restoration | Pending (GPU) |
| 5 | Existing hybrid models still work (regression) | Pending (GPU) |

### Compatibility Matrix

| Model | Architecture | Expected Result |
|---|---|---|
| mistralai/Mamba-Codestral-7B-v0.1 | Mamba2ForCausalLM (pure SSM) | New support |
| ibm-granite/granite-4.0-h-tiny | GraniteHybrid (dense hybrid) | No regression |
| NVIDIA Nemotron-Cascade | NemotronH (MoE hybrid) | No regression |
| Qwen3-Coder-Next | DeltaNet (non-Mamba recurrent) | No regression |

## Files Changed

| File | Action | Lines |
|---|---|---|
| `python/sglang/srt/configs/mamba2.py` | NEW | ~130 — pure Mamba2 config |
| `python/sglang/srt/models/mamba2.py` | NEW | ~270 — Mamba2ForCausalLM model class |
| `python/sglang/srt/configs/model_config.py` | MODIFIED | Guard `architectures[0]` accesses |
| `python/sglang/srt/configs/__init__.py` | MODIFIED | Export Mamba2Config |
| `python/sglang/srt/model_executor/model_runner.py` | MODIFIED | Recognize Mamba2Config |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | MODIFIED | Fix cell_size=0 |
