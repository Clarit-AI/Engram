Investigate and fix the Mamba model configuration bug in this SGLang Mamba
  fork.

  ## Bug Description
  When trying to start the server with a Mamba model (e.g.,
  `state-spaces/mamba-2.8b`), the server crashes with:
  TypeError: 'NoneType' object is not subscriptable
  at `python/sglang/srt/configs/model_config.py` line 149:
  ```python
  if self.hf_config.architectures[0] in mm_disabled_models:
  ```

  The root cause is that Mamba models have `architectures: None` in their
  HuggingFace config, so accessing `config.architectures[0]` fails.

  Where to Look

  1. python/sglang/srt/configs/model_config.py - line 149 (the crash point)
  2. The error occurs during prepare_server_args() → use_mla_backend() →
  get_model_config()
  3. Mamba models from state-spaces/mamba-2.8b have architectures: None

  Context

  - This is a fork of SGLang with Mamba snapshot persistence added
  - The fork: /Users/bbrenner/sglang-mamba (or clone at
  /home/bbrenner/sglang-mamba on gcloud instance)
  - Mamba models should be supported since this fork's whole purpose is Mamba
  SSM state management
  - See CLAUDE.md and phase3/ plans for more context

  What to Do

  1. Read model_config.py around line 149 and trace back where the error
  originates
  2. Find why Mamba models are hitting this code path (is_mamba check should
  bypass this?)
  3. Fix the bug so Mamba models can be loaded properly
  4. Verify the fix allows server startup: python -m sglang.launch_server
  --model-path state-spaces/mamba-2.8b --port 30000
  5. Run tests: pytest test/sglang/snapshot/ -v

  Expected Outcome

  Server starts successfully with a Mamba model and can accept inference
  requests.

  ---

  **To run on gcloud instance directly:**
  ```bash
  gcloud compute ssh --zone "asia-east1-c" "sglang-test-v100-20260325-230245"
  --project "gen-lang-client-0471830999"
  ```

  Then run the investigation from /home/bbrenner/sglang-mamba/.