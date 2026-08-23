# Files

## Inspected

- `.tmp/nemo-upstream-replay/examples/configs/recipes/llm/grpo-glm5.1-64n8g-megatron.yaml` - native GLM-5.1 Megatron/vLLM baseline.
- `.tmp/nemo-upstream-replay/examples/configs/recipes/llm/performance/grpo-deepseek-v3-64n4g-async-1off.yaml` - closest 4-GPU-node large-MoE async-vLLM analog.
- `.tmp/nemo-upstream-replay/examples/configs/recipes/llm/performance/grpo-deepseek-v3-64n8g-async-1off.yaml` - parent non-colocated async resource layout.
- `/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.1/config.json` - actual GLM dimensions and MoE layout.
- `session/20260820_003344/{handoff.md,session_state.md,timeline.md}` - prior image/probe evidence and runtime caveats.

## Changed

- `.tmp/gh200-throughput-baseline/nemo_rl/utils/flops_tracker.py` - recognize the exact GH200 device string for MFU reporting.
- `.tmp/gh200-throughput-baseline/tests/unit/utils/test_flops_tracker.py` - cover GH200 BF16 and FP32/TF32 peak lookup.
- `.tmp/apertus-refit-fix/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/models/apertus/apertus_bridge.py` - stop exporting xIELU architecture constants as synthetic refit weights.
- `.tmp/apertus-refit-fix/nemo_rl/models/generation/vllm/patches.py` - keep vLLM xIELU constants engine-owned and validate legacy checkpoint values.
- `.tmp/apertus-refit-fix/nemo_rl/models/policy/lm_policy.py` - fail closed when worker refit manifests disagree for IPC or NCCL reshard.
- `.tmp/apertus-refit-fix/infra/slurm/cscs/probe_nemo_rl_grpo_vllm0251_image.slurm` - permanent configurable PP regression gate.
- `.tmp/apertus-refit-fix/infra/slurm/cscs/autoresearch/submit_apertus70b_async_smoke.sh` - opt-in exact-image source overlay for committed development gates.
- `.tmp/nemo-upstream-7ea-curated/` - curated upstream replay and canonical PR #24 head; merged to public `main` at `8f22e59195f5`.
- `.tmp/apertus70b-exact-cert/` - exact-image Apertus 70B fail-closed harness; published branch `autoresearch/2026-08-23-apertus70b-cert/exact-image` at `92db67ae0`.
- `.tmp/glm51-exact-prototype/` - exact-image offline GLM-5.1 TP32/EP32 harness; published branch `autoresearch/2026-08-23-glm51-prototype/exact-image` at `76217c755`.

## Generated

- `session/20260822_014118/` - durable campaign state and handoff.
- `.tmp/gh200-throughput-baseline/` - isolated shared MFU baseline worktree at commit `6bb1dfe49`.
- Experiment worktrees, configs, logs, and TSV ledgers will be recorded here after creation.
- `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.sqsh` - final exact-head release image, SHA-256 `4aaf2b1bba8613a1e515281d84ab9e330c41d2774ccd3992b5f0c0f81e9dd002`.
- `.tmp/apertus70b-exact-cert/logs/apertus70b_async_smoke/terminal_green_20260823T044937Z_235193.json` - terminal metrics evidence for all three 70B steps and refits.
