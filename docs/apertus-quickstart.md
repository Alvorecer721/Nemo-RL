# Apertus 1.5 8B on NeMo-RL: clone-and-run (CSCS GH200)

How to reproduce online GRPO post-training for Apertus 1.5 8B on a CSCS GH200 node from a clean checkout.
This branch runs on the **stock `nvcr.io/nvidia/nemo-rl:v0.6.0` image** via `uv run --locked` — no custom image build.
For the architecture gotchas behind the gates here, see [apertus-traps-and-invariants.md](apertus-traps-and-invariants.md); for Slurm submission details, see [infra/slurm/cscs/README.md](../infra/slurm/cscs/README.md).

## Prerequisites

- A CSCS GH200 allocation (e.g. Clariden), account `infra01`.
- The stock image `nvcr.io/nvidia/nemo-rl:v0.6.0` — referenced by `docker/nemo_rl.toml`, no build needed.
- The shared wheelhouse `/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/` — provides the prebuilt CUDA xIELU kernel (`aarch64/xielu-site-0.1.1-cp313`) and the `libjson-c.so.5` used for compute-node submission.
- The Apertus SFT checkpoint and tokenizer referenced in the recipe (already staged under `/capstor/store/cscs/swissai/infra01/`).

## 1. Get the code

```bash
git clone https://github.com/Alvorecer721/Nemo-RL.git
cd Nemo-RL
git submodule update --init --recursive   # Megatron-LM, Megatron-Bridge (fork), Automodel, Gym, kernels
```

The submodules are required: the Megatron driver imports `megatron.core` (an editable workspace member), and the checkpoint converter and per-step refit run on the Megatron-Bridge fork.

## 2. Run the GRPO smoke (from a login node)

```bash
mkdir -p logs
sbatch infra/slurm/cscs/probe_grpo_fixgate.slurm
```

> **First run on a fresh stock image — rebuild the worker venvs.** The checkout's `uv.lock` diverges from the image's pre-baked Ray worker venvs (the `tokenizers` relock + the forked Bridge/`kernels` submodules). `uv run --locked` re-syncs only the *driver* venv, so the first submission must rebuild the worker venvs too — otherwise the vLLM worker dies with `ImportError: libscipy_openblas64_-*.so: cannot open shared object file` (NumPy). Submit the first time with the opt-in rebuild knob:
>
> ```bash
> sbatch --export=ALL,NRL_FORCE_REBUILD_VENVS=true infra/slurm/cscs/probe_grpo_fixgate.slurm
> ```
>
> This is slow (it recompiles the worker venvs), and it **does not persist** — they build into `/opt/ray_venvs`, the container's ephemeral overlay, so you must pass the flag on *every* run on the stock image (a prior rebuild never carries over, even on the same node). `NRL_IGNORE_VERSION_MISMATCH` alone is **not** enough — it only silences the version-check gate, it does not rebuild venvs. For real cross-job reuse, set `NEMO_RL_VENV_DIR` to a persistent mounted path (see the Slurm README).

This runs 3 steps of colocated online GRPO on one node (4 GPUs, TP2/PP1) against `examples/configs/recipes/llm/probe-grpo-apertus1p5-8b-1n4g-megatron.yaml`.

**Expected:** every step prints `Generation KL Error: 0.0003` — the train↔generate logprob-agreement gate — and the run completes all 3 steps with no OOM.
A KL above ~0.002 means the generation path regressed; start from the traps page.

Submitting from *inside* a compute-node container (e.g. a coding agent that can't reach a login node) needs an extra incantation — see the "Submit from inside a compute node" section of [the Slurm README](../infra/slurm/cscs/README.md).

## 3. What the recipe sets, and the knobs that matter

- `policy.model_name` / `policy.tokenizer.name` — the Apertus SFT checkpoint and the Apertus instruct tokenizer.
- `megatron_cfg.tensor_model_parallel_size: 2` (TP2/PP1) and `converter_type: ApertusForCausalLM`.
- `megatron_cfg.optimizer.use_distributed_optimizer: true` — ZeRO-1; matches both the stock base default and the Apertus pretraining.
- `policy.generation.vllm_cfg.sleep_level: 2` — **the colocation memory fix.** vLLM discards generation weights on offload instead of backing them up to host (~17 GiB/worker), because refit repopulates them every step. Without it the host-RAM peak crosses Ray's 95% threshold at the step-2 refit and a worker is OOM-killed. See the OOM-fix commit for the full accounting.
- `gpu_memory_utilization` — the launcher overrides it to 0.40 for the colocated case.

## Why the v0.6.0 base

The Apertus work originally sat on a cu13 upstream base whose `uv.lock` does not match the stock `v0.6.0` image, so `uv run --locked` refuses and you would have to maintain a custom overlay image.
Rebasing onto the `v0.6.0` tag aligns the lock with the image: reproducible runs on the stock image, zero rebuild.
The only deliberate override on top is `tokenizers` 0.22.2 (a pure-Python wheel) for a 204s→3.3s tokenizer cold-load speedup, with byte-identical tokenization.
