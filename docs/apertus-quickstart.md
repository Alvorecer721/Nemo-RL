# Apertus 1.5 8B on NeMo-RL: clone-and-run (CSCS GH200)

How to reproduce online GRPO post-training for Apertus 1.5 8B on a CSCS GH200 node from a clean checkout.
This branch runs on the **stock `nvcr.io/nvidia/nemo-rl:v0.7.0` image** via `uv run --locked` — no custom image build.
For the architecture gotchas behind the gates here, see [apertus-traps-and-invariants.md](apertus-traps-and-invariants.md); for Slurm submission details, see `infra/slurm/cscs/README.md` in the repo.

## Prerequisites

- A CSCS GH200 allocation (e.g. Clariden), account `infra01`.
- The stock image `nvcr.io/nvidia/nemo-rl:v0.7.0` — referenced by `docker/nemo_rl.toml`, no build needed.
- The shared wheelhouse `/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/` — provides the prebuilt CUDA xIELU kernel (`aarch64/xielu-site-0.1.0-cp313-torch2.11.0-cu130`) and the `libjson-c.so.5` used for compute-node submission.
- The Apertus SFT checkpoint and tokenizer referenced in the recipe (already staged under `/capstor/store/cscs/swissai/infra01/`).

## 1. Get the code

```bash
git clone https://github.com/Alvorecer721/Nemo-RL.git
cd Nemo-RL
git submodule update --init --recursive   # Megatron-Bridge (Apertus fork), Automodel, Gym
```

The submodules are required: the checkpoint converter and per-step refit run on the Megatron-Bridge fork (which also vendors `megatron.core`).

## 2. Run the GRPO smoke (from a login node)

```bash
mkdir -p logs
sbatch infra/slurm/cscs/probe_grpo_fixgate.slurm
```

> **First launch on a fresh account:** run one probe to completion before submitting anything else.
> The first `uv run --locked` populates your uv cache and the checkout-local venvs; two cold-cache
> jobs racing the same source build deadlock on uv's per-package lock (300 s timeout).
> Subsequent submissions start in minutes and may run concurrently.
> The async variant is `sbatch infra/slurm/cscs/probe_grpo_async.slurm` (same KL gate, 2+2 GPU split).

This runs 3 steps of colocated online GRPO on one node (4 GPUs, TP2/PP1) against `examples/configs/recipes/llm/probe-grpo-apertus1p5-8b-1n4g-megatron.yaml`.

**Expected:** every step prints `Generation KL Error: 0.0003` — the train↔generate logprob-agreement gate — and the run completes all 3 steps with no OOM.
A KL above ~0.002 means the generation path regressed; start from the traps page.

Submitting from *inside* a compute-node container (e.g. a coding agent that can't reach a login node) needs an extra incantation — see the "Submit from inside a compute node" section of `infra/slurm/cscs/README.md`.

## Environment model (what gets built where)

- The first `uv run --locked` materializes the project venv at `<repo>/.venv` and per-worker venvs
  under `<repo>/venvs/` — both persistent across jobs (the launchers export
  `UV_PROJECT_ENVIRONMENT`/`NEMO_RL_VENV_DIR`; the image defaults are container-overlay paths that
  die with the job). Delete those two directories to force a clean rebuild.
- Package downloads and source builds resolve through the image-seeded uv cache
  (`UV_CACHE_DIR=/root/.cache/uv`, readable under CE), so a fresh checkout should not recompile
  TransformerEngine. Steady-state job setup is single-digit minutes.
- vLLM torch.compile caches persist under `~/.cache/vllm*`; the HF→Megatron checkpoint conversion is
  cached under `$HF_HOME/nemo_rl/` and reused across runs and algorithms.

## 3. What the recipe sets, and the knobs that matter

- `policy.model_name` / `policy.tokenizer.name` — the Apertus SFT checkpoint and the Apertus instruct tokenizer.
- `megatron_cfg.tensor_model_parallel_size: 2` (TP2/PP1).
- `megatron_cfg.optimizer.use_distributed_optimizer: true` — ZeRO-1; matches both the stock base default and the Apertus pretraining.
- `policy.generation.vllm_cfg.sleep_level: 2` — **the colocation memory fix.** vLLM discards generation weights on offload instead of backing them up to host (~17 GiB/worker), because refit repopulates them every step. Without it the host-RAM peak crosses Ray's 95% threshold at the step-2 refit and a worker is OOM-killed. See the OOM-fix commit for the full accounting.
- `gpu_memory_utilization` — the launcher overrides it to 0.40 for the colocated case.

## Why the stock-release base

The Apertus branch always sits on a release tag whose `uv.lock` matches the corresponding stock NGC image, so `uv run --locked` works with zero rebuild.
This checkout tracks `v0.7.0`; the Megatron-Bridge submodule points at the Apertus fork rebased onto the same bridge pin the release ships.
