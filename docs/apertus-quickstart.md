# Apertus 1.5 8B on NeMo-RL: clone-and-run (CSCS GH200)

How to reproduce online GRPO post-training for Apertus 1.5 8B on a CSCS GH200 node from a clean checkout.
The default path runs on the **stock `nvcr.io/nvidia/nemo-rl:v0.7.0` image** via `uv run --locked` — no custom image build. Since the vLLM 0.25.1 bump, the lock installs vLLM 0.25.1 (official prebuilt aarch64 wheel) into the checkout venvs; the stock image supplies the host runtime, not the Python stack.
For the architecture gotchas behind the gates here, see [apertus-traps-and-invariants.md](apertus-traps-and-invariants.md); for Slurm submission details, see `infra/slurm/cscs/README.md` in the repo.
The faster vLLM 0.25.1 stack is also clone-and-run: a certified prebuilt image is shared under `MLLM/containers/` and the checkout ships its EDF (`docker/nemo_rl_vllm0251.toml`) — see the "Custom vLLM 0.25.1 GH200 image" section of the Slurm README.

## Prerequisites

- A CSCS GH200 allocation (e.g. Clariden), account `infra01`.
- The stock image `nvcr.io/nvidia/nemo-rl:v0.7.0` — `docker/nemo_rl.toml` serves it from the shared pre-pulled copy in `MLLM/containers/` (no pull, no build; see the Slurm README for the registry alternative).
- The shared wheelhouse `/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/` — provides the prebuilt CUDA xIELU kernel for the **training side** (`aarch64/xielu-site-current`, a symlink the launchers follow so kernel bumps need no launcher edits) and the `libjson-c.so.5` used for compute-node submission. vLLM generation deliberately runs its fused-Python xIELU — a measured tie against the kernel under compile; see [the xIELU reference](apertus-xielu.md).
- The Apertus SFT checkpoint and tokenizer referenced in the recipe (already staged under `/capstor/store/cscs/swissai/infra01/`).

## 1. Get the code

```bash
git clone https://github.com/Alvorecer721/Nemo-RL.git
cd Nemo-RL
git submodule update --init --recursive   # Megatron-Bridge (Apertus fork), Automodel, Gym, kernels (xIELU source)
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

This runs 3 steps of colocated online GRPO on one node (4 GPUs, TP2/PP1) against `examples/configs/recipes/llm/grpo-apertus1p5-8b-1n4g-megatron-probe.yaml`.

**Expected:** every step prints `Generation KL Error: 0.0003` — the train↔generate logprob-agreement gate — and the run completes all 3 steps with no OOM.
A KL above ~0.002 means the generation path regressed; start from the traps page.

Submitting from *inside* a compute-node container (e.g. a coding agent that can't reach a login node) needs an extra incantation — see the "Submit from inside a compute node" section of `infra/slurm/cscs/README.md`.

## Environment model (what gets built where)

- The first `uv run --locked` materializes the project venv at `<repo>/.venv` and per-worker venvs
  under `<repo>/venvs/` — both persistent across jobs (the launchers export
  `UV_PROJECT_ENVIRONMENT`/`NEMO_RL_VENV_DIR`; the image defaults are container-overlay paths that
  die with the job). Delete those two directories to force a clean rebuild.
- Package downloads and source builds resolve through the team uv cache on capstor
  (`UV_CACHE_DIR=/capstor/store/cscs/swissai/infra01/MLLM/uv-cache`, seeded from the image), so a
  fresh checkout recompiles nothing — a full cold start measured 0 downloads, 0 source builds.
  Clone on capstor: uv then hardlinks from the cache instead of copying, so venv creation is fast
  and near-free on disk. Steady-state job setup is single-digit minutes.
- vLLM torch.compile caches persist under `~/.cache/vllm*`; the HF→Megatron checkpoint conversion is
  cached under `$HF_HOME/nemo_rl/` and reused across runs and algorithms.

## 3. What the recipe sets, and the knobs that matter

- `policy.model_name` / `policy.tokenizer.name` — the Apertus SFT checkpoint and the Apertus instruct tokenizer.
- `megatron_cfg.tensor_model_parallel_size: 2` (TP2/PP1).
- `megatron_cfg.optimizer.use_distributed_optimizer: true` — ZeRO-1; matches both the stock base default and the Apertus pretraining.
- `policy.generation.vllm_cfg.sleep_level: 2` — **the colocation memory fix.** vLLM discards generation weights on offload instead of backing them up to host (~17 GiB/worker), because refit repopulates them every step. Without it the host-RAM peak crosses Ray's 95% threshold at the step-2 refit and a worker is OOM-killed. See the OOM-fix commit for the full accounting.
- `gpu_memory_utilization` — the launcher overrides it to 0.40 for the colocated case.

## Why the stock-release base

The tree is rooted on the `v0.7.0` release whose stock NGC image supplies the host runtime (CUDA, drivers, system libraries); the Megatron-Bridge submodule points at the Apertus fork rebased onto the same bridge pin the release ships.
The Python stack comes from `uv.lock`, which has deliberately moved ahead of the image's baked packages — most notably vLLM 0.20.0 -> 0.25.1 — and still resolves with zero source builds: the vLLM pin is an official prebuilt aarch64 wheel and everything else comes from the team uv cache.
Prefer the certified baked image (see the header note) when you want the 0.25.1 stack preinstalled instead of venv-resolved.
