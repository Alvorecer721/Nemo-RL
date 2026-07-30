# Apertus GH200 NeMo-RL Image

Container + runtime pieces for running Apertus through the NeMo-RL Megatron backend on GH200.

## Image

The public `nvcr.io/nvidia/nemo-rl:v0.6.0` image works but resolves the `mcore`/`modelopt` extras (TE included) into the venv on first `uv run` of every job.
`docker/Dockerfile.nemo_rl_v0_6_0_megatron` is a thin overlay that bakes them in and installs the fused CUDA xIELU from the `3rdparty/kernels` submodule (fork of nathanrchn/kernels with contiguity fix — run `git submodule update --init 3rdparty/kernels` before building; kernel contract: bf16 only, numel divisible by 128 — always true for ffn 21504 under TP 1/2/4/8).
Build and push:

```sh
docker buildx build \
  -f docker/Dockerfile.nemo_rl_v0_6_0_megatron \
  --tag <registry>/nemo-rl:v0.6.0-apertus-mcore-gh200 \
  --push .
```

> **Put `--environment` on `srun`, not at the batch level.** The slurm client plugins (`switch/hpe_slingshot`, pyxis) `dlopen` `libjson-c.so.5`.
> With `--environment` on `srun` (as the launchers do), the batch script and the slurm client run on the host that already has the lib, and only the task is containerized — so the image needn't bake it in.
> Promote it to `#SBATCH` (or inherit it from a CE session) and the client runs *inside* the container instead, failing on the switch plugin (then you'd need a baked libjson-c5 or an `LD_LIBRARY_PATH` stopgap).

The overlay also re-locks `tokenizers` to 0.22.2: the shipped 0.22.1 loads our 135k-added-token tokenizer in 204 s per fresh process (hf tokenizers#1635; fixed by #1891), 0.22.2 loads it in 3.3 s with byte-identical tokenization (verified in swiss-ai/apertus-omni-tokenizer#1).
The pin lives in `uv.lock` so `uv run --locked` cannot revert it; drop the `uv lock --upgrade-package` step when a future base image ships tokenizers >= 0.22.2 (the build assert keeps honesty either way).

Then distribute via the shared containers dir (preferred — no registry auth for consumers): `enroot import -o nemo-rl-v0.6.0-apertus-mcore-gh200.sqsh docker://<tag>` and move the .sqsh to `/capstor/store/cscs/swissai/infra01/containers/`; point the EDF `image =` at that absolute path.
Or put the registry tag into `docker/nemo_rl_apertus_gh200.toml` (kept in sync with `docker/nemo_rl.toml` — same mounts/env, only the image differs) and copy it to `~/.edf/` or point `sbatch`/`srun` at it.

## Apertus on upstream megatron-core

The Bridge fork checkout at `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge` (branch `yxu/apertus-v060`, rebased onto the Bridge SHA the v0.6.0 release records) runs Apertus on upstream megatron-core: bridge-owned XIELU (`models/apertus/xielu_activation.py`, optional fused CUDA kernel; the eager fallback logs a WARNING with the reason, the CUDA path logs INFO), native llama3 RoPE scaling driven by the HF config, and a finalize()-time guard that forces `bias_activation_fusion` off (incompatible with a module activation, and NeMo-RL's `megatron_cfg` would otherwise clobber it back on).

At runtime only the fork's `src` goes on `PYTHONPATH` — megatron-core must come from the worker venvs (NeMo-RL needs its own mcore version).

Gates (run them in the real env via `uv run --locked --extra mcore python ...`):

- `models/apertus/test_apertus_provider.py [tokenizer_dir]` — unit suite, 1 GPU
- `models/apertus/test_checkpoint_parity.py <hf_ckpt>` — HF-vs-Megatron logits at seq 128 and 12288 (past the rope original context, so factor-32 scaling must be active)

## DPO probe

`infra/slurm/cscs/probe_nemo_rl_dpo_megatron_apertus.slurm` — 3-step smoke on 1 node / 4 GPUs / TP2.
Checkpoint, tokenizer, bridge, and xielu paths are `${VAR:-default}`-overridable at submission.
The default tokenizer (`apertus_emu3.5_wavtok_instruct_thinking_token_fixed`) is the data-prep variant — do not use it for GRPO rollouts or eval, see its NOTES.md.

## Raw Megatron checkpoint loading (backport branch)

`yxu/v0.6.0-mlm-restore` = the v0.6.0 tag + upstream `d9efd04` (`checkpointing.pretrained_checkpoint`, landed post-release).
It keeps v0.6.0's `uv.lock`, so it runs in the upstream v0.6.0 container unchanged.

```sh
git clone --recurse-submodules -b yxu/v0.6.0-mlm-restore <this repo> nemo-rl-mlm
cd nemo-rl-mlm && uv run --locked examples/run_dpo.py --config <recipe> \
  '+checkpointing.pretrained_checkpoint.path=<.../checkpoints/iter_0004200>' \
  '+checkpointing.pretrained_checkpoint.format=megatron_lm'
```

Validated: loads raw iter_0004200 with a step-1 fingerprint identical to the certified HF conversion.
Delete this branch at the next NeMo-RL version bump (the feature is native upstream from May 2026).
