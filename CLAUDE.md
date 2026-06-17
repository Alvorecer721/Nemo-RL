# NeMo-RL

NeMo-RL is an RLHF training framework built on Ray and PyTorch (FSDP2 / Megatron-Core). It supports algorithms like GRPO, DPO, and SFT for LLMs and VLMs.

## This fork: Apertus 1.5 on CSCS GH200

This is the `Alvorecer721/Nemo-RL` fork, rebased onto the upstream `v0.6.0` tag. On top of stock NeMo-RL it adds online RL post-training (GRPO / DPO / MPO) for **Apertus 1.5 8B** on **CSCS GH200** (Clariden, Grace-Hopper / aarch64), runnable from a clean checkout on the stock `nvcr.io/nvidia/nemo-rl:v0.6.0` image via `uv run --locked` (no custom image build). The Apertus work was rebased off its original cu13 upstream base onto the `v0.6.0` line to align `uv.lock` with the stock image; the only deliberate dep override is `tokenizers` 0.22.2 (204s→3.3s cold load, byte-identical).

**Worker-venv caveat:** that relock plus the forked Bridge + `kernels` submodules (editable workspace members) make the checkout's `uv.lock` diverge from the stock image's pre-baked Ray **worker** venvs under `/opt/ray_venvs/`. `uv run --locked` re-syncs only the *driver* venv, not those, so without a rebuild the vLLM worker's NumPy fails to load `libscipy_openblas*.so` (`ImportError: cannot open shared object file`). Rebuild with `NRL_FORCE_REBUILD_VENVS=true` — the CSCS launchers expose it as an opt-in knob (default off): `sbatch --export=ALL,NRL_FORCE_REBUILD_VENVS=true infra/slurm/cscs/probe_grpo_fixgate.slurm`. `NRL_IGNORE_VERSION_MISMATCH` only silences the startup version-check gate; it does **not** rebuild venvs. **The rebuild does not persist:** `/opt/ray_venvs` lives in the container's ephemeral writable overlay (not a mounted path), discarded at every job's end, and NeMo-RL's reuse check is only "does `bin/python` exist" (no `uv.lock` comparison, `nemo_rl/utils/venvs.py`) — so you must pass the flag on **every** stock-image run, on any node. For genuine cross-job reuse, point `NEMO_RL_VENV_DIR` at a persistent mount (e.g. `/iopsstor/scratch/cscs/$USER/nemo_rl_venvs`) and rebuild only when `uv.lock` changes; or bake the venvs into the overlay image.

Where the fork-specific work lives:

- `nemo_rl_apertus/` — additive top-level package (zero edits to upstream files). Contains `mpo_loss.py` (MPO = DPO + BCO quality term), `runtime_guard.py` (`assert_apertus_runtime()` — fails loudly if `import nemo_rl` resolves to the container's stock `/opt/nemo-rl` instead of this checkout), and a test suite.
- `infra/slurm/cscs/` — Clariden/GH200 Slurm launchers + probes. See `infra/slurm/cscs/README.md` for login-node vs compute-node (coding-agent) submission. `infra/pythonpath/nemo_rl_apertus` is a symlink used to put the checkout on `PYTHONPATH`.
- `examples/run_mpo_apertus.py`, plus recipes `examples/configs/recipes/llm/probe-grpo-apertus1p5-8b-1n4g-megatron.yaml` and `dpo-apertus1p5-8b-maxmin-megatron.yaml`.
- `tools/export_megatron_to_hf.py` — Megatron→HF export CLI on the certified Bridge path.
- `docker/` — Apertus GH200 overlay (`Dockerfile.nemo_rl_v0_6_0_megatron`, `.toml` env files, pretrained-checkpoint backport patch).

A few **edits do touch upstream in-tree files** (so the runtime guard above matters):

- `nemo_rl/models/huggingface/common.py` — `is_apertus_model()` wired into `ModelFlag.VLLM_LOAD_FORMAT_AUTO`. Apertus must vLLM-disk-load (`load_format="auto"`), because its xIELU `beta`/`eps` **buffers** are in the HF checkpoint but NOT in the Bridge `export_hf_weights` refit stream; with `dummy` load they stay at noise and gen↔train KL silently regresses to ~0.79 (fixed: 0.0003).
- `nemo_rl/algorithms/grpo.py` — guards `vllm_cfg.sleep_level>=2` against `val_at_start` in colocated mode.

Read these before changing anything in the weight/generation path: `docs/apertus-quickstart.md` (clone-and-run) and `docs/apertus-traps-and-invariants.md` (gates, the KL=0.0003 invariant, certification ladder). Both are also linked from `docs/index.md`.

### Submodules (changed by this fork — and why)

`git submodule update --init --recursive` is **required**, not optional: the Megatron driver imports `megatron.core` as an editable workspace member, and the checkpoint converter + per-step refit run on the Bridge. A GRPO launcher guard fails fast on uninitialized submodules. Changes vs upstream (`.gitmodules`):

- **`Megatron-Bridge` → fork `Alvorecer721/Megatron-Bridge`, branch `yxu/apertus-v060`** (was NVIDIA's). The Apertus converter (`ApertusForCausalLM`) and the per-step `export_hf_weights` refit live in the fork, not upstream Bridge.
- **`3rdparty/kernels` → new submodule, `Alvorecer721/kernels.git`** — supplies the CUDA xIELU activation kernel Apertus' MLP needs on GH200 (prebuilt wheel staged in the shared wheelhouse; `XIELU_SITE` defaults to it). Note: the kernel masks the buffer-corruption bug above, so never debug that class by comparing environments with/without the kernel.
- **`Megatron-LM` entry restored** after the v0.6.0 rebase dropped it; the GRPO launchers pass `--extra mcore` because the driver imports `megatron.core`.

## Coding Guidelines

Coding guidelines are organized as Claude skills in `.claude/skills/`. Each skill covers a specific topic (style, config conventions, error handling, testing, copyright, docs).

## Code Review

Use `/review-pr <pr-number>` for interactive local PR review.

When reviewing code, follow these principles:

- **Be concise and actionable.** Focus on bugs, logic errors, missing tests, outdated docs, and guideline violations.
- **Do NOT flag:** style/formatting (linters handle it), minor naming suggestions, architectural opinions, or performance unless there is a clear measurable issue.
- **High confidence only.** Only flag issues you are confident about. If unsure, skip it.
- **Verify upstream API usage.** When code calls into megatron-bridge, megatron-lm, automodel, or gym APIs, look up the actual API to verify correct usage. Evaluate each such call with scrutiny — don't assume the author got the signature, return type, or semantics right.
- It is perfectly acceptable to have nothing to comment on. Say "LGTM" if so.
