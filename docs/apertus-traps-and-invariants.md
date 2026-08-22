# Apertus × NeMo-RL: traps, invariants, and gates

Hard-won knowledge from bringing Apertus 1.5 up on NeMo-RL (2026-06-12/13).
Every entry names a *class* of failure, not just the instance we hit — the pattern is always the same: **two stacks agreeing by coincidence instead of by contract**.
When you add a model variant or bump a dependency, re-read this page.

> New here? Start with the [Apertus quickstart](apertus-quickstart.md) to clone-and-run; this page is the deeper gotchas behind the gates.

## 1. vLLM dummy-load orphans non-trainable state  ⚠ fixed, stays fixable

NeMo-RL forces `load_format="dummy"` for training-mode vLLM engines (`nemo_rl/models/generation/__init__.py`) — engines start as noise and rely on refit to deliver real weights.
Refit streams **trainable parameters only** (`bridge.export_hf_weights`).
Any architecture whose checkpoint carries forward-load-bearing **buffers** is silently corrupted: for Apertus, the 64 xIELU `act_fn.{beta,eps}` buffers (32 layers × 2) stayed at noise → Generation KL Error 0.7935 ≈ a genuinely off-policy generator.

- **Fix (shipped)**: the bridge emits the xIELU `beta`/`eps` buffers into the HF refit stream (`apertus_bridge.maybe_modify_converted_hf_weight`, in the pinned Apertus bridge fork), so vLLM may dummy-load and the step-0 refit still delivers correct constants. Verified originally via disk-load parity: gen KL 0.7935 → **0.0003**.
- **Posture**: prefer `load_format=auto` for any new architecture until its buffer inventory is audited (`state_dict` keys vs `named_parameters`).
- **Upstream asks**: (a) NeMo-RL: auto-refuse dummy when the checkpoint carries buffers absent from the refit stream; (b) vLLM: `XIELU`'s Python path should read the init-captured scalars like its CUDA path does — makes the class unconstructible.
- **Masking hazard**: the vLLM CUDA xielu path hides this bug (scalars captured at `__init__`). An environment with the kernel installed shows nothing; one without it shows 0.79. Never debug this class by comparing environments with different kernel availability.

## 2. llama3 RoPE scaling: 4 parameters, 1 exposed  ⚠ latent

The 256k SFT config carries `rope_scaling = {factor 32, low_freq_factor 1.0, high_freq_factor 4.0, original_max_position_embeddings 8192}`.
mcore's API accepts **only the factor**; the other three are hardcoded (`rotary_pos_embedding.py`: 1, 4, 8192) and match our config **by coincidence**.
A future long-context variant rescaled from a different original_max (e.g. 262144) would silently train wrong positional geometry.

- **Action (queued)**: bridge provider asserts the three unexposed sub-parameters equal mcore's constants at load; refuse loudly otherwise.
- Same class, same cure for Megatron's xIELU constants: the Megatron-side `beta/eps` are constructor constants that happen to equal the checkpoint's values — assert that too.

## 3. Chat template hardcodes BOS × vLLM text-prompt path  ⚠ latent for stock VLM flows

Our `chat_template.jinja` emits `{{ bos_token }}`.
NeMo-RL's vLLM worker submits **token ids** for text-only rollouts (safe: tokenizer never reruns) but **rendered text** for the `vllm_content` VLM path — which vLLM re-tokenizes (with `add_special_tokens=True` defaults in newer vLLM) → **double BOS**, one-position shift, catastrophic logprob comparisons.

- Our discrete-token multimodal design dodges this structurally (image tokens ride inside the ids path). Keep it that way.
- **Tripwire to add in any prompt builder**: assert `ids[0] == bos and ids[1] != bos` before submission.

## 4. Dependency pins must live in uv.lock, not the venv

`uv run --locked` re-syncs the environment to the lockfile at every launch — bare pip installs into the venv are silently reverted.
(Applied for tokenizers 0.22.2: 204 s → 3.3 s cold tokenizer load; see the Dockerfile's `uv lock --upgrade-package` step.)

## 5. Resume requires an identical training horizon

`max_num_steps` / `max_num_epochs` / gbs feed `train_iters` → Megatron's scheduler state counts in samples (iters × 2 × gbs for DPO).
Changing any of them at resume trips `OptimizerParamScheduler ... do not match`.
Short resume tests: keep the config byte-identical and kill externally.

## 6. The gauges and what they must read

| gauge | healthy | meaning |
|---|---|---|
| step-1 `preference_loss` (DPO, policy≡reference) | **0.6931 (ln 2)** exactly | pins reference init, logprob roll, pair interleave |
| `gen_logprob_error_mean` (online tripwire) / GRPO gen-KL | ~0.001–0.04 (engine numerics); warn ≥0.05 | "generator ≡ trainer" health check; growth = weight-path bug |
| policy-vs-reference KL (implicit, via β) | grows during training | that's learning, not a bug |
| cross-engine logprob parity (Megatron/vLLM/HF) | ~0.02–0.05 mean abs | measured: Megatron-HF 0.044, vLLM-HF 0.036 |

## 7. Certification ladder (run after touching anything in the weight path)

```bash
# 1. step-0 refit self-diff: NRL_DEBUG_REFIT_SELFDIFF_DIR=... run_grpo (debug recipe)
#    → expect 0 tensors changed
# 2. GRPO probe (grpo-apertus1p5-8b-1n4g-megatron-probe.yaml, vllm util 0.40)
#    → expect Generation KL Error < 0.002
# 3. online-DPO smoke (infra/slurm/cscs/probe_nemo_rl_dpo_megatron_apertus.slurm)
#    → expect step-1 preference_loss ≈ 0.6931
```

Historical (v0.6.0-era) probe harnesses live in `nemo-rl-worktrees/v060-online-dpo/debug/` (self-diff analyzer, Megatron-vs-HF forward parity, vLLM disk parity).

## 8. vLLM compile-cache is blind to kernel presence  ⚠ dormant since the generation-side kernel removal

vLLM caches compiled graphs under a key blind to which xIELU implementation was
importable at trace time; both poisoning directions occurred (full mechanics and the
"+27%" autopsy: `docs/apertus-xielu.md`).

- **Rule**: purge `~/.cache/vllm*/torch_compile_cache` at every kernel-presence boundary.
  Dormant while generation stays kernel-free (homogeneous caches); re-arms instantly if
  anyone re-injects `XIELU_SITE` into vLLM workers.
- **Upstream ask (queued with the vLLM compile-safety PR)**: include the resolved
  activation implementation in the cache key.

## 9. vLLM throughput snapshots measure duty cycle, not speed  ⚠ permanent metric trap

`Avg generation throughput: N tokens/s` lines average over wall-clock windows that
span sleep/training time, so they encode duty cycle and print alignment, not burst
speed; cross-run comparison of them produced the false "+27%" (full autopsy:
`docs/apertus-xielu.md`).

- **Rule**: never compare snapshot tok/s across runs. Throughput claims come from the
  per-step `generation:` phase timer normalized by `Mean Generation Length`
  (ms/token), stall-steps stated separately; A/Bs must be same-node paired runs.

## 10. Slurm drops empty-valued variables from --export  ⚠ permanent launcher trap

`sbatch --export=NONE,VAR=` does **not** deliver `VAR` set-to-empty — the variable
arrives unset, so `${VAR:-default}` applies the default. An "arm without the kernel"
submitted this way ran *with* the kernel and reported plausible numbers; only an in-log
attestation (`grep -c 'Using experimental xIELU CUDA'`) exposed it.

- **Rule**: to force an empty/absent path through sbatch, point the variable at an
  existing empty directory rather than passing an empty value — and have every
  experiment arm print its own configuration attestation into its log (the shared
  runner in `infra/slurm/cscs/bench/arm_lib.sh` does both legs). Intended
  configuration proves nothing; logs attest actual configuration.
