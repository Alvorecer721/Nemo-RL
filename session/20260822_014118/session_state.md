# Session State

- Session: `20260822_014118`
- Repo: `/capstor/store/cscs/swissai/infra01/users/xyixuan/nemo-rl/v0.7.0`
- Active sync worktree: `.tmp/nemo-upstream-7ea-curated`
- Active branch: `chore/sync-upstream-7ea279abf`
- Active clean commit: `8f22e59195f547c5715ed250cd49d4776cda5d43`
- Final exact image target: `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.sqsh`
- Reservation: `SD-69241-apertus-1-5-0`, expires 2026-08-31 12:00 CEST
- Updated: 2026-08-23 07:05 CEST

## Goal

Find correct, high-throughput NeMo-RL configurations on CSCS GH200 for Apertus-1.5 70B and GLM-5.1 async GRPO with Megatron training plus vLLM rollout. Measure MFU, trained-token throughput, generation throughput, refit cost, trajectory age, HBM, and end-to-end step time.

## Current Subtask

Complete. PR #24 is merged by exact-SHA fast-forward, the release image is source-identical to public `main`, Apertus-1.5 70B passed the production smoke, and GLM-5.1 is preserved as a measured scale prototype.

## Terminal Outcome

- Public `main` and merged PR #24 resolve to exact certified source `8f22e59195f547c5715ed250cd49d4776cda5d43`.
- Exact release image job `3156560`: `COMPLETED 0:0` in 2:07:40.
- Release image: `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.sqsh`.
- Image size: `50,049,777,664` bytes. SHA-256: `4aaf2b1bba8613a1e515281d84ab9e330c41d2774ccd3992b5f0c0f81e9dd002`.
- Final one-node Apertus 70B preflight `3156865`: `COMPLETED 0:0`.
- Final four-GPU Apertus PP2 refit regression `3156866`: `COMPLETED 0:0`, generation KL `0.0002`.
- Final five-node Apertus-1.5 70B async GRPO job `3156886`: `COMPLETED 0:0`; three learning steps, three refits, reward range `0..1` and nonzero advantages on every step, generation KL `0.000982/0.000994/0.000951`, and final E2E throughput `128.98` tokens/s/GPU at `14.96%` measured train MFU.
- Terminal 70B evidence: `.tmp/apertus70b-exact-cert/logs/apertus70b_async_smoke/terminal_green_20260823T044937Z_235193.json`.
- Final eight-node/32-GPU GLM-5.1 TP32/EP32 prototype `3156867`: `COMPLETED 0:0`; semantic arithmetic gate passed, `92.936` output tokens/s, about `74.00 GiB` peak HBM/GPU.
- Apertus harness `92db67ae0` and GLM harness `76217c755` are clean and published to their named `autoresearch/2026-08-23-*/exact-image` branches.
- The requested twelve-commit upstream increment is already included in frozen base `7ea279abf`. Upstream advanced after certification by one commit, `f0557321e`; it only fixes a sequence-level importance-ratio metric and remains the first post-merge follow-up because changing source now would invalidate exact-image identity.

## 2026-08-23 Final-Sync Lane

- Worktree: `.tmp/nemo-upstream-7ea-curated`
- Branch: `chore/sync-upstream-7ea279abf`
- Clean exact-image commit: `8f22e59195f547c5715ed250cd49d4776cda5d43`
- Hermetic cache job `3153709`: `COMPLETED`, fingerprint `ae440a7e8b2ab39353e47d5d879e9cfcbb3f6bdda5ca3133015592611772141d`
- Release job `3155280`: `COMPLETED 0:0`; source-exact image `084ade...` passed the built-in runtime smoke and remains the evidence base for the cheap gates.
- Release jobs `3156361`, `3156398`, and `3156544`: intentionally `CANCELLED` early for DCO, canonical-history, and hosted-CI source corrections; no dependency rebuild was lost.
- PR #24 head `8f22e5919`: copyright, secrets, lockfile, submodule, title, and triage checks pass. The only red check is the known PR-comment permission failure.
- Final release job `3156560`: `RUNNING`; exact source/fingerprint checks passed and the completed hermetic cache is being restored for source-only assembly.
- Expected final image: `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-8f22e59195f5-2a9bd7b13c00.sqsh`
- Fresh upstream fetch still resolves `upstream/main` to `7ea279abf`; there are zero post-base commits to absorb.

## Apertus Refit Fix Lane

- Branch: `fix/apertus-pp-refit-static-state`
- Clean base: `58ffe9ae1cd980b992db48a4e9f4d7dfae6864e2`
- Exact image: `/iopsstor/scratch/cscs/xyixuan/ce-images/nemo-rl/nemo-rl-apertus-vllm-0.25.1-e9416845542a-6c7d469c3e2a.sqsh`
- Proven failure: PP-rank-dependent Bridge emission of xIELU `beta`/`eps`; IPC fails closed at PP2 while the 70B NCCL misc path silently accepted incomplete static state and produced generation KL `0.58–0.73`.
- Fix contract: Apertus owns static-state semantics; NeMo-RL core contains no xIELU literals or Apertus branches and enforces only generic manifest invariants.
- Bridge fix commit: `cc3ada00325842a1a187b061dd1fba88fbc08f98`, published on `Alvorecer721/Megatron-Bridge:fix/apertus-refit-static-state`.
- NeMo-RL fix commit: `abdfe0375` in `.tmp/apertus-refit-fix`; the clean source-overlay job uses this exact commit.
- PP2 runtime gate: exact-image dependencies plus committed source, Megatron TP2/PP2 and colocated vLLM on four GH200 completed with generation KL `0.0002`, one logprob/backward/train step, and `apertus_pp2_refit_fix=OK`.
- Five-node Apertus 70B attempt `3149468` failed before model load/refit in the Bridge config broadcast with OFI `PTLTE_NOT_FOUND`/`VNI_NOT_FOUND`; the identical cache, venv and topology loaded successfully in the preceding run, so the cache remains preserved.
- Clean-allocation retry `3149806` exited application code 1 after 29:14. It spent about 17:40 starting Ray, validating the configuration, and prebuilding ReplayBuffer/AsyncTrajectoryCollector environments on five fresh nodes; about 9:20 creating the full vLLM and Megatron actor environments; then initialized vLLM and all 16 Megatron NCCL ranks. It failed at 15:45:42 in the first Bridge `read_run_config` object broadcast with the same OFI `PTLTE_NOT_FOUND`/`VNI_NOT_FOUND` errors as `3149468`, before checkpoint/model load, refit, KL, or training. Because the failure reproduced on a fresh allocation, do not classify it as a one-off bad allocation; isolate the 16-rank Megatron/OFI startup boundary before another full five-node run.
- Static A/B against successful async job `3147845` found no relevant distributed-code or topology delta. The successful actors did not inherit `FI_CXI_*`, `NCCL_NET*`, OCI hook, or `LD_PRELOAD`; both failed jobs inherited the current compute allocation's profile via the launcher's `sbatch --export=ALL`. That profile is incompatible with the checkout's CSCS contract: `cuda12` instead of `cuda-dl` for the CUDA-13 image, `NCCL_NET_PLUGIN=ofi` in addition to `NCCL_NET=AWS Libfabric`, GDR level `0` instead of `PHB`, TX size `32768` instead of `16384`, and host-only `libstdbuf.so` in `LD_PRELOAD`. The checked-in CSCS README explicitly forbids `--export=ALL` for compute-node submission.
- A syntax-checked eight-rank/two-node Ray reproducer is staged under the ignored `.tmp/apertus-refit-fix/.tmp/ofi_probe/` directory. It performs the exact failing object broadcast plus a tensor all-reduce and barrier. Run matched dirty-export versus explicit clean CSCS-OFI jobs before modifying production launchers.
- GLM cross-allocation checkpoint Phase A job `3148504` trained/refit step 1 and scheduled an async optimizer checkpoint, but is currently stalled with 272/288 rank shards written (7.6 TiB) and 16 rank shards absent. Do not call this checkpoint complete or launch Phase B.

## Accepted Evidence

- Apertus-1.5 70B topology selection is complete. All candidates ran 10 steps and exited `0:0` on four nodes/16 GH200:
  - TP4/PP4 `3146385` selected: 19.349 s steady step, 183.35 TFLOPS/GPU, 18.53% external GH200 BF16 MFU, 85.70 valid tokens/s/GPU, 78.72 GiB peak HBM.
  - TP8/PP2 `3146827`: 24.301 s, 145.78 TFLOPS/GPU, 14.73% MFU, 68.22 valid tokens/s/GPU, 70.17 GiB.
  - TP2/PP8 `3146572`: 28.143 s, 125.98 TFLOPS/GPU, 12.73% MFU, 58.89 valid tokens/s/GPU, 90.60 GiB.
- GLM-5.1 real-weight vLLM TP32/EP32 proof `3146245` completed `0:0` on eight nodes/32 GH200: 98.741 output tokens/s for the measured sampled batch and about 73.98 GiB peak HBM/GPU.
- Historical 10-step GLM job `3147936` reached step 10 but is not accepted as a clean benchmark: Slurm recorded `FAILED 1:0`, and the strict post-hoc validator rejected it because not every step had a nonzero GRPO advantage range. Preserve it as prototype/body evidence only.
- GLM conversion cache is complete at `/iopsstor/scratch/cscs/xyixuan/.cache/huggingface/nemo_rl_glm51_tp1pp18ep4`, 72 DCP shards plus metadata, about 1.488 TB decimal. Fresh-process reload `3147363` completed `0:0` with 72/72 ranks and about 35.3 GiB minimum free HBM.
- Baseline topology is 80 nodes/320 GPUs: 72 training nodes TP1/PP18/EP16/DP16 and eight rollout nodes vLLM TP32/EP32, async max age 1, importance-sampling correction, and NCCL reshard refit.
- The KL-free throughput mode is explicit: reference logprobs skipped and reference KL penalty zero. Keep separate from the unit-gated but not runtime-proven KL=0.01 fix in `.tmp/glm-reference-object-load`.
- Actor-venv hardening is runtime-proven. Two-node gate `3147462` passed fresh build plus reuse, and 80-node smoke `3147466` completed `0:0` with all four actor types converged, three async train steps, repeated refits, and weight version 3.
- Final exact-image preflight `3147583` completed `0:0`: 82 tests, Ruff, 24-file formatting, shell syntax, data/topology checks, and Megatron actor Bridge/MFU import.

## Active Job

- None. All final build and runtime jobs reached authoritative terminal states.

## Earned Failures

- `3147380`: primary 288-rank policy loaded, but inherited KL=0.01 triggered a second reference-policy load whose WORLD loaded-object exchange corrupted serialized payloads. Throughput mode was made explicitly KL-free; a separate Bridge/MCore per-rank-object fix is unit-gated only.
- `3147405`: one of 80 late control-actor venv builds hit uv `Directory not empty`. The bounded retry, serialized installs, early 80-node prebuild, and regression/integration gates closed this failure mode.
- `3147552`: both control actor venvs reached 80/80, then setup failed before model load because positive `val_period=1000` enabled validation while the fixed real-DAPO dataset has no validation split. Commit `c2dc28262` sets `val_period=0` and gates the complete no-validation contract for smoke and benchmark.

## Benchmark Contract

- Ten steps, 32 prompts x 8 generations, GBS256/MBS1, configured sequence 2048, max-new-tokens 1536, DAPO SHA `734c1ae1ec27af3b28eafba86fd38bc44a2e0c1acfde0c134bbb0408fcf246ea`.
- Benchmark submission refuses to run without a terminal-green smoke artifact matching the exact Git head and exact image SHA-256.
- Evidence includes actual-length-corrected GLM MFU, all 320 GPU telemetry series, all 18 PP stages, trained valid-token throughput, estimated output-only critical-path throughput, raw vLLM token deltas, queue/KV metrics, refit cost, age, HBM, step timings, provenance and resolved config.
- The hard-coded 12-hour limit is not worst-case safe: 10 full 1536-token batches take about 11.06 hours at the measured 98.741 tokens/s, and one async lookahead batch raises generation alone to about 12.17 hours. Preserve exact source identity and extend the submitted job operationally to about 18 hours with `scontrol update` if cluster policy permits; otherwise do not pretend 12 hours is sufficient.

## Plan

- [x] Apertus 70B matched topology comparison and selection.
- [x] GLM real-weight vLLM fit/throughput proof.
- [x] GLM conversion plus fresh-process reload.
- [x] GLM KL-free topology/config, venv convergence, telemetry and exact-image gates.
- [x] Definitive exact-HEAD image assembly `3156560`.
- [x] Exact-image one-node Apertus preflight and PP2 refit regression.
- [x] Three-step five-node Apertus 70B async GRPO production gate with nonzero reward/advantage and KL `<0.002` on every step.
- [x] Eight-node GLM-5.1 vLLM TP32/EP32 scale prototype on the same image.
- [x] Update PR evidence and merge only after all source/runtime gates are coherent.

## Non-Blocking Follow-ups

- Runtime-prove the KL=0.01 per-rank reference-policy load fix in `.tmp/glm-reference-object-load` before publication.
- After baseline metrics, consider 64 train/16 rollout nodes with two TP32 replicas. First run a PP16 load-only reshard probe and two-step HBM/refit gate; doubled refit receiver volume is the primary risk.
- Native SGLang and long endurance/cross-failure testing remain separate from this vLLM throughput campaign.

## Rules

- Do not call any Slurm run successful without authoritative terminal `State/ExitCode` and semantic markers.
- Use escalated Slurm queries; sandbox DNS failures can falsely report the controller down.
- Do not modify `.tmp/glm51-async-topology` while a source-bound smoke/benchmark is running; the artifact intentionally fails closed on dirty or changed source.
- Do not delete or reconvert the 1.488-TB GLM DCP cache.
