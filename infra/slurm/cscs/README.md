# CSCS Slurm Probes

This directory contains the Clariden/GH200 Slurm wrappers used to validate the NeMo-RL `nvcr.io/nvidia/nemo-rl:v0.6.0` container on Slingshot.

The default container environment is `docker/nemo_rl.toml` in this checkout. The wrappers set `CUDA_CACHE_PATH` and Hugging Face cache paths in shell code because TOML values are not shell-expanded by Pyxis/EDF.

## Submit from a login node (humans)

Submit these from a **login node** (e.g. `clariden-ln001`).
The wrappers follow CSCS guidance — `--environment` lives on each `srun`, never on `#SBATCH` — so the batch script and the Slurm client run on the **host**, which has the system libraries the Slingshot/pyxis plugins need (`libjson-c.so.5`); only the task is containerized.
A login node has those libraries natively, so nothing extra is required.

Run from the repository root after creating the log directory:

```bash
mkdir -p logs
sbatch infra/slurm/cscs/probe_nemo_rl_env.slurm
sbatch infra/slurm/cscs/probe_nemo_rl_nccl_2n_4r.slurm
sbatch infra/slurm/cscs/probe_nemo_rl_dpo_megatron_2n.slurm
```

Useful overrides:

```bash
CONTAINER_ENV=/path/to/your/.edf/nemo_rl.toml sbatch infra/slurm/cscs/probe_nemo_rl_env.slurm
GPUS_PER_NODE=4 TRAIN_GLOBAL_BATCH_SIZE=16 sbatch infra/slurm/cscs/probe_nemo_rl_dpo_megatron_2n.slurm
```

## Rebuilding worker venvs (first run after a lock change)

A launcher that runs a **fork checkout** needs the Ray *worker* venvs rebuilt the **first** time that checkout's `uv.lock` diverges from the stock image's pre-baked venvs. The checkout does diverge here: the fork relocks `tokenizers` 0.22.2 and pulls in the forked Megatron-Bridge + `kernels` submodules as editable members. `uv run --locked` re-syncs only the *driver* venv, so without a worker rebuild the vLLM worker dies with:

```
ImportError: libscipy_openblas64_-*.so: cannot open shared object file   # numpy, in /opt/ray_venvs/...
```

This **always** applies to the GRPO probes (`probe_grpo_*`) and `probe_nemo_rl_dpo_megatron_apertus`, which run `$REPO_DIR` (the checkout). It applies to `submit_nemo_rl_dpo*` **only when** you point `RUNTIME_DIR` at a checkout/worktree — by default they set `RUNTIME_DIR=/opt/nemo-rl` (the image copy), whose lock already matches the baked venvs and needs no rebuild.

The relevant launchers expose an opt-in knob, `NRL_FORCE_REBUILD_VENVS` (default `false`). Set it on the first run after a lock change:

```bash
sbatch --export=ALL,NRL_FORCE_REBUILD_VENVS=true infra/slurm/cscs/probe_grpo_fixgate.slurm
```

It is slow (the worker venvs recompile flash-attn / TransformerEngine). **The rebuild does not persist across jobs:** the venvs build into `/opt/ray_venvs`, which is the container's *ephemeral writable overlay* (`/opt` is not a mounted path — only `/capstor`, `/iopsstor`, `/users` are), discarded when the job ends. Worse, NeMo-RL's reuse check is only whether `<venv>/bin/python` exists (no `uv.lock` comparison — see `nemo_rl/utils/venvs.py`), so the next job silently reuses the stale **baked** venv and fails again. **So on the stock image you must pass `NRL_FORCE_REBUILD_VENVS=true` on every run, on any node** — there is no "first run only."

To get genuine cross-job reuse, point `NEMO_RL_VENV_DIR` at a persistent mounted path, e.g.:

```bash
NEMO_RL_VENV_DIR=/iopsstor/scratch/cscs/$USER/nemo_rl_venvs \
  sbatch --export=ALL,NRL_FORCE_REBUILD_VENVS=true,NEMO_RL_VENV_DIR=/iopsstor/scratch/cscs/$USER/nemo_rl_venvs \
  infra/slurm/cscs/probe_grpo_fixgate.slurm
```

Then the first run rebuilds onto Lustre and later runs reuse it — but **invalidate it yourself** (`rm -rf` the dir, or pass the flag again) whenever the checkout's `uv.lock` changes, since the cache is keyed only on `bin/python` existing. The alternative "bake once" path is the Docker overlay image (`docker/Dockerfile.nemo_rl_v0_6_0_megatron`). `NRL_IGNORE_VERSION_MISMATCH` is **not** a substitute for any of this — it only silences the startup version-check gate, it does not rebuild venvs. Launchers that run the image's own `/opt/nemo-rl` copy (`probe_nemo_rl_dpo_megatron`, `probe_nemo_rl_dpo_megatron_2n`, the env probes) do not need this — their lock already matches the baked venvs.

## Online DPO (judge-served, two-job)

Online DPO is a **two-job** Slurm setup: a judge **server** job and the **training** job, wired by the JUDGE env vars (`JUDGE_BASE_URL`, `JUDGE_MODEL` = the server's served-model-name, `JUDGE_API_KEY`). For how online DPO works — the judge, datasets, what the judge receives, logging, validation — see `docs/apertus-dpo.md`; this section covers Slurm submission.

Probe (recommended, repoint-and-go) — `probe_online_dpo_1n_1judge.sh` is a thin preset over the general `online_dpo_launcher.sh` (both online presets — this probe and `launch_online_dpo_maxmin.sh` — share that engine, which submits the orchestrator as a 1-node job → launches the judge → discovers its URL → health-checks → sbatches training). Probe scale by default (single-node judge + the 1-node/4-GPU DeepScaler recipe):

```bash
cd <repo>   # run from the repo root so relative logs/ resolve consistently
JUDGE_SERVE_MODEL=/path/to/judge-model JUDGE_API_KEY=$MYKEY \
    infra/slurm/cscs/probe_online_dpo_1n_1judge.sh
```

Or drive the orchestrator directly from a login node (no wrapping job):

```bash
mkdir -p logs
JUDGE_SERVE_MODEL=/path/to/judge-model JUDGE_API_KEY=$MYKEY \
    infra/slurm/cscs/online_dpo_orchestrator.sh
```

Manual (two steps) — bring the server up, read its URL from the log, then submit training:

```bash
sbatch --export=ALL,JUDGE_SERVE_MODEL=/path/to/judge-model infra/slurm/cscs/serve_judge.slurm
# grep "JUDGE SERVER URL:" logs/judge_server_<jobid>.out
sbatch --export=ALL,JUDGE_BASE_URL=http://<host>:8080/v1,JUDGE_MODEL=<served-name>,JUDGE_API_KEY=$MYKEY \
    infra/slurm/cscs/submit_online_dpo.slurm
```

**Single-node judge** (`serve_judge.slurm`, the default backend) knobs — no models/paths baked in: `JUDGE_SERVE_MODEL`, `JUDGE_SERVED_NAME`, `JUDGE_SERVE_PORT`, `JUDGE_SERVE_TP_SIZE`, `JUDGE_SERVE_GPUS`, `JUDGE_SERVE_EXTRA_ARGS` (e.g. `--max-model-len 8192 --gpu-memory-utilization 0.9`), `JUDGE_API_KEY`, `RECIPE`. **Router-balanced judge**: set `MODEL_LAUNCH_DIR` to the SwissAI model-launch checkout (needs `SERVER_WORKERS>=2`); the replica topology is `SERVER_WORKERS` (replicas), `SERVER_NODES_PER_WORKER`, `SERVER_NODES` (total → `submit_job.py --slurm-nodes`), `SERVER_TP_SIZE` (a vLLM arg, passed inside `--framework-args`), `SERVER_FRAMEWORK`, with SLURM envs `MODEL_LAUNCH_ENV` (workers), `ROUTER_ENV` (router), and `MODEL_LAUNCH_RUN_ENV` (the env that *runs* `submit_job.py`; the reference uses `activeuf`). Note the TP knob differs by backend — `JUDGE_SERVE_TP_SIZE` (single-node) vs `SERVER_TP_SIZE` (model-launch). The training job is replica-agnostic; its only judge-load knob is `env.online_dpo_judge.max_concurrency` (the judge backend/prompt config surface — `type`, `aspects`, … — is documented in `docs/apertus-dpo.md`).

For an end-to-end run on the **reference MaxMin online prompt set**, point `RECIPE` at `examples/configs/recipes/llm/online-dpo-apertus1p5-8b-maxmin-megatron.yaml` (the datasets + prompt-only loader are documented in `docs/apertus-dpo.md`). The concrete launcher `infra/slurm/cscs/launch_online_dpo_maxmin.sh` pins the reference's **Qwen3.6-27B** judge + this recipe (single-node judge by default; `MODEL_LAUNCH_DIR` + `SERVER_WORKERS=8` reproduce the reference's 8-replica router) — run with `JUDGE_API_KEY=$KEY infra/slurm/cscs/launch_online_dpo_maxmin.sh`.

**Rollout inspection** (per-step `<log_dir>/online_dpo_rollouts_step<N>.jsonl` dumps, capped by `online_dpo.num_logged_rollouts`) and **validation** (opt-in held-out judge eval via `grpo.val_period`/`val_at_start`/`val_at_end` + a val split) are documented in `docs/apertus-dpo.md` (§Logging & rollout inspection, §Validation).

## Submit from inside a compute node (coding agents)

This subsection is specifically for a **coding agent** (e.g. Claude Code / VS Code) running **inside a compute-node Container Engine (CE) session** — it cannot reach a login node, and its CE container ships only `libjson-c.so.3`.
Two things break as a result:

- `sbatch`/`srun` can't load the Slingshot/pyxis plugins (`libjson-c.so.5: cannot open shared object file`) — the plugins are loaded by the *client*, which here runs inside the lib-less container.
- the CE session's inherited pyxis `--environment` collides with the launcher's per-`srun` one (`--environment specified multiple times`).

To submit from inside a compute node, feed the client the shared `libjson-c.so.5` from the wheelhouse and unset the inherited pyxis options:

```bash
env -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts \
    LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:$LD_LIBRARY_PATH \
    sbatch infra/slurm/cscs/probe_grpo_fixgate.slurm
```

This path is **only** for agents/automation that must submit from a compute node; humans should submit from a login node (above), where none of this is needed.
See <https://docs.cscs.ch/software/container-engine/run/>.
