# CSCS Slurm Probes

This directory contains the Clariden/GH200 Slurm wrappers used to build, probe, and train with the NeMo-RL `nvcr.io/nvidia/nemo-rl:v0.7.0` container on Slingshot.

The default container environment is `docker/nemo_rl.toml` in this checkout. The wrappers set `CUDA_CACHE_PATH` and Hugging Face cache paths in shell code because TOML values are not shell-expanded by Pyxis/EDF.

## Submit from a login node (humans)

Submit these from a **login node** (e.g. `clariden-ln001`).
The wrappers follow CSCS guidance — `--environment` lives on each `srun`, never on `#SBATCH` — so the batch script and the Slurm client run on the **host**, which has the system libraries the Slingshot/pyxis plugins need (`libjson-c.so.5`); only the task is containerized.
A login node has those libraries natively, so nothing extra is required.

Run from the repository root after creating the log directory:

```bash
mkdir -p logs
sbatch infra/slurm/cscs/build_xielu.slurm                       # one-time CUDA xIELU kernel build
sbatch infra/slurm/cscs/probe_grpo_fixgate.slurm                # 3-step online GRPO smoke (sync, colocated)
sbatch infra/slurm/cscs/probe_grpo_async.slurm                  # same gate on the async non-colocated path
sbatch infra/slurm/cscs/probe_nemo_rl_dpo_megatron_apertus.slurm  # DPO probe + provider gate
sbatch infra/slurm/cscs/submit_nemo_rl_dpo.slurm                # full MaxMin DPO run
```

Useful overrides:

```bash
CONTAINER_ENV=$HOME/.edf/nemo_rl.toml sbatch infra/slurm/cscs/probe_grpo_fixgate.slurm
RECIPE=examples/configs/recipes/llm/dpo-apertus1p5-8b-maxmin-megatron.yaml sbatch infra/slurm/cscs/submit_nemo_rl_dpo.slurm
```

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

## Troubleshooting first launches

To skip first-launch building entirely, pre-provision the checkout once with
`sbatch --chdir=<repo> infra/slurm/cscs/prepare_env.slurm` — it builds `.venv` plus the worker
venvs behind an flock (safe to rerun; concurrent submissions queue instead of deadlocking), and
every later training job starts hot.

- **Two cold-cache jobs deadlock on a source build** (`Failed to acquire lock on the distribution
  cache`, 300 s timeout): run the first launch alone (or use `prepare_env.slurm` above); every
  later job reuses the built wheels.
- **`Pretrained run config not found ... iter_0000000/run_config.yaml` on rank>0**: a stale,
  half-written conversion cache. Delete the `$HF_HOME/nemo_rl/model__*` directory for that
  checkpoint and rerun — rank 0 reconverts cleanly.
- **Worker import errors like `module 'torch' has no attribute 'Tensor'` right after venv
  creation**: a crashed builder left a partial worker venv. Since the readiness-marker fix a
  venv without `NEMO_RL_VENV_READY` is repaired in place on the next run and a stale
  `STARTED_ENV_BUILDER` claim expires after `NRL_VENV_BUILD_TIMEOUT_SECS` (default 3600 s), so
  plain resubmission heals it; `rm -rf <repo>/venvs/<worker-name>` remains the manual override.
- **A personal `uv` in your dotfiles shadows the image's** (`/root/.local/bin/uv`) and its cache
  keys may miss the image-seeded wheel cache, causing one-time source rebuilds. Harmless but slow;
  remove the personal uv from PATH inside jobs (or accept the one-time builds into your own cache).
- **Never submit with `sbatch --export=ALL`**: it leaks the interactive session's environment
  (including a different `uv`) into the job and breaks dependency resolution. The launchers use
  `--export=NONE`; pass parameters the way `probe_grpo_async.slurm` does (a wrapper that exports
  them in the job body).
