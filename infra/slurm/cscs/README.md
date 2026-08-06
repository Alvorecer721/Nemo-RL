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
sbatch infra/slurm/cscs/probe_grpo_fixgate.slurm                # 3-step online GRPO smoke
sbatch infra/slurm/cscs/probe_nemo_rl_dpo_megatron_apertus.slurm  # DPO probe + provider gate
sbatch infra/slurm/cscs/submit_nemo_rl_dpo.slurm                # full MaxMin DPO run
```

Useful overrides:

```bash
CONTAINER_ENV=/users/xyixuan/.edf/nemo_rl.toml sbatch infra/slurm/cscs/probe_grpo_fixgate.slurm
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
