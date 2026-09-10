# Apertus remote Gym smoke

The remote recipe runs Apertus 1.5 8B generation and the Gym simple agent locally,
and sends math verification to `https://gym-dev.swissai.svc.cscs.ch/math_with_judge`.
It uses the five-row fixture in this directory. Resource settings in the recipe
do not reconfigure the already deployed verifier.

From the NeMo-RL checkout on a CSCS login node:

```bash
NEMO_RL_DIR=$(pwd -P)
export NEMO_GYM_EXTRA_ROOTS="$NEMO_RL_DIR/3rdparty/Gym-workspace/Gym"
uv lock
uv lock --check
mkdir -p "$NEMO_RL_DIR/logs"
sbatch \
  --chdir="$NEMO_RL_DIR" \
  --export=ALL,REPO_DIR="$NEMO_RL_DIR",RUNTIME_REPO_DIR="$NEMO_RL_DIR",CONTAINER_ENV="$NEMO_RL_DIR/docker/nemo_rl_vllm026_ncclext.toml",RECIPE="$NEMO_RL_DIR/examples/configs/recipes/llm/grpo-apertus1p5-8b-1n4g-megatron-probe-gym-remote.yaml",MAX_STEPS=1,MIN_GEN_KL_SAMPLES=1,NRL_FORCE_REBUILD_VENVS=true,PROJECT_VENV_DIR="$NEMO_RL_DIR/.venv-gym-remote",WORKER_VENV_DIR="$NEMO_RL_DIR/venvs-gym-remote" \
  "$NEMO_RL_DIR/infra/slurm/cscs/probe_grpo_gym.slurm"
```

The source port changes the Gym submodule commit but not its dependency metadata;
`uv lock` can therefore succeed without changing `uv.lock`. The image fingerprint
still differs. `NRL_FORCE_REBUILD_VENVS=true` uses NeMo-RL's supported rebuild path;
the dedicated environment directories keep this probe separate from other runs.
`NEMO_GYM_EXTRA_ROOTS` makes component imports prefer this checkout over any
copies installed in the image.
Do not use these directories concurrently with another rebuild. Once an image
has been rebuilt for the new pin, omit the force-rebuild flag.

Success requires a completed training step and the launcher's generation-KL gate,
not merely a submitted job. Inspect `logs/grpo_fixgate_<jobid>.run.log` and
`results/probe_ledger.tsv`. This training smoke covers math; Gym's opt-in remote
resource integration tests cover math rewards and blackjack/SWE session state.

Run the resource contract smoke explicitly from the pinned Gym checkout:

```bash
cd 3rdparty/Gym-workspace/Gym
uv run python -m scripts.smoke_remote_resources
```

This makes real requests to the deployed services, including a temporary SWE
workspace. It does not deploy the ported server or invoke a policy model.

For submission from an existing compute-container session, also apply the
Slurm client environment cleanup documented in `infra/slurm/cscs/README.md`.
If the batch-side `srun` still cannot load `libjson-c.so.5`, set the wheelhouse
library path inside the batch shell before sourcing the launcher; setting it
only on the submission client did not suffice in the compute-container test.
