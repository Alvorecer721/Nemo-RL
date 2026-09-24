# Apertus remote Gym probes

The three Slurm entry points run Apertus 1.5 8B with the vLLM 0.29.0 image,
local NeMo-Gym agents, and the hosted resources at `gym-dev`. They are designed
to be submitted one at a time; each training job performs 40 GRPO steps.

From the NeMo-RL checkout on a CSCS login node, submit them in this order,
waiting for each job to complete before submitting the next:

```bash
sbatch --chdir="$PWD" infra/slurm/cscs/prepare_gym_math_with_judge_40step.slurm
MATH_WITH_JUDGE_TRAIN_PATH="$PWD/outputs/nemo_gym/math_with_judge/<prepare-job-id>/train.jsonl" \
  sbatch --chdir="$PWD" --export=MATH_WITH_JUDGE_TRAIN_PATH \
  infra/slurm/cscs/probe_grpo_gym_math_with_judge_remote_40step.slurm
sbatch --chdir="$PWD" infra/slurm/cscs/probe_grpo_gym_blackjack_remote_40step.slurm
sbatch --chdir="$PWD" infra/slurm/cscs/prepare_gym_workspace_mbpp.slurm
sbatch --chdir="$PWD" infra/slurm/cscs/train_grpo_gym_workspace_mbpp.slurm
```

All jobs use `infra/slurm/cscs/environments/nemo_rl_vllm029.toml` and the `preemptable`
partition. The two preparation jobs run inside their own allocations; neither
training launcher performs a remote-server preflight.

The training launcher rebuilds driver and worker environments from the checked-out
lockfile. It does not bypass the container dependency check. With an image built
for the exact checkout, `NRL_FORCE_REBUILD_VENVS=false` allows environment reuse;
the normal dependency check still applies. Pass submission-time overrides through
`sbatch --export` explicitly because these wrappers use `--export=NONE`.

Workspace runs one prompt with four generations per training step, so it opens
at most four stateful remote sessions at once. This is intentional for the
hosted Workspace gateway; do not add a retry around its state-changing
`/step` requests.

The hosted services still require integer usage-detail fields even though vLLM
correctly reports two counts as unknown. The remote configurations enable a
narrow client compatibility conversion only for their `/verify` or `/step`
requests. Remove that setting after the services are redeployed with Gym
`b1e0f67c` or newer.

Logs are written as `logs/grpo_gym_<task>_remote_40_<jobid>.run.log`; the
generation-KL gate also records every run in `results/probe_ledger.tsv`. The
Math probe validates the hosted deterministic verifier; it does not claim to
exercise a server-side LLM-judge fallback.

## W&B and generated rollouts

All three 40-step recipes enable the existing W&B logger in the `nemo-gym`
project. The entity comes from your W&B configuration or `WANDB_ENTITY`.
No additional logging service is needed.

Each training step reports scalar task metrics to W&B. A recipe can also log
per-rollout W&B Tables. Use `log_nemo_gym_rollout_tables` for the filtered
projection: it contains decoded agent actions, visible conversation turns,
token counts, scalar rewards, and selected provenance, but excludes raw Gym
result metadata. Do not enable raw `log_nemo_gym_full_result_tables` for an
environment that can return secrets or hidden tests. The native
`train_data_step<N>.jsonl` files are also retained for every step, including
decoded conversation content, rewards, token IDs, and loss masks.

Recipes may additionally set `env.log_nemo_gym_rollouts_jsonl: true`. This
writes `rollouts/step_<N>.jsonl` below the same experiment directory, with one
record per rollout. It contains the decoded agent actions, visible workspace
turns, token counts, scalar rewards, and selected dataset provenance. It
deliberately excludes the raw Gym result and verifier metadata, so hidden-test
sources are not copied into this local artifact. This optional dump is disabled
in these recipes; enable it explicitly when needed.

Authentication must be available **inside the compute container**. The wrappers
use `#SBATCH --export=NONE`; merely exporting a key on the login node is not
enough. Existing container-visible W&B credentials work. If credentials are
already exported in the submission environment, override the batch export
setting so they reach the compute node:

```bash
sbatch --chdir="$PWD" --export=ALL \
  infra/slurm/cscs/probe_grpo_gym_blackjack_remote_40step.slurm
```

The launcher explicitly forwards the batch environment to the Gym container
step. No credentials are stored in YAML or supplied in command arguments.

Reservation policy is launcher-specific. The Blackjack POC below deliberately
uses the preemptable partition without a reservation, as authorized for this run.
All Python, dataset preparation, and training must run inside a new allocation
using this container, never on the login node.

## Dataset choices

Blackjack learns from fresh simulator episodes. The current server samples a
new deck sequence on every reset, so repeating episode requests is intentional;
a larger file of prompts adds no new game states. Its reset currently ignores
seeds in metadata. Do not describe train/validation prompt files as reproducible
held-out game splits, or claim this custom server is identical to Farama's
[Blackjack-v1](https://gymnasium.farama.org/environments/toy_text/blackjack/).
The Blackjack POC therefore uses its five valid Gym request rows repeated 32
times: four prompts and four generations per update produce 16 independently
dealt games per step and 640 total games in 40 steps. The recipe explicitly
sets those values and writes the native training JSONL. Enable
`env.log_nemo_gym_rollouts_jsonl` for additional per-game records.

The default Apertus probe's HF source is
[`agentica-org/DeepScaleR-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset),
a math post-training dataset. It cannot replace Blackjack's request JSONL:
`NemoGymDataset` requires each row to name `blackjack_gymnasium_agent`, and the
remote service ignores the math problem when it deals a game. Converting its
rows to identical `Deal me in` requests would add I/O without creating a
Blackjack dataset. Keep it only for math training; fresh Blackjack resets are
the POC's online data source.

The Blackjack POC uses the `preemptable` partition without a reservation:

```bash
sbatch --chdir="$PWD" infra/slurm/cscs/probe_grpo_gym_blackjack_remote_40step.slurm
```

The wrapper uses `infra/slurm/cscs/environments/nemo_rl_vllm029.toml` and runs 40 steps with
16 rollouts per step.

For the existing Workspace service, the optional MBPP recipe uses the published
[MBPP full dataset](https://huggingface.co/datasets/google-research-datasets/mbpp):
374 training tasks and 90 validation tasks, preserving the official split.
This is function-level Python implementation, not repository issue repair.
Reference solutions are excluded from the task files. One assertion is shown
as a public interface example; remaining distinct assertions become withheld
pytest tests. The service returns their pass fraction, so this adaptation is
not the official MBPP benchmark metric. MBPP is attributed under CC BY 4.0 in
the generated manifest, which also records source revision, task IDs and hashes.

Prepare the data in its dedicated compute job:

```bash
sbatch --chdir="$PWD" infra/slurm/cscs/prepare_gym_workspace_mbpp.slurm
```

The output directory must be new; existing data is never overwritten. The
prepared files are `mbpp_train.jsonl`, `mbpp_validation.jsonl` and `manifest.json`.
Then submit from the login node:

```bash
sbatch --chdir="$PWD" infra/slurm/cscs/train_grpo_gym_workspace_mbpp.slurm
```

This launcher uses the `preemptable` partition, an
8192-token context, eight agent actions, 512 generated tokens per action and
four concurrent rollouts. It enables W&B and retains native training JSONL. It
runs at most 40 training steps; the full training split is available but a
40-step run with one prompt per step does not cover it all. Validation is
configured separately but the inherited probe has periodic validation disabled.

Blackjack and Workspace enable native Gym token capture and verified prefix
supply. Each continuation sends the exact previous tokens to NeMo-RL's vLLM
endpoint and checks the generation-time prompt IDs. Capture uses node-local
`/tmp/nemo_gym_token_capture`; set `NEMO_GYM_TOKEN_CAPTURE_DIR` to another
absolute path if needed. The runner retires each frozen capture only after the
rollout consumer accepts its result; failed or abandoned deliveries retain their
records for diagnosis. These recipes use the synchronous Gym runner; the
SingleController external-staging path has its own capture configuration.

## Optional stopping-token experiment

The standard MBPP recipe retains the checkpoint's original EOS IDs `[2, 68, 72]`.
Prefix reconstruction must preserve the token actually sampled, including its
ending. Removing `72` changes generation stopping and is only a diagnostic
experiment; it is not a replacement for the prefix correction.

Prepare the diagnostic overlay and then launch it from the same checkout:

```bash
sbatch --chdir="$PWD" infra/slurm/cscs/prepare_apertus_mbpp_no_eos72_checkpoint.slurm
# Wait for preparation to finish before submitting training.
sbatch --chdir="$PWD" infra/slurm/cscs/train_grpo_gym_workspace_mbpp_no_eos72.slurm
```

Both wrappers use `OVERLAY_CHECKPOINT`, defaulting to this checkout's
`outputs/model_overlays/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200-mbpp-no-eos72`.
For a custom destination, set `OVERLAY_CHECKPOINT` and add
`--export=OVERLAY_CHECKPOINT` to both submissions. Direct invocation of the
diagnostic YAML also requires this environment variable.
