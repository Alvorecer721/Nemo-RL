#!/bin/bash
# Create the MBPP EOS ablation checkpoint on a compute node. The model files
# stay read-only through absolute symlinks; only generation_config.json is copied.
#SBATCH --account=infra01
#SBATCH --job-name=ap1p5-mbpp-no72-prep
#SBATCH --output=logs/ap1p5_mbpp_no72_prep_%j.out
#SBATCH --error=logs/ap1p5_mbpp_no72_prep_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=preemptable
#SBATCH --export=NONE
#SBATCH --no-requeue

set -euo pipefail

REPO_DIR=${REPO_DIR:-$PWD}
export SOURCE_CHECKPOINT=/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200
export OVERLAY_CHECKPOINT="$REPO_DIR/outputs/model_overlays/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200-mbpp-no-eos72"
export CONTAINER_ENV=${CONTAINER_ENV:-$REPO_DIR/docker/nemo_rl_vllm026_ncclext.toml}

mkdir -p "$REPO_DIR/logs"

srun --environment="$CONTAINER_ENV" --chdir="$REPO_DIR" --ntasks=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" --export=ALL bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail

source_config="$SOURCE_CHECKPOINT/generation_config.json"
overlay_config="$OVERLAY_CHECKPOINT/generation_config.json"

[[ -d "$SOURCE_CHECKPOINT" ]] || {
  echo "ERROR: source checkpoint is absent: $SOURCE_CHECKPOINT" >&2
  exit 1
}
[[ -f "$source_config" ]] || {
  echo "ERROR: source generation config is absent: $source_config" >&2
  exit 1
}
[[ ! -e "$OVERLAY_CHECKPOINT" && ! -L "$OVERLAY_CHECKPOINT" ]] || {
  echo "ERROR: refusing to modify existing overlay: $OVERLAY_CHECKPOINT" >&2
  exit 1
}

source_sha_before=$(sha256sum "$source_config" | awk '{print $1}')
mkdir -p "$(dirname "$OVERLAY_CHECKPOINT")"
mkdir "$OVERLAY_CHECKPOINT"

while IFS= read -r -d '' entry; do
  ln -s "$entry" "$OVERLAY_CHECKPOINT/$(basename "$entry")"
done < <(find "$SOURCE_CHECKPOINT" -mindepth 1 -maxdepth 1 ! -name generation_config.json -print0)

python - "$source_config" "$overlay_config" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
overlay_path = Path(sys.argv[2])
source = json.loads(source_path.read_text())
eos_token_ids = source.get("eos_token_id")
if eos_token_ids != [2, 68, 72]:
    raise SystemExit(f"unexpected source eos_token_id: {eos_token_ids!r}")

overlay = dict(source)
overlay["eos_token_id"] = [token_id for token_id in eos_token_ids if token_id != 72]
if overlay["eos_token_id"] != [2, 68]:
    raise SystemExit(f"unexpected overlay eos_token_id: {overlay['eos_token_id']!r}")

overlay_path.write_text(json.dumps(overlay, indent=2) + "\n")
written = json.loads(overlay_path.read_text())
if written != overlay:
    raise SystemExit("overlay generation config did not round-trip")
if {key: value for key, value in source.items() if key != "eos_token_id"} != {
    key: value for key, value in written.items() if key != "eos_token_id"
}:
    raise SystemExit("unexpected non-EOS generation config change")
PY

[[ -f "$overlay_config" && ! -L "$overlay_config" ]] || {
  echo "ERROR: overlay generation config must be a regular file" >&2
  exit 1
}
[[ "$(stat -c '%d:%i' "$source_config")" != "$(stat -c '%d:%i' "$overlay_config")" ]] || {
  echo "ERROR: overlay generation config must not be hard-linked to source" >&2
  exit 1
}
[[ "$source_sha_before" == "$(sha256sum "$source_config" | awk '{print $1}')" ]] || {
  echo "ERROR: source generation config changed" >&2
  exit 1
}

while IFS= read -r -d '' entry; do
  link="$OVERLAY_CHECKPOINT/$(basename "$entry")"
  [[ -L "$link" && "$(readlink "$link")" == "$entry" ]] || {
    echo "ERROR: overlay entry is not the expected source symlink: $link" >&2
    exit 1
  }
done < <(find "$SOURCE_CHECKPOINT" -mindepth 1 -maxdepth 1 ! -name generation_config.json -print0)

echo "Source generation config SHA256: $source_sha_before"
echo "Overlay generation config SHA256: $(sha256sum "$overlay_config" | awk '{print $1}')"
echo "Overlay path: $OVERLAY_CHECKPOINT"
echo "Generation-config difference:"
diff -u "$source_config" "$overlay_config" || true
echo "Overlay verification passed."
CONTAINER_SCRIPT
