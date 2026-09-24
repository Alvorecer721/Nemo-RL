# Upstream Sync Implementation Plan (2026-09-10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline, one controller) for this plan. The three merges are sequential by construction (Core -> Bridge -> NeMo-RL) and share conflict context; do not fan them out. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the frozen upstream snapshots of Megatron-Core, Megatron-Bridge and NeMo-RL into the fork lineage while retaining the qualified Apertus 70B and GLM-5.1 behavior on CSCS GH200, then requalify one image and open (not merge) a sync PR with a carry/drop ledger.

**Architecture:** Merge-based sync in dependency order inside one isolated worktree. Each repository gets exactly one merge commit plus, where needed, one follow-up commit per reconciled capability. Behavior is carried by verifying the merged tree against a ledger of named capabilities, not by replaying historical patch chains. Dependency identity (pins, lock, fingerprint) changes once, at the end of the source phase, and is qualified once.

**Tech Stack:** git (rerere on), uv 0.11.28 in-image, Podman/Enroot image builder + receipt-based assembly (build PR), Slurm on GH200, pytest, ruff, pyrefly.

**Spec:** `session/20260910_git_cleanup/claude-sync-handoff.md` (root checkout, untracked) with companions `feature-audit.md` and `inventory-tables.md`. Frozen SHAs: `session/20260910_152226/targets.md`.

## Global Constraints

- Base: `build/2026-09-10-image-assembly-split` = `c00cb93cf7f0683fe12c530c0780cd0d2d6c007b`. Never base on the old root, the Sep 5 foundation topic alone, or `build/2026-09-10-image-refresh` alone.
- Upstream targets (frozen; do not refetch during qualification): NeMo-RL `c49d53e2e46e95ba645870fead6d48e36c37259b`, Bridge `4386117130f9e86024fc694e77ccc9c903899b9d`, Core `c9b53d0a87cb926f47115259593ecfeb351ca29f`, Gym `fd5e84d6b1c485c80e7ae61553bbd485611c03b4`.
- Keep TE `27486e03cfc1fa41f6932dcecdc47c71c47eac3e` (2.18.0+27486e03), vLLM 0.26.0, flashinfer 0.6.14, cutlass-dsl 4.6.0, nccl-extensions. Every other dependency change is adopted deliberately and reported with the layer it invalidates.
- Rollout PP stays 1. Do not enable `refit_with_reload_api`, NCCL M2N/EP, HybridEP, Mooncake GDR, or Megatron's dataset packing scheduler.
- Never set `NRL_IGNORE_VERSION_MISMATCH`. Never `git clean`, force-remove submodules, delete checkpoints, or rewrite published history. Do not edit tracked files in the root checkout or any other worktree.
- Fork submodule pins must be reachable from a branch on the fork before any PR references them (`integrate/2026-09-10-upstream-sync` in `Alvorecer721/Megatron-LM` and `Alvorecer721/Megatron-Bridge`).
- Commits: DCO sign-off (`-s`), no GPG. PR is opened only after review and is not merged.
- If the build PR is squash-merged before the sync PR opens: create a fresh branch from the new main and redo the NeMo-RL merge there (rerere replays resolutions); do not rebase the merge commit or replay build commits.

---

### Task 1: Merge upstream Core into the fork Core

**Files:**
- Repo: `.worktrees/upstream-sync/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM` (non-shallow store)
- Expected touched by merge: `megatron/core/distributed/fsdp/...` (upstream #6924, #6484 only)
- Fork-owned, must survive: `megatron/core/optimizer/__init__.py`, `megatron/core/optimizer/distrib_optimizer.py`, `megatron/core/transformer/multi_latent_attention.py`, `tests/unit_tests/dist_checkpointing/optimizer/test_distrib_optimizer_load_state.py`, `tests/unit_tests/transformer/test_multi_latent_attention_imports.py`

**Interfaces:**
- Produces: `CORE_SHA` (merge commit) consumed by Task 2 as `.main.commit` and gitlink.

- [x] **Step 1: Branch from the fork pin and fetch the target**

```bash
C=.worktrees/upstream-sync/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
git -C $C remote add upstream https://github.com/NVIDIA/Megatron-LM.git 2>/dev/null || true
git -C $C fetch upstream c9b53d0a87cb926f47115259593ecfeb351ca29f
git -C $C switch -c integrate/2026-09-10-upstream-sync 1e5025f8521cb5b6c3c9276429fc012f45b5c291
```

- [x] **Step 2: Merge and confirm no conflicts**

```bash
git -C $C merge --no-ff -s ort c9b53d0a87cb926f47115259593ecfeb351ca29f -m "merge: sync Megatron-Core with upstream c9b53d0a" -S no 2>&1 | tail -5
```
Expected: clean merge (merge-tree dry run showed none). If a conflict appears, stop and resolve toward keeping the optimizer topic exactly as in `session/20260905_114410/source_cleanup/carry/mcore/0001-*.patch`.

- [x] **Step 3: Verify fork behavior survived**

```bash
git -C $C grep -n "_can_reuse_precision_aware_checkpoint_state\|_load_optimizer_param_groups_without_state" -- megatron/core/optimizer/distrib_optimizer.py
git -C $C grep -n "store_param_remainders and p.dtype == torch.bfloat16" -- megatron/core/optimizer/__init__.py
git -C $C grep -n "QuantizedTensor" -- megatron/core/transformer/multi_latent_attention.py
git -C $C diff --stat b1fe7599e18a213292600177ad7ff290a1127160 HEAD -- megatron/core/optimizer megatron/core/transformer/multi_latent_attention.py tests/unit_tests/dist_checkpointing/optimizer/test_distrib_optimizer_load_state.py tests/unit_tests/transformer/test_multi_latent_attention_imports.py
```
Expected: all three greps hit; the diff-stat equals the fork's 5-file net delta (`+306/-9`).

- [x] **Step 4: Sign off and record**

```bash
git -C $C commit --amend -s --no-edit
CORE_SHA=$(git -C $C rev-parse HEAD); echo $CORE_SHA
```
Append `CORE_SHA` to `session/20260910_152226/targets.md` under a "Produced pins" heading.

---

### Task 2: Merge upstream Bridge into the fork Bridge

**Files:**
- Repo: `.worktrees/upstream-sync/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge` (currently shallow, main-only refspec; must be unshallowed first)
- Conflicts predicted: `.main.commit`, `3rdparty/Megatron-LM` (gitlink)
- Fork-owned, must survive: `src/megatron/bridge/models/apertus/*` (10 files), `src/megatron/bridge/models/__init__.py` (Apertus registration), `src/megatron/bridge/training/checkpointing.py` (configured `ckpt_load_validate_sharding_integrity` forwarding), `tests/unit_tests/models/apertus/*`, `tests/unit_tests/training/test_checkpointing.py`, `.gitmodules` (fork Core URL)

**Interfaces:**
- Consumes: `CORE_SHA` from Task 1.
- Produces: `BRIDGE_SHA` consumed by Task 3 as the NeMo-RL gitlink.

- [x] **Step 1: Unshallow and fetch**

```bash
B=.worktrees/upstream-sync/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git -C $B config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
git -C $B fetch --unshallow origin
git -C $B remote add upstream https://github.com/NVIDIA-NeMo/Megatron-Bridge.git 2>/dev/null || true
git -C $B fetch upstream 4386117130f9e86024fc694e77ccc9c903899b9d
git -C $B rev-parse --is-shallow-repository   # expected: false
git -C $B switch -c integrate/2026-09-10-upstream-sync 1d9a69fde22cb1e607f7ff84e4b62cc9e890dc75
```

- [x] **Step 2: Merge, resolve the two metadata conflicts toward `CORE_SHA`**

```bash
git -C $B merge --no-ff 4386117130f9e86024fc694e77ccc9c903899b9d -m "merge: sync Megatron-Bridge with upstream 4386117" ; git -C $B status --short | grep -E '^(UU|AA|DU|UD)'
printf '%s\n' "$CORE_SHA" > $B/.main.commit
git -C $B -c protocol.file.allow=always submodule update --init 3rdparty/Megatron-LM || true
git -C $B/3rdparty/Megatron-LM checkout -q "$CORE_SHA"
git -C $B add .main.commit 3rdparty/Megatron-LM
git -C $B status --short | grep -E '^(UU|AA)' && echo "UNRESOLVED" || git -C $B commit -s --no-edit
```
Rule: `.gitmodules` must still point at `https://github.com/Alvorecer721/Megatron-LM.git` (the fork carries Core commits). If the merge changed it, restore the fork URL before committing.

- [x] **Step 3: Verify Apertus and checkpoint behavior survived, and probe upstream API drift**

```bash
git -C $B diff --stat 75d7b9eb89b05b7edbef7daf931045c38c6ca9df HEAD -- src/megatron/bridge/models/apertus src/megatron/bridge/models/__init__.py tests/unit_tests/models/apertus | tail -1
git -C $B grep -n "ckpt_load_validate_sharding_integrity" -- src/megatron/bridge/training/checkpointing.py
git -C $B grep -n "APERTUS_XIELU_STATIC_STATE_OWNER" -- src/megatron/bridge/models/apertus
# API drift: every symbol the Apertus provider/bridge imports must still exist upstream
git -C $B grep -h -E "^from megatron|^import megatron" -- src/megatron/bridge/models/apertus/*.py | sort -u
```
For each imported module/symbol, confirm it exists at HEAD (`git -C $B grep -n "def <name>\|class <name>" -- src/`). Specifically re-check against upstream commits `2fdc85e2b` (tokenizer sp args removed), `a420021a5` (model parallel init for builder loads), `d71945307` (local BF16 export iterator) and `12d27f63c` (pipeline layout clearing): none may change the Apertus provider signature or the `local_hf_param_specs` contract used by NeMo.

- [ ] **Step 4: CPU-side Bridge checks (in the container; see Task 4 for the sbatch template)**

Targets: `tests/unit_tests/models/apertus/test_apertus_bridge.py`, `tests/unit_tests/training/test_checkpointing.py`, and `pre-commit run --all-files`. Record pass/fail in the session timeline; GPU-only Apertus parity tests are listed as deferred to Task 6.

- [x] **Step 5: Record**

```bash
BRIDGE_SHA=$(git -C $B rev-parse HEAD); echo $BRIDGE_SHA
```
Append `BRIDGE_SHA` to `targets.md` under "Produced pins".

---

### Task 3: Merge upstream NeMo-RL and reconcile dependencies

**Files:**
- Repo: `.worktrees/upstream-sync`, branch `sync/2026-09-10-upstream-c49d53e`
- Conflicts predicted by `git merge-tree` (29): see groups below.
- Gitlinks: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge` -> `BRIDGE_SHA`; `3rdparty/Gym-workspace/Gym` -> `fd5e84d6b1c485c80e7ae61553bbd485611c03b4`.
- Dependency files: `pyproject.toml`, `uv.lock`, `docker/Dockerfile`, `tools/generate_fingerprint.py`, `infra/slurm/cscs/build_nemo_rl_image.slurm` (TE assertion / cache constants).

**Interfaces:**
- Consumes: `BRIDGE_SHA`.
- Produces: merge commit + reconciliation commits; a regenerated `uv.lock`; `LOCK_DELTA.md` in the session dir.

- [x] **Step 1: Start the merge**

```bash
W=.worktrees/upstream-sync
git -C $W merge --no-ff c49d53e2e46e95ba645870fead6d48e36c37259b -m "merge: sync NeMo-RL with upstream c49d53e2" ; git -C $W diff --name-only --diff-filter=U
```

- [x] **Step 2: Resolve group A, build/actor environments (upstream #4002, #4020, #4050 vs the build PR)**

Files: `docker/Dockerfile`, `nemo_rl/distributed/actor_environments.py` (add/add), `nemo_rl/distributed/ray_actor_environment_registry.py`, `nemo_rl/utils/prefetch_venvs.py`, `nemo_rl/utils/venvs.py`, `nemo_rl/modelopt/registry.py`, `tools/generate_fingerprint.py`, `pyrefly.toml`, `tests/unit/distributed/test_actor_environments.py`, `tests/unit/utils/test_prefetch_venvs.py`.

Rules:
1. The shared actor-environment manifest is ours (build PR); upstream #4002 introduced the same idea (`actor_environments.py` as the leaf module). Keep ONE module: take the fork file as the base, then port any upstream symbol that upstream callers import (`git -C $W grep -n "actor_environments" -- nemo_rl docker tools tests`), so both the fork Dockerfile/finalizer and upstream `virtual_cluster.py` resolve.
2. Adopt upstream #4020 mechanism (`PY_EXECUTABLES._resolve_system_overrides()`, `MODELOPT_*` entries in `virtual_cluster.py`; the 43-line removal from `modelopt/registry.py`; the interpreter guard at the top of `create_local_venv`). Re-apply the fork's `venvs.py` additions (readiness marker / claim logic) below the guard.
3. `docker/Dockerfile`: keep the fork's builder/assembly split intact (no installation in assembly). Port only upstream's HybridEP multi-node build args (#4038) if they do not add a compile step outside the builder stage; otherwise record as deferred. Port #4050's `NEMO_GYM_VLLM_VERSION` bump only if the fork Dockerfile still has that knob; check what Gym `fd5e84d6` pins (`git -C $W/3rdparty/Gym-workspace/Gym grep -n "vllm" -- pyproject.toml`).
4. `tools/generate_fingerprint.py`: union of both (upstream adds actor-list hashing; fork adds receipts). Fingerprint must include the recursive submodule SHAs.

Verify: `git -C $W grep -n "NEMO_RL_PY_EXECUTABLES_SYSTEM" -- nemo_rl` shows the single call site in `virtual_cluster.py`; `python -c "import nemo_rl.distributed.actor_environments as a; print(sorted(k for k in vars(a) if k.isupper()))"` runs.

- [x] **Step 3: Resolve group B, single controller (upstream #3837, #3923, #3924, #3978, #3730 vs fork SC guards/metrics)**

Files: `nemo_rl/algorithms/single_controller.py`, `single_controller_utils/setup.py`, `single_controller_utils/utils.py`, `nemo_rl/experience/payload.py`, `nemo_rl/experience/rollout_manager.py`, `tests/unit/experience/test_payload.py`, `test_rollout_manager.py`, `test_rollouts.py`.

Rules:
1. Upstream owns structure (token sink, rollout checkpointing, MOPD). Take upstream's version of each conflicted region first, then re-insert the fork's capabilities as additive hunks, each anchored to a named behavior:
   - fused/selected-token logprob capability guards and resolver (`single_controller_utils/config.py` is not conflicted; `setup.py`/`utils.py` call sites are),
   - NumPy logprob-tail reduction / tail-evidence metrics,
   - rollout quality metrics and no-signal-group metrics,
   - occurrence-ID group alignment for GRPO/ALP (`single_controller_utils/rewards.py`, not conflicted; its callers are),
   - worker-side serializable finish-step metrics.
2. Keep upstream's advantage clipping placement (after OPD statistics); do not re-introduce the older fork placement.
3. Rollout checkpointing (#3923/#3924) is accepted as upstream code; verify with the unit suite only. It is not a substitute for Task 7's optimizer-state resume check.

Verify: `uv run --group test python -m pytest tests/unit/single_controller tests/unit/experience -q` in the container (Task 4 template).

- [x] **Step 4: Resolve group C, refit/transport (upstream #3651, #3659, #3501, #3809 vs fork local-view bulk refit / int16 transport / worker metadata)**

Files: `nemo_rl/data_plane/worker_mixin.py`, `nemo_rl/utils/packed_tensor.py`, `nemo_rl/models/policy/workers/megatron_policy_worker.py`, `nemo_rl/models/policy/__init__.py`, `tests/unit/models/generation/test_nccl_reshard_backend.py`, `tests/unit/models/policy/test_split_api_wrappers.py`.

Rules:
1. `packed_tensor.py`: upstream #3651 rewrote the consumer (iterator cleanup, fallback lifecycle). Take upstream, then confirm 0-dim/scalar shapes and unaligned slices still round-trip (`tests/unit/utils/test_packed_tensor.py`, both versions of the tests must pass).
2. `worker_mixin.py`: upstream broadcasts int16 as int32 then narrows; fork has PR35's value-preserving transport. Keep the fork's dtype/value-preserving semantics if the tests in `tests/unit/data_plane/test_leader_broadcast.py` demand them; otherwise adopt upstream and keep the fork tests. Record the choice in the ledger.
3. `megatron_policy_worker.py`: keep `_iter_local_hf_param_shards` on `task.local_hf_param_specs()`, the o_proj/vocab-parallel bulk whitelist, direct-target shape checks, and the wire-safe `nccl_reshard_refit_info` conversion. Accept upstream's worker-extension FQN hooks (#3809) and reload-API plumbing as dormant code; `refit_with_reload_api` stays false for `nccl_reshard`.
4. `models/policy/__init__.py`: union of config keys (upstream adds lora warm-start, quantizer factory, extension FQNs, MOPD; fork adds its NotRequired keys). Exemplar YAML defaults follow config-conventions.

Verify: `pytest tests/unit/weight_sync tests/unit/models/generation/test_nccl_reshard_backend.py tests/unit/utils/test_packed_tensor.py tests/unit/data_plane -q`.

- [x] **Step 5: Resolve group D, data/algorithms/tools**

Files: `nemo_rl/data/processors.py` (fork `render_single_turn_prompt` single-template-call fix vs upstream #3345/#4009 additions: keep both; the fork helper is used by both math processors), `nemo_rl/algorithms/dpo.py` (upstream #3345 vs fork DPO fused-logprob knobs: union), `tools/config_cli.py` (fork `_override_`-aware minimizer vs upstream #3917: keep fork logic, add upstream's new sections), `tests/test_suites/disabled.txt` (union of lines).

Verify: `pytest tests/unit/data/test_data_processor.py tests/unit/algorithms/test_dpo.py tests/unit/tools -q`.

- [x] **Step 6: Gitlinks and pyproject reconciliation**

```bash
git -C $W/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge checkout -q "$BRIDGE_SHA"
git -C $W/3rdparty/Gym-workspace/Gym fetch -q origin fd5e84d6b1c485c80e7ae61553bbd485611c03b4 && git -C $W/3rdparty/Gym-workspace/Gym checkout -q fd5e84d6b1c485c80e7ae61553bbd485611c03b4
git -C $W add 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Gym-workspace/Gym
```
`pyproject.toml` decisions (auto-merged by git, but each is deliberate; write the resulting table into the ledger):

| Item | Decision |
|---|---|
| vLLM 0.26.0 / flashinfer 0.6.14 / cutlass-dsl 4.6.0 / TE 2.18 metadata | keep fork |
| `nccl-extensions` dependency | keep fork |
| `megatron-energon[av-decode]~=7.3` in `mcore` extra and `av` no longer excluded (upstream #3917) | adopt upstream; energon flows from Megatron-LM anyway and `av` is needed by the SFT v2 path; report as a new package in the mcore layer |
| Gym as workspace member + package-qualified conflicts (fork) vs editable path dep (upstream #4050) | adopt upstream shape unless the fork finalizer's Gym venv build depends on workspace membership; test with `uv lock` and the Gym venv selection in `tools/check_image_workers.py` |
| `flashinfer-cubin` source for both `sglang` and `vllm` extras | keep fork (needed for 0.6.14 on vllm) |
| new upstream extras/overrides (lora warm-start, MOPD, HybridEP, Mooncake GDR) | adopt; features stay disabled by config |

- [x] **Step 7: Regenerate `uv.lock` in-image and inspect the delta**

The login node cannot lock (submodule path deps only resolve inside the image). Submit a lock job using the container TOML of the last qualified image and the overlay script's lock phase pattern:

```bash
env -i PATH="$PATH" HOME="$HOME" USER="$USER" sbatch --chdir=$PWD/.worktrees/upstream-sync --export=PHASE=lock,BASE_SQSH=<qualified sqsh from targets.md> --time=00:40:00 infra/slurm/cscs/build_nemo_rl_overlay_image.slurm
```
Then:
```bash
git -C $W diff --stat uv.lock
git -C $W diff uv.lock | grep -E '^[-+]name = |^[-+]version = ' | paste - - | sort | uniq -c | sort -rn > session/20260910_152226/LOCK_DELTA.txt
```
Report every changed package identity and which layer it invalidates (hermetic deps layer: TE/FlashMLA/mamba compile inputs unchanged? If yes, note that only Python wheels move; if `torch`/`nvidia-*`/`transformer-engine` move, the native layer is invalid).

- [x] **Step 8: Commit the merge, then one commit per reconciliation group that needed post-merge edits**

```bash
git -C $W commit -s --no-edit            # merge commit
# follow-ups (only if edits were made after the merge commit):
git -C $W commit -s -m "fix(sync): reconcile actor environments with upstream #4002/#4020"
git -C $W commit -s -m "build(sync): adopt Bridge <BRIDGE_SHA12> / Gym fd5e84d6 pins and relock"
```

---


> **Re-anchor (2026-09-10, after Task 3):** the build branch moved to `28f599e9f` (CSCS
> layout reorganization: `--actors` selection from `infra/slurm/cscs/profiles/`, container
> TOMLs under `infra/slurm/cscs/environments/`, receipt tool and tests relocated). Per the
> Global Constraints, the sync branch is re-created from `28f599e9f`, upstream is re-merged
> with the recorded rerere resolutions, group A is redone against the new manifest CLI, and
> the post-merge fixes are replayed. Build/qualification paths below refer to the new layout.

### Task 4: Carry ledger, focused tests, static checks, TQ restored-schema regression

**Files:**
- Create: `docs/superpowers/specs/2026-09-10-upstream-sync-ledger.md` (in the worktree; the PR's carry/drop ledger)
- Create: `tests/unit/data_plane/test_transfer_queue_restored_schema.py`
- Modify (if fix chosen): `nemo_rl/data_plane/adapters/transfer_queue.py` (`register_partition` warmup dtype)

**Interfaces:**
- Consumes: merged tree from Task 3.
- Produces: ledger with four sections (retained / upstream-covered / replaced / deferred), each row `behavior | files | anchor symbol or test | SHA`.

- [x] **Step 1: Build the ledger skeleton from file ownership**

```bash
python3 - <<'PY'
import json,subprocess
cfg=json.load(open('session/20260905_114410/source_cleanup/carry/nemo-config.json'))
overlap=set(open('/tmp/claude-30214/-capstor-store-cscs-swissai-infra01-users-xyixuan-nemo-rl-v0-7-0/4637b44d-a056-4939-8ea8-942c6f8f6105/scratchpad/overlap.txt').read().split())
for topic in cfg['topics']:
    files=topic['paths'] if 'paths' in topic else topic['files']
    touched=[f for f in files if f in overlap]
    print(f"## {topic.get('title',topic.get('name'))}: {len(files)} files, {len(touched)} touched by upstream")
    for f in touched: print("  -",f)
PY
```
(Adjust key names to the JSON's actual schema; print them first with `python3 -c "import json;print(list(json.load(open('...'))['topics'][0]))"`.) Fork-only files are "retained verbatim"; overlap files each get a one-line resolution note.

- [x] **Step 2: Add the TQ restored-integer-schema regression (red first)**

```python
# tests/unit/data_plane/test_transfer_queue_restored_schema.py
import torch
from nemo_rl.data_plane.adapters import transfer_queue as tq


def test_register_partition_warms_existing_integer_fields_with_matching_dtype(tmp_path):
    adapter = tq.make_test_adapter(tmp_path)          # use the existing test factory in tests/unit/data_plane
    adapter.install_schema({"input_ids": torch.int64, "reward": torch.float32})
    adapter.save_state(tmp_path / "state")
    restored = tq.make_test_adapter(tmp_path)
    restored.load_state(tmp_path / "state")
    restored.register_partition("p0")
    assert restored.field_dtype("input_ids") == torch.int64
    assert not restored.errors(), restored.errors()
```
Replace the factory/API names with the ones the existing `tests/unit/data_plane` suite uses (read that suite first; do not invent helpers). Run: `pytest tests/unit/data_plane/test_transfer_queue_restored_schema.py -q`. Expected: FAIL with `existing=torch.int64, incoming=torch.float32`.

- [x] **Step 3: Fix or report**

Fix: in `register_partition`, warm fields using the already-installed schema dtype instead of a float32 placeholder. If the fix is not local to the adapter (e.g., TQ upstream package behavior), leave the test marked `xfail(strict=True)` with the reason and write the finding in the ledger's deferred section. Run the test again: PASS (or strict xfail).

- [ ] **Step 4: Focused CPU tests and static checks in the container**

Template (`session/20260910_152226/unit_cpu.sbatch`, submitted with `env -i ... sbatch`): `srun --environment=<qualified TOML>` then inside: `export HF_HOME=/iopsstor/scratch/cscs/xyixuan/.cache/huggingface; cd tests && bash run_unit.sh unit/<targets>`. Targets: the verify lists of Task 3 steps 2–5 plus `tests/unit/test_config_validation.py`, `tests/unit/tools/test_image_*.py`, `tests/unit/distributed`. Also `ruff check` + `ruff format --check` on changed Python and the pyrefly job pattern from the pp-foundation ledger. Note: the qualified image's fingerprint will not match the new tree; run tests via the worktree `PYTHONPATH` as the certification probes do, never via `NRL_IGNORE_VERSION_MISMATCH`.

- [ ] **Step 5: Review**

Run `/code-review` (high) on the branch diff vs `c00cb93cf`; fix findings; commit `test(sync): ...` / `fix(sync): ...` as needed.

---

### Task 5: Publish fork pins and the sync branch

- [ ] Push Core: `git -C $C push origin integrate/2026-09-10-upstream-sync` (store is non-shallow). Verify: `git ls-remote https://github.com/Alvorecer721/Megatron-LM.git | grep $CORE_SHA`.
- [ ] Push Bridge: `git -C $B push origin integrate/2026-09-10-upstream-sync` (after unshallow). Verify with ls-remote for `$BRIDGE_SHA`.
- [ ] Push NeMo-RL: `git -C $W push -u origin sync/2026-09-10-upstream-c49d53e`. Verify fork CI checkout does not fail with "not our ref".

---

### Task 6: Build and gate the synchronized image

- [ ] Fresh allocation 1: `HERMETIC_CACHE_TAG=rebuild NVTE_WITH_NCCL_EP=0` via `infra/slurm/cscs/build_nemo_rl_image.slurm`; confirm printed fingerprint and pyproject/lock digests equal the committed tree. Preserve successful dependency layers; if the generated cache identity proves native layers reusable, say so with the identity, otherwise do not claim reuse.
- [ ] Fresh allocation 2: builder publishes the OCI image + `RELEASE_RECEIPT`; then `assemble_nemo_rl_image.slurm` from that receipt (retryable independently).
- [ ] `qualify_nemo_rl_image.sh`: all selected native worker environments on GH200, vLLM native CUDA kernels (RMSNorm parity), fingerprint/inventory gates.
- [ ] Distributed GPU gates on 1 node: refit (nccl_reshard, TP2xPP2 8B smoke as in memory refit-local-views-line), finite logprob/loss checks, Bridge Apertus parity tests deferred from Task 2.
- [ ] Record image path, digest, job IDs and timings in the ledger.

---

### Task 7: Bounded 70B initial + resume qualification

- [ ] Recipe: GSM8K, thinking off, 2k context, 12 trainer nodes TP2/PP4 + 4 rollout nodes TP4/PP1, 48 prompts x 16 responses, GBS 768, save optimizer state; 2 updates, then resume for 2 more. Reuse the launcher and config recorded in `.worktrees/image-assembly-split/.tmp/image-assembly-split/full-validation/` with only the image path changed.
- [ ] Checks beyond exit 0: Adam step counters 2 -> 4, scheduler samples continuity, replay/sample progression, 48 non-empty DCP shards per save, all logged scalars finite, and an explicit tensor comparison of the resumed optimizer state against the saved step-2 shards (load both with the dist-checkpointing reader on CPU; assert dtype, shape and allclose for exp_avg/exp_avg_sq/master params of a sampled subset of parameters). Confirm the TQ `int64 vs float32` warning is absent (or explicitly reported if Task 4 deferred it).

---

### Task 8: GLM-5.1 10-step probe

- [ ] Identify the latest main-line 10-step probe: search `session/*/` ledgers and `infra/slurm/cscs/autoresearch/` for the most recent GLM-5.1 main-line run that records image + config + launcher (distinguish from the Aug 24 R3 probe in `.tmp/glm51-r3-10step`). Write the resolved launcher, config and prior job ID into the session timeline BEFORE submitting.
- [ ] Run it on the new image with rollout PP1 and the CSCS limitations unchanged; compare step metrics (gen-KL, loss, refit time) against the recorded run. Do not claim speedups.

---

### Task 9: Review and PR

- [ ] Final `/code-review` on the full branch; resolve findings.
- [ ] Open the PR against `Alvorecer721/Nemo-RL:main` (or rebase onto the merged build PR first, see Global Constraints). Body: source SHAs (three repos + Gym), pins and lock delta, image digest, build/assembly timings, tests, job IDs, the carry/drop ledger link, remaining limitations (rollout PP1 only, 12k study not run, deferred items). Hyperlink pushed SHAs only.
- [ ] Do not merge. Update `session/20260910_152226/handoff.md`.
