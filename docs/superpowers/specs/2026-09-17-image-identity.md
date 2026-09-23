# Image identity: fingerprint what the image bakes (2026-09-17)

Design for replacing the raw-input container fingerprint with an identity
derived from the resolved package set per actor environment plus the pins of
submodules whose artifacts the image actually bakes. The goal is that a source
sync with no dependency change produces an identical identity, so the qualified
image passes every gate unchanged, and that a dependency change is classified
mechanically as reusable, overlay, or rebuild before anyone decides by hand.

## Problem

The fingerprint in `tools/generate_fingerprint.py` is the MD5 of
`pyproject.toml`, the MD5 of `uv.lock`, the MD5 of
`nemo_rl/distributed/actor_environments.py`, and the SHA of every top-level git
submodule. Four gates consume it:

| Gate | Where | Behavior on mismatch |
|---|---|---|
| Import gate | `nemo_rl/__init__.py::_check_container_fingerprint` | `RuntimeError` at `import nemo_rl` whenever `NRL_CONTAINER=1`, unless `NRL_IGNORE_VERSION_MISMATCH` |
| Worker readiness | `nemo_rl/utils/venvs.py::_dependency_fingerprint` (digests the raw bytes of `uv.lock` and `pyproject.toml` plus the worker command) | every prebuilt venv is re-synced with `uv sync --offline --frozen`; a wheel missing from the image cache fails the worker |
| Image qualification | `tools/check_image_workers.py` | build fails |
| Overlay restamp | `tools/restamp_overlay_fingerprint.py` | overlay refused if any submodule pin differs |
| Hermetic cache key | `tools/image_build_manifest.py` (`dependency_fingerprint` and `recursive_submodules`) | the multi-hour hermetic stage is rebuilt |

All five answer "did the source inputs change" when the question is "did what
the image bakes change". Evidence from the pending sync of upstream
`5d49fbf4e..d4b446886` (24 commits):

| Input | Change | Installed effect |
|---|---|---|
| `uv.lock` | gitpython 3.1.59 to 3.1.62 (Gym raised its floor) | one universal wheel |
| `actor_environments.py` | adds `RolloutReassemblerActor` with the `nemo_gym` extra | one new worker venv, all wheels already cached |
| `3rdparty/Gym-workspace/Gym` | fd5e84d6 to 267305e2 | none; Gym is an editable path dependency and runs from the checkout |

Under the current tooling this costs a full hermetic rebuild: the import gate
raises, the worker markers invalidate, the overlay refuses the pin change, and
the hermetic key includes the pin. The honest cost is one wheel and one venv.

A sync up to upstream `88ee6c1a` (the commit PR #42 cherry-picked) changes none
of the three inputs and is free today; it is listed here only to show the
boundary is arbitrary.

## Contract

The image identity is exactly:

1. **Per actor environment, the resolved installed set.** For every actor in
   `ACTOR_ENVIRONMENTS` whose extras are not `None`, and once for the base
   environment with no extras, the SHA-256 of
   `uv export --frozen --no-dev --no-header --no-emit-project --format requirements-txt [--extra ...]`
   with the `# via` comment lines removed. This reads only `uv.lock`, works
   offline, is deterministic (verified on the current lock), and differs between
   extras sets (verified: `mcore` vs `vllm,nemo_gym`). Lock reformatting, marker
   rewrites, and dependency-edge changes that leave the resolution untouched do
   not change it. A version, hash, or editable-path change does.
2. **Baked submodule pins.** The SHA of every top-level git submodule that
   `uv.lock` does not declare as an editable source. Today that is
   `3rdparty/kernels` only; Megatron-Bridge (with its nested Megatron-LM),
   Automodel, and Gym are `source = { editable = ... }` and drop out. The rule is
   mechanical and fails closed: a submodule the lock does not mention counts as
   baked.
3. **The uv version** that produced the export, read from `ARG UV_VERSION` in
   `docker/Dockerfile`. Two identities are comparable only when this matches;
   otherwise comparison is an error naming both versions.

Everything else the image records (the hermetic recipe, build scripts, base
image, platform, profile, TensorRT-LLM workspace files) stays in
`tools/image_build_manifest.py::dependency_files` and is unchanged by this
design. The identity is not a build cache key; it is the answer to "can this
source tree run on this image".

Identity JSON, schema 2:

```json
{
  "schema": 2,
  "uv": "0.11.28",
  "base": "<sha256>",
  "actors": {
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker": "<sha256>",
    "...": "..."
  },
  "baked_submodules": {
    "3rdparty/kernels": "90b92d6b..."
  }
}
```

## Where the identity is computed and consumed

- **`tools/image_identity.py`** (new, stdlib only, fork-owned): computes the
  identity, classifies deltas, and caches. Both `tools/generate_fingerprint.py`
  and `nemo_rl/utils/venvs.py` load it by path so `nemo_rl/__init__.py` never
  imports it as a package (that would recurse into the import gate).
- **`tools/generate_fingerprint.py`** becomes a shim that prints
  `image_identity.compute(repo_root)`. The filename, the stdout JSON contract,
  and the `.nemo_rl_source_fingerprint.json` embedded copy for release trees
  without `.git` are kept, because the Dockerfile, the CI test template, the
  slurm launchers, and `nemo_rl/__init__.py` all reference them.
- **Host and container split.** The only git-derived part of the identity is
  the pin map. The slurm build and overlay scripts pass the pin map
  (`git submodule status`, stdlib) as `NEMO_RL_SUBMODULE_PINS_B64`; the
  Dockerfile computes the full identity in-container with its pinned uv. The
  host never needs uv. `NEMO_RL_BUILD_FINGERPRINT_B64` is retired.
- **`nemo_rl/utils/venvs.py::_dependency_fingerprint`** digests the normalized
  worker command plus the export digest for the extras that command names, so a
  source-only sync leaves every readiness marker valid and no offline re-sync
  runs.
- **`tools/check_image_workers.py`** validates schema 2 and requires the probed
  actor to have an entry. The full-dict comparison is unchanged.
- **`tools/restamp_overlay_fingerprint.py`** compares `baked_submodules` only,
  then lets the Dockerfile recompute the identity in-container.
- **`tools/image_build_manifest.py`** takes `dependency_fingerprint` (schema 2)
  and drops the separate `recursive_submodules` input. Reusing a hermetic layer
  across an editable-pin bump is safe because the release stage copies the
  source tree again and `prefetch_venvs --prebuilt` re-syncs every actor
  frozen and offline, which re-points editables and rebuilds Megatron-LM's
  in-place pybind11 extension from the new tree, exactly as it does today.
- **`nemo_rl/__init__.py`** is unchanged in logic (it compares two dicts key by
  key). Its generic `except Exception` path must not swallow an identity
  computation failure inside a container; `image_identity` raises
  `RuntimeError` for a missing or wrong-version uv, and the check re-raises
  `RuntimeError`.
- **Import-time cost.** Every `import nemo_rl` in every Ray worker runs the
  check. Computing the identity is one `uv export` per actor (about 30 calls).
  `image_identity` caches the result under `NEMO_RL_VENV_DIR` keyed by the
  SHA-256 of the raw inputs (`uv.lock`, `pyproject.toml`,
  `actor_environments.py`, the pin map, the uv version). Raw digests are a cache
  key, never the identity.

## Verdict tool

`tools/image_compat.py <image-fingerprint.json> [--repo-root .]` compares an
image's stamped identity with the source tree and prints one of:

| Verdict | Exit | Condition |
|---|---|---|
| `reusable` | 0 | identical identities, or only actors removed |
| `overlay` | 10 | every changed or added lock entry resolves to universal wheels only (`*-none-any.whl` in the lock's `wheels` list), and no baked pin changed |
| `rebuild` | 20 | any platform wheel, sdist-only entry, or baked pin change |
| error | 1 | uv version mismatch, unreadable inputs, schema other than 2 |

The report lists per actor the added, removed, and changed packages with old
and new versions, and names the reason for the verdict. Universal-wheel
classification reads `uv.lock` with `tomllib`; nothing is downloaded.

Each launch environment TOML under `infra/slurm/cscs/environments/` gets a
sibling `<name>.fingerprint.json`, the verbatim stamp of the image it selects.
The fork CI gate runs the verdict against the environment the sync line uses
(`nemo_rl_vllm029.toml`) on every PR and writes the verdict to the job
summary. The assemble script verifies the committed copy equals the image's
stamp so the two cannot drift.

## Assumptions

- Editable path dependencies run from the mounted checkout after the frozen
  worker re-sync, so their source SHA does not affect what the image bakes.
  Megatron-LM's pybind11 dataset helper is built in place from that checkout
  during the same re-sync, offline, with build inputs already in the image's uv
  cache; this is existing behavior, not new.
- Worker syncs are `--offline --frozen`, so a wheel missing from the image's
  uv cache is the only launch-time failure a lock change can cause. `--frozen`
  never validates the lock against `pyproject.toml`, so a lock text change with
  an identical resolution installs identically.
- A universal wheel can always be added by the overlay Dockerfile; platform
  wheels and source builds cannot be assumed to.
- `uv export --frozen` output is stable for a fixed uv version. Identities are
  compared only under an identical uv version; the pinned version is read from
  `docker/Dockerfile` so there is one source of truth.
- Gym's per-server venvs prefetched under `/opt/gym_venvs` are built from
  Gym's own lock and are outside this identity. A Gym bump can still fail those
  paths offline. They are out of scope here and tracked as an open item.
- The `apertus` profile builds six actor venvs; the identity covers every
  actor in the table regardless of profile, as the current fingerprint does.
  The verdict tool reports per actor, so a change confined to an unbuilt actor
  is visible and classified `reusable` for that profile.

## No backward compatibility

- Schema 1 fingerprints are not recognized. `image_compat.py` and
  `check_image_workers.py` reject them with a message naming the schema.
- The qualified images in use are stamped with schema 1. Their schema 2
  identity is computed once from each image's source commit and committed as
  the environment's `.fingerprint.json`; the launcher mounts that file over
  `/opt/nemo_rl_container_fingerprint`. Images built after this change carry
  schema 2 natively and need no mount.
- Existing worker readiness markers hold the old digest. The first launch
  after this change re-syncs every prebuilt venv once, offline, from the
  image's cache. That is the current behavior for any lock change and succeeds
  because the identity is unchanged.
- `NEMO_RL_BUILD_FINGERPRINT_B64` is removed from the Dockerfiles and slurm
  scripts. `NEMO_RL_SUBMODULE_PINS_B64` replaces it.
- The overlay restamp rule changes meaning from "no submodule pin may change"
  to "no baked submodule pin may change".
- The hermetic cache key no longer includes editable submodule pins, so
  manifests written before this change do not match and their layers are not
  reused. One hermetic rebuild.
- `test_fingerprint_covers_the_actor_table` and the fixtures in
  `tests/unit/tools/test_image_workers.py`, `test_overlay_fingerprint.py`,
  `test_image_build_manifest.py`, and `tests/unit/test_version_check.py` are
  rewritten for schema 2 rather than extended.

## Upstream divergence

`tools/generate_fingerprint.py` and the `_check_container_fingerprint` contract
are upstream's. The shim keeps upstream's entry point and output channel so the
divergence is one small file; `tools/image_identity.py`, `image_compat.py`, the
restamp tool, the manifest tool, and the readiness marker are already fork
code. Future upstream edits to `generate_fingerprint.py` conflict in the shim
only.

## Open items

- Gym server venvs: derive an identity from Gym's lock and the prefetch config
  list, or fail closed by treating a Gym pin change as `rebuild` whenever
  `NEMO_GYM_PREFETCH_CONFIGS` is non-empty for the image.
- The verdict tool classifies universal wheels only. A platform wheel already
  present in the image's uv cache is overlayable in practice; detecting that
  needs the cache listing, which the tool does not read.
