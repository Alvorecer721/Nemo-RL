# Image environments: reuse what the image already installed (2026-09-18)

Three changes with one purpose: a source change that installs nothing new must
not cost a worker re-sync at launch or a base image build. The runtime-sync path
stays the default and keeps working; using the image's venvs as built is opt-in.

This is the first slice of the image identity design
(`2026-09-17-image-identity.md`). It covers the worker readiness gate and the
overlay gate. The import gate, the hermetic cache key and the verdict tool stay
as that design describes them and are not part of this change.

## Contracts

1. **A worker venv is current when what it installs is current.**
   `nemo_rl/utils/venvs.py` records, per venv, the normalized worker command and
   a digest of the resolved environment: `uv export --frozen` for the command's
   extras with comment lines removed, plus the `[tool.uv]` table of
   `pyproject.toml`. Default dependency groups stay in the export because
   workers install them. The marker is JSON with the keys `command`, `uv` and
   `environment`; the uv version is recorded because the export text differs
   between uv releases (same lock, `--extra mcore`: one digest under 0.11.28,
   another under 0.11.18 and 0.9.5). Raw `uv.lock` and `pyproject.toml` bytes
   are no longer part of
   it, so a lock reformat, a change confined to another extra, or an edit
   outside `[tool.uv]` leaves every venv current.
2. **`NEMO_RL_IMAGE_VENVS=1`: the image's worker venvs are used exactly as
   built.** `create_local_venv_on_each_node` returns
   `$NEMO_RL_VENV_DIR/<actor>/bin/python` without scheduling a build and without
   running uv. The venv's recorded `uv` and `environment` must equal the
   checkout's; otherwise the launch fails and names the uv or the image as the
   thing to fix. The recorded `command` is not compared: it names the source
   tree the image was built from. Source reaches every actor through
   `PYTHONPATH`: `init_ray` calls `add_checkout_to_pythonpath(os.environ)` once,
   before any runtime_env is captured, which puts the checkout first and then
   every other editable project the lock names (`src/` when the project has
   one). The checkout leads because sibling projects also carry top-level
   `tools`, `tests` and `examples` packages. `sys.path` is
   searched before a venv's editable finders, so the checkout wins over the
   image's baked copy; verified with a prebuilt venv for `nemo_rl`,
   `megatron.bridge` and `megatron.core`.
3. **An overlay may move an editable submodule, never a baked one.**
   `tools/restamp_overlay_fingerprint.py` derives "editable" from `uv.lock`: a
   submodule is editable when it holds an `editable` source. The overlay build
   ships the editable submodules whose pin moved since the base release and
   removes the base's copy first. The list is computed once on the host and
   handed to the restamp (`--shipped`), which refuses any moved pin that is not
   both shipped and editable, so what was copied and what is excused cannot
   disagree. The base is addressed by the release receipt next to the SquashFS
   (`image_ref` by digest, `vllm_version`, base pins), read only through
   `image_release_receipt.py`, not by a tag derived from the file name.

`infra/slurm/cscs/autoresearch/launch_gsm8k_baked.py` carried a private copy of
contract 2 (five hard-coded actors, a rewritten command string, a registry
update). It now calls the shared functions and the bench launcher exports
`NEMO_RL_IMAGE_VENVS=1`. Its per-node preflight also resolves the checkout's
import roots, so an uninitialized submodule fails before Ray starts.

## Assumptions

- `uv export --frozen` is stable for a fixed uv version. Image builds and
  launches both run the image's pinned uv (`$UV`).
- Editable projects keep their import root at the project directory or its
  `src/` directory. True for nemo-rl, Megatron-Bridge, Megatron-LM, Automodel,
  Gym and the research template today.
- With image venvs, Megatron-LM's in-place pybind11 dataset helper is whatever
  the checkout holds. RL training does not import it; Megatron builds it on
  demand where it is needed.
- Gym server venvs are not baked into our images (`NEMO_GYM_PREFETCH_CONFIGS`
  is unset), so a Gym pin change has no baked counterpart.
- An overlay extends a release that has a receipt. An overlay's output gets no
  receipt, so the next overlay starts from the same release again rather than
  from the previous overlay. Chaining needs a receipt for overlays and is the
  next step towards "always build on the last image we ran".
- With `NEMO_RL_IMAGE_VENVS=1` the checkout's submodules must be initialized,
  which the bench did not need while it imported them from the image.

## No backward compatibility

- Markers written before this change are not recognized. Under the default
  path the first launch re-syncs each venv once, offline. With
  `NEMO_RL_IMAGE_VENVS=1` an older image is rejected; the bench therefore needs
  an image built or overlaid from this change onwards.
- `restamp_overlay_fingerprint.py` takes subcommands (`restamp`,
  `shipped-submodules`) and requires `--lock`; `image_release_receipt.py fields`
  prints a third line, the vLLM version, and gains a `fingerprint` action.
- `build_nemo_rl_overlay_image.slurm` requires the base's release receipt and no
  longer accepts a base known only by its file name.

## Applied to the 0.29 line

Against the release `nemo-rl-apertus-vllm-0.29.0-c6b52f981f54-c8d570483215`, the
sync branch differs by gitpython 3.1.59 to 3.1.62 (pure-Python wheel) and the Gym
pin (editable). `shipped-submodules` lists `3rdparty/Gym-workspace/Gym` only and
the restamp accepts it, so the image for that branch is an overlay on the 0.29
release, not a base build.
