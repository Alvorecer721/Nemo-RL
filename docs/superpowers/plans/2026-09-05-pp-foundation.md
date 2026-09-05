# PP Foundation Integration Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for isolated submodule tasks; the controller integrates NeMo-RL and owns image validation. Steps use checkbox syntax.

**Goal:** Integrate the approved upstream improvements into a reproducible NeMo-RL/Bridge/MCore/TE baseline suitable for subsequent vLLM rollout PP development.

**Architecture:** Preserve the successful Apertus/GLM fork lineage and merge fixed upstream revisions into isolated repositories. Keep vLLM 0.26.0, update TE to its exact 2.18 tag, build one combined image, and validate the existing 70B training/refit path before changing rollout topology.

**Tech Stack:** NeMo-RL, Megatron-Bridge, Megatron-Core, Transformer Engine, vLLM, uv, Podman, Slurm/Pyxis on GH200.

**Spec:** The user's approved integration list in this conversation (2026-09-05), including acceptance of useful checkpoint format changes. This task prepares the foundation; rollout PP implementation follows validation.

## Global Constraints

- Preserve original worktrees, historical checkpoints and images. Work only in the new pp-foundation repositories and their task records.
- NeMo-RL starts at b7a6a2a61d8d9aa76becd156d123c36a58159ceb and integrates upstream 5368eff5f6bc7287f5a3d7fdc762530396610b1c.
- Bridge starts at 3ce0cddefc84ba8f18515e410bb913978d27d63c and integrates upstream 75d7b9eb89b05b7edbef7daf931045c38c6ca9df.
- MCore starts at 7851ef7cb9fbd261d592b58e3ede406aa151e60b and integrates upstream b1fe7599e18a213292600177ad7ff290a1127160 plus applicable PR7088 head 68db7a4071afdbefbc7e854a978b5c04c9d11d62.
- Preserve Apertus XIELU/model conversion, optimizer-load fix, local refit views, direct-bulk shape guards, wire-safe metadata and streaming metric serialization.
- vLLM remains 0.26.0; TE becomes 27486e03cfc1fa41f6932dcecdc47c71c47eac3e (2.18.0); retain NVTE_WITH_NCCL_EP=0 on the existing NCCL2.29 base.
- Use uv. Do not disable fingerprint validation or set NRL_IGNORE_VERSION_MISMATCH. Do not publish or push to shared branches.
- Existing 70B baseline configuration remains the comparison reference. Initial validation uses rollout PP1; do not claim rollout PP2 works before its implementation.

## Task 1: Integrate MCore

**Files:** MCore integration clone, megatron/core/optimizer/distrib_optimizer.py, megatron/core/transformer/multi_latent_attention.py, and the associated upstream tests.

**Interface:** Produces one local MCore commit containing the approved upstream baseline, optimizer fix, and optional-TE import guard. Bridge consumes its SHA.

- [ ] In the independent pp-foundation-mcore clone, create integrate/2026-09-05-pp-foundation from 7851ef7cb9fbd261d592b58e3ede406aa151e60b.
- [ ] Read AGENTS.md and relevant build/testing skills. Fetch the exact upstream baseline and PR7088 into this clone.
- [ ] Merge b1fe7599e18a213292600177ad7ff290a1127160; reconcile rather than discard the optimizer-load correction. Apply PR7088 if absent.
- [ ] Verify ancestry for PRs6912,6952,6998 and source equivalence/preservation of the optimizer correction. Run focused import/checkpoint regressions where the environment supports them; record unavailable CUDA checks separately.
- [ ] Run required formatting/static checks and create signed-off, signed local commits. Write the report with exact SHAs, conflict decisions, commands/results, and remaining tests.

## Task 2: Integrate Bridge

**Files:** Bridge integration clone, .main.commit, 3rdparty/Megatron-LM gitlink, pyproject.toml, uv.lock, and conflict resolutions.

**Interface:** Consumes Task1 MCore SHA; produces a clean Bridge integration commit with matching .main.commit and gitlink. Root NeMo-RL consumes the Bridge SHA.

- [ ] Create integrate/2026-09-05-pp-foundation from 3ce0cddefc84ba8f18515e410bb913978d27d63c in the independent pp-foundation-bridge clone.
- [ ] Merge exact upstream 75d7b9eb89b05b7edbef7daf931045c38c6ca9df while preserving Apertus support and fork submodule location.
- [ ] Set .main.commit and MCore gitlink to Task1 output, retaining upstream .dev.commit. Include source versions of PRs5925,5927,5953 and preserve existing EP RNG handling.
- [ ] Relock through uv in a compatible environment. Run required pre-commit and focused provider/checkpoint/Apertus conversion checks; report unavailable GPU checks.
- [ ] Commit and report exact source pins, conflict decisions and validation evidence.

## Task 3: Integrate NeMo-RL and image inputs

**Files:** Root pyproject.toml, uv.lock, Bridge gitlink, infra/slurm/cscs/build_nemo_rl_image.slurm, affected upstream source/tests, and integration records.

**Interface:** Consumes Task2 Bridge checkout/SHA and produces a clean, fingerprinted release source suitable for the existing image builder.

- [ ] Merge upstream 5368eff5f6bc7287f5a3d7fdc762530396610b1c into the isolated b7a6a2a61 lineage, preserving runtime fixes and unified ALP configuration.
- [ ] Reconcile conflicts using both parent versions and upstream tests. Preserve selected-token logprob optimization, MTP split-training semantics, router replay exclusion, HTTP loop ownership and Transformers cache compatibility.
- [ ] Populate submodules at pinned revisions without mutating their original checkouts. Update Bridge gitlink and TE override/metadata to 27486e03cfc1fa41f6932dcecdc47c71c47eac3e / 2.18.0+27486e03.
- [ ] Update the build-time TE assertion and regenerate root uv.lock. Review the dependency delta and fingerprint; keep existing installed backend pins unless resolution demands a justified change.
- [ ] Run relevant CPU/static checks and prepare focused GPU tests from existing refit, MTP, logprob and checkpoint suites. Record baseline versus newly introduced failures.

## Task 4: Build and validate

**Files:** Integration record, build logs, candidate image descriptor and bounded 70B validation launch configuration.

**Interface:** Produces an immutable image, validation results and a documented baseline for rollout PP implementation.

- [ ] Review integrated diffs and resolve findings; require clean committed source and recursive submodule state.
- [ ] In a fresh first allocation, submit the image builder with HERMETIC_CACHE_TAG=rebuild and NVTE_WITH_NCCL_EP=0. This builds and publishes only the hermetic cache, prints its fingerprint and pyproject/lock digests, and exits without producing a release image or SquashFS. Verify the printed pins exactly match the committed target values before proceeding.
- [ ] In a second fresh allocation, use the verified hermetic cache pin to assemble the release image and SquashFS, then verify baked imports, exact package versions, source fingerprint and output image integrity. The allocations are separate because the 334 GiB allocation-local Podman graph cannot safely hold both the dependency rebuild and release-layer commit.
- [ ] Monitor both build jobs; diagnose and fix concrete failures. Do not mark either phase complete until its allocation succeeds and its required outputs are verified.
- [ ] Run focused GPU regressions and a bounded 70B training/refit plus checkpoint-resume validation against the candidate image, with unchanged dataset/model identities.
- [ ] Report source pins, image path, jobs, completed checks and remaining limitations; do not report the foundation as validated while required checks remain outstanding.
