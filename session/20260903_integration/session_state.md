# Session State

- Session: 20260903_integration
- Repo: /capstor/store/cscs/swissai/infra01/users/xyixuan/nemo-rl-worktrees/glm51-on-upstream-a952
- Branch: integration/glm51-on-upstream-a952
- Started: 2026-09-03 03:00 CEST
- Updated: 2026-09-03 (see handoff.md status log)

## Goal
One certified NeMo-RL line for the GLM-5.1 136-node GRPO campaign: upstream main plus the hermetic CSCS image plus the campaign fixes, reviewed, simplified, and reproducible.

## Current Subtask
Draft PR of this branch into the fork's `main`; 136-node rerun pending the user's go.

## Loaded Skills
- `nemo-rl-session-memory` - ledger discipline for this record.
- `simplify`, `review-pr`, `contributing`, `build-and-dependency`, `config-conventions`, `nemo-rl-docs`, `nemo-rl-auto-research` - review, PR and dependency conventions applied during the session.

## Current Status
Certified HEAD `33c1e8f5f` (probe 3278193 green: fingerprint OK, 33/3/117/9/7/31/750 tests, 3240688 revalidation OK). Lint gates green repo-wide. Campaign worktree retired, submodules initialised, launcher dry run passes.

## Plan
- [x] Rebase campaign onto the sync line, restore merge losses, certify
- [x] Deep code review, simplify, re-certify
- [x] Merge upstream main b7ce030b2, re-certify
- [x] Retire the campaign worktree, init submodules
- [ ] Draft PR into fork main
- [ ] 136-node MTP3+fused rerun on the new image

## Assumptions
- The certified image's fingerprint (pyproject, uv.lock, submodule SHAs) is unchanged by the b7ce030 merge; verified by probe 3278193.

## Blockers
- None known. The 136-node launch is a resource decision for the user.
