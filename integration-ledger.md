# PP foundation integration ledger

User authorized execution with "go" on 2026-09-05. Useful format changes are accepted. Source integration, dependency reconciliation, image build and bounded 70B validation are authorized; rollout PP implementation is the next phase.

| Task/interface | Review | Decision |
|---|---|---|
| Task1 internally | Exact upstream baseline and import fix compatible in scope; optimizer fix must remain | Merge in standalone MCore repository |
| Task2 internally | New upstream Bridge includes approved correctness fixes and TE2.18 dependency update | Preserve Apertus fork while merging |
| Task3 internally | Existing successful runtime fixes are ahead of bumped image branch | Start from b7a6a2a61 |
| Task4 internally | Validation must not imply rollout PP is implemented | Use existing PP1 rollout path first |
| Task1 -> Task2 | MCore SHA consumed by Bridge .main.commit and gitlink | Both must agree |
| Task2 -> Task3 | Bridge SHA and dependency metadata consumed by NeMo-RL lock | Root integration waits for final Bridge commit |
| Task3 -> Task4 | Clean source and recursive pins determine image fingerprint | Rebuild dependencies; no fingerprint bypass |

Ruling: use repository-local .git/info/exclude for the worktree directory rather than modifying the user's dirty root .gitignore. This preserves the existing root source while avoiding accidental worktree staging.

Ruling: execute independent source integration locally while an isolated submodule implementer works. Only one implementation subagent at a time; all edits remain in disjoint repositories.

Status: source integration is complete. NeMo-RL merge440ae6a22 integrates upstream5368eff5f. Bridge1d9a69fde integrates upstream75d7b9eb and pins MCore1e5025f85 in both .main.commit and its gitlink. MCore includes upstreamb1fe7599, PR7088, the preserved optimizer-load correction and a reviewed dtype correction. TE is27486e03 /2.18.0+27486e03; vLLM remains0.26.0. Combined image and GPU qualification remain outstanding.

User clarification: finish the agreed commit integration and verification, then stop. PR to main is a subsequent step; no PP feature work or publication in this task.

Validation: job3303570 passed90 tests before an older policy padding test failed. Commit f554b5205 made stamping idempotent without updating that test; the test now exercises _isolated_meta and checks the original metadata is preserved. Job3303576 passed104 tests before the value-policy fixture lacked required make_sequence_length_divisible_by; the fixture is corrected. Final job3303593 passed109 tests, including selected-token logprob parity. All67 changed Python files passed Ruff check/format.

Ruling: preserve production padding behavior and correct the two incomplete test contracts, based on the pre-merge source/history and the existing policy/value dispatch API. No production padding changes are required.

Ruling: the controller resolves Bridge's small source merge while the isolated MCore implementer finishes; independent review still covers both outputs before adoption.

Task1: source complete. Independent review found inherited precision-aware state initialization applying BF16 remainders to FP32/FP16. The correction matches TE's effective optimizer flag and parameter dtype. Regression failed3/passed1 before correction, then final optimizer/MLA suite passed7. Reviewer accepted the correction. Original optimizer state reuse remains covered.

Task2: source complete. Full Bridge pre-commit passed after final repin. Two signature-helper tests and25 checkpoint/RNG/Apertus tests passed in the existing worker environment. Four MoE-logging cases need CUDA and failed in fixture setup here; retain them for GPU qualification. No new integration regressions found by independent review.

Task3: dependency resolution complete with image-pinned uv0.11.28. The shell default uv0.9.5 cannot interpret current project settings. Relocking resolves562 packages: TE replacement and nvdlfw-inspect0.2.2 are the only changed package identities; Bridge/MCore extra metadata also updates. TE static runtime requirements were compared to the exact source by independent review. Root/source/submodule checks precede image submission.

Ruling: fix the inherited optimizer dtype error now because it is in the state-loading correction this baseline must preserve; BF16/FP32/FP16 regression protects the change. No optimizer algorithm or learning-rate setting changes.

Ruling: local integration commits use DCO sign-off without GPG signatures because the configured signing identity has no secret key. User signing configuration is untouched; publication remains deferred.

Ruling: keep unrelated package identities fixed during integration; update TE's static runtime metadata to match its pinned PyTorch source instead of retaining the incomplete old declaration. The newly required nvdlfw-inspect package is part of TE's declared dependency set.

Ruling: image qualification requires two fresh allocations because the 334 GiB allocation-local Podman graph cannot safely contain both the hermetic dependency rebuild and the release-layer commit. The first allocation will use HERMETIC_CACHE_TAG=rebuild to build and publish only the hermetic cache, print its fingerprint and pyproject/lock digests, and exit without a release image or SquashFS. The printed values must match the committed target pins before a second fresh allocation uses that cache to assemble the release image and SquashFS and run final version checks. Neither phase has been run or completed; GPU checks and bounded 70B training/refit with checkpoint save/resume remain outstanding, and rollout validation stays at PP1 before any later PP implementation.

Review artifacts and detailed command logs are under /tmp/nrl-pp-integration; durable validation records will accompany the image/run results. No main branch, historical checkpoint, or existing user allocation was changed.
