# Handoff

Read `session_state.md` first.

Job `3203601` was intentionally cancelled before model loading because a broad Git-cleanliness preflight blocked on transient Capstor metadata access after Ray had connected all `544/544` worker units. The narrowed runtime-input preflight and recovered submit-wrapper fixes are ready. Commit them, run submission/config checks, and submit `infra/slurm/cscs/autoresearch/submit_glm51_sc_scale.sh`. Do not mutate this checkout after submission until the job reaches a terminal state: the runner verifies the source SHA and imports the live checkout through `PYTHONPATH`.

Acceptance requires three completed training steps, nonzero loss and gradient plus signed advantage range on every step, generation KL below 0.001, green Router Replay trace validation, and scheduler exit 0:0. Report total-step time, exposed-generation time, policy-training time, weight-sync time, and valid tokens/s/GPU against completed job 3186645.

The experiment is a throughput comparison, not a substitute for the completed 10-step correctness evidence.
