# Handoff

Read `session_state.md` first.

Job `3217346` proved the 136-node launch and model setup but failed before step 1 in the first SingleController logprob fetch. Router Replay uses `torch.int16` routes; `_broadcast_batched_data_dict` passed that tensor directly to NCCL, which does not support `Short`. The previously reviewed byte-wire, shared-helper, and topology-padding commits were missing from the scale branch. They are now applied, and exact-EDF job `3217574` passed the two-rank NCCL non-contiguous-int16 round trip. Rerun `infra/slurm/cscs/autoresearch/submit_glm51_sc_scale.sh` from the clean signed HEAD. Do not mutate this checkout after submission until the job reaches a terminal state: the runner verifies the source SHA and imports the live checkout through `PYTHONPATH`.

Acceptance requires ten completed training steps, at least eight steps with nonzero loss and gradient plus signed advantage range, generation KL below 0.001, green Router Replay trace validation, and scheduler exit 0:0. Report total-step time, exposed-generation time, policy-training time, weight-sync time, and valid tokens/s/GPU against completed job 3186645.

The experiment is both the matched rollout-capacity comparison and the full 10-step exact-source correctness gate. Treat job `3217571` only as evidence that container-internal submissions must unset inherited Slurm/Pyxis variables before `sbatch`; it never exercised Python or NCCL.
