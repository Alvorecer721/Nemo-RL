# Session state

- Goal: keep the synchronized Apertus NeMo-RL fork production-ready and reduce GLM-5.1 trainer starvation without weakening Router Replay or refit correctness.
- Active branch: `autoresearch/2026-08-29-glm51-mtp3`.
- Base: fork `main` at `d7ced9573cadf1a391ea55205c60a3013b4d8b27`, plus the exact file diff from open PR #27 (`1c7d48f207dcfca8a4377ebdc58a6d77c236da42`).
- Experiment: 72 Megatron nodes plus 64 vLLM nodes, ten SingleController/TransferQueue ready-first steps, TP2/PP18/ETP1/EP16 training and eight TP32/EP32 vLLM replicas.
- Reservation: `SD-69241-apertus-1-5-0`; the latest snapshot left roughly 473 of 600 nodes free before the canaries.
- Constraint: freeze the committed source while the Slurm job is queued or running.
- Remote cleanup: ten merged NeMo-RL branch heads that were zero commits ahead of `main` were deleted; every other remote branch was left untouched.
- First launch: job `3203601` was intentionally cancelled before model loading after the driver blocked in the broad Git-cleanliness preflight. Ray itself reached `544/544` worker units in 107 seconds.
- Infrastructure recovery: on 2026-08-28, 12 consecutive Capstor, Git, Iopsstor, and Slurm probes completed without error; Capstor/Git/Iops latency was about 12-16 ms and Slurm latency 29-38 ms.
- Infrastructure gate: exact-image 136-node Pyxis fan-out jobs `3215160` and `3215161` passed sequentially in 52 and 44 seconds, each with 136/136 ranks on 136 distinct nodes and no nonempty task errors.
- First A0 attempt: job `3217346` reached all 544 workers and model setup, then failed before step 1 because the SingleController TQ leader-broadcast path sent Router Replay `torch.int16` metadata directly through NCCL (`Short` is unsupported). The already-reviewed byte-transport and topology fixes had not been included on the scale branch.
- Focused fix evidence: the three signed commits are now present, and exact-EDF job `3217574` passed a two-rank NCCL round trip with non-contiguous `int16` routes. Job `3217571` never entered the container because inherited Pyxis variables duplicated the EDF; it is not runtime evidence.
- Completed baseline: job `3217663` finished ten 72-train + 64-rollout-node steps with generation KL `0.000366-0.000448`, Router Replay trace/CP identity green under the corrected validator, and all ten steps carrying learning signal.
- Topology decision: the BF16 checkpoint is 1,507,728,316,928 bytes, so TP16 would consume about 94.2 GB/GPU for weights alone. NCCL reshard also rejects generation PP greater than one. Retain TP32/PP1/EP32 and do not spend allocations on TP16/PP2, TP8/PP4, or TP4/PP8.
- Current subtask: run one matched MTP3 challenger against `3217663`. It keeps the 72+64 topology and all quality gates, loads GLM's native layer-78 MTP drafter once from disk, and records speculative acceptance plus the existing throughput metrics.
- Stop rule: this campaign stage has exactly one challenger. Stop immediately on missing MTP weights, missing speculative counters, Router Replay fallback/trace mismatch, KL at or above `0.001`, or terminal scheduler failure. Compare throughput only after warm steps complete.
- Measurement contract: total step time, exposed generation time, policy training time, refit time, valid tokens/s/GPU, generation KL, Router Replay logprob tails, idle/buffer-starvation/refit-bubble time, and terminal scheduler status.
- Learning handoff: explain sequence importance ratios, policy lag, Router Replay logprob tails, shard quarantine/restart/refit-communicator rebuild, and the limits of every throughput comparison in `handoff.md` for the user to study.
