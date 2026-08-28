# Session state

- Goal: keep the synchronized Apertus NeMo-RL fork production-ready and measure whether doubling the GLM-5.1 rollout fleet reduces trainer starvation.
- Active branch: `autoresearch/2026-08-27-glm51-64inf-scale`.
- Base: fork `main` at `d7ced9573cadf1a391ea55205c60a3013b4d8b27`, plus the exact file diff from open PR #27 (`1c7d48f207dcfca8a4377ebdc58a6d77c236da42`).
- Experiment: 72 Megatron nodes plus 64 vLLM nodes, three SingleController/TransferQueue ready-first steps, TP2/PP18/ETP1/EP16 training and eight TP32/EP32 vLLM replicas.
- Reservation: `SD-69241-apertus-1-5-0`; the latest snapshot left roughly 473 of 600 nodes free before the canaries.
- Constraint: freeze the committed source while the Slurm job is queued or running.
- Remote cleanup: ten merged NeMo-RL branch heads that were zero commits ahead of `main` were deleted; every other remote branch was left untouched.
- First launch: job `3203601` was intentionally cancelled before model loading after the driver blocked in the broad Git-cleanliness preflight. Ray itself reached `544/544` worker units in 107 seconds.
- Infrastructure recovery: on 2026-08-28, 12 consecutive Capstor, Git, Iopsstor, and Slurm probes completed without error; Capstor/Git/Iops latency was about 12-16 ms and Slurm latency 29-38 ms.
- Infrastructure gate: exact-image 136-node Pyxis fan-out jobs `3215160` and `3215161` passed sequentially in 52 and 44 seconds, each with 136/136 ranks on 136 distinct nodes and no nonempty task errors.
- Current subtask: run an overnight matched campaign: (A0) current-source 136-node baseline; (A1) upstream-synced exact-image baseline with behavior held fixed; (A2) telemetry plus generation-fleet health on a healthy fleet; (A3) deliberate vLLM-shard loss with communicator rebuild and continued training.
- Stop rule: preserve interpretable comparisons. Do not stack a behavior change before its matched baseline is terminal; do not call recovery proven without a deliberate shard-loss event and a subsequent completed training step.
- Measurement contract: total step time, exposed generation time, policy training time, refit time, valid tokens/s/GPU, generation KL, Router Replay logprob tails, idle/buffer-starvation/refit-bubble time, and terminal scheduler status.
- Learning handoff: explain sequence importance ratios, policy lag, Router Replay logprob tails, shard quarantine/restart/refit-communicator rebuild, and the limits of every throughput comparison in `handoff.md` for the user to study.
