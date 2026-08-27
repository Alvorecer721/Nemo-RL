# Session state

- Goal: keep the synchronized Apertus NeMo-RL fork production-ready and measure whether doubling the GLM-5.1 rollout fleet reduces trainer starvation.
- Active branch: `autoresearch/2026-08-27-glm51-64inf-scale`.
- Base: fork `main` at `d7ced9573cadf1a391ea55205c60a3013b4d8b27`, plus the exact file diff from open PR #27 (`1c7d48f207dcfca8a4377ebdc58a6d77c236da42`).
- Experiment: 72 Megatron nodes plus 64 vLLM nodes, three SingleController/TransferQueue ready-first steps, TP2/PP18/ETP1/EP16 training and eight TP32/EP32 vLLM replicas.
- Reservation: `SD-69241-apertus-1-5-0`; 469 of 600 nodes were available at the last capacity check.
- Constraint: freeze the committed source while the Slurm job is queued or running.
- Remote cleanup: ten merged NeMo-RL branch heads that were zero commits ahead of `main` were deleted; every other remote branch was left untouched.
- First launch: job `3203601` was intentionally cancelled before model loading after the driver blocked in the broad Git-cleanliness preflight. Ray itself reached `544/544` worker units in 107 seconds.
- Infrastructure recovery: on 2026-08-28, 12 consecutive Capstor, Git, Iopsstor, and Slurm probes completed without error; Capstor/Git/Iops latency was about 12-16 ms and Slurm latency 29-38 ms.
- Current subtask: commit the narrowed runtime-input preflight and recovered submit-wrapper fixes, then relaunch the same 136-node experiment.
