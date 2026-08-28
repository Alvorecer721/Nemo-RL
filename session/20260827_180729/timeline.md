# Timeline

- Audited all 30 branches on `Alvorecer721/Nemo-RL` against `main` and the pull-request ledger.
- Deleted exactly ten merged branch heads with `ahead_by=0`; verified 20 branches remain.
- Confirmed completed job 3186645 used SingleController, TransferQueue, ready-first sampling, 72 training nodes, and 32 inference nodes.
- Confirmed the historical 104-node launch depended on an untracked temporary runner; replaced it with tracked experiment files.
- Confirmed the new experiment changes only inference capacity relative to that 104-node topology: 32 to 64 rollout nodes. It intentionally runs three measured steps.
- Shell syntax and Python byte-compilation passed. Host pytest was blocked by missing Ray; the exact EDF container performs config preflight before model setup.
- Submitted job `3203601`; Ray connected all `544/544` worker units in 107 seconds, then the driver blocked before model loading in a broad `git status` preflight touching a stalled Capstor inode.
- Cancelled job `3203601` intentionally rather than leave 136 nodes idle; Slurm recorded `CANCELLED by 30214` and no training step ran.
- Replaced the broad driver-side cleanliness query with an explicit runtime-input path set. Recovered the submit wrapper's writable scheduler-log path and non-recursive first-level submodule check.
- Sampled infrastructure health 12 times on 2026-08-28: every Capstor, Git, Iopsstor, and Slurm probe succeeded with stable low latency.
- Submitted two sequential exact-image 136-node Pyxis fan-out canaries on 2026-08-29. Jobs `3215160` and `3215161` both passed (`0` from `srun`), producing 136/136 completion markers from 136 unique nodes in 52 and 44 seconds respectively; neither produced a nonempty task error. This clears the earlier `520/544` worker startup failure as transient infrastructure evidence rather than a deterministic image or capacity failure.
