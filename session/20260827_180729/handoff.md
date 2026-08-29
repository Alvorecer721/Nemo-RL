# Handoff

Read `session_state.md` first. The matched 72-train + 64-rollout baseline is complete as job `3217663`; do not rerun it. Continue from branch `autoresearch/2026-08-29-glm51-mtp3`.

Swiss AI's public BF16 GLM-5.1 serving layout is also eight GH200 nodes / TP32. NeMo-RL cannot lower TP by adding rollout PP because NCCL reshard currently requires PP1, and TP16 cannot hold the 1.508-TB BF16 checkpoint with runtime headroom. The only retained experiment keeps TP32/PP1/EP32 and enables the checkpoint's native MTP layer for three speculative tokens with TP32 local argmax reduction.

Finish focused tests and commit the branch, then submit `infra/slurm/cscs/autoresearch/submit_glm51_sc_mtp3.sh` once `squeue` responds. Freeze the checkout while it runs. Acceptance requires the MTP disk-load marker, nonzero and internally consistent speculative acceptance counters, all existing ten-step learning/KL/tail/Router Replay/CP-identity gates, and scheduler exit `0:0`. Compare warm-step total/exposed-generation time and valid-token throughput directly with `3217663`; startup time is not a throughput result.

Do not delete reservation `SD-69241-apertus-1-5-0`, the 8.946-TB complete checkpoint, or the 1.488-TB conversion cache. Do not add FP8/MXFP8 to this challenger: Swiss AI's FP8 serving precedent is informative, but current GH200 RL certification is BF16 and the newest MXFP8 refit path targets SM100 kernels.
