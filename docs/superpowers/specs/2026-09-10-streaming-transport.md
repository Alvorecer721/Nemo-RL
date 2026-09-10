# Streaming, byte transport and refit regression design

The user approved this design on September 10, 2026. Base: synchronized source
`7197ac715`, on an isolated branch; do not modify the active upstream-sync tree.

1. Select whole GRPO groups compatible with the actual backend's DP and enabled
   training/logprob microbatch constraints before claiming them. With DP6 and
   16 responses/group, 25 ready groups select 24 and retain one. Keep upstream's
   `claim_for_training`, checkpoint ownership and successful-step release lifecycle.
   Validate impossible bounds/targets and handle dropped-group tails explicitly.
   Preserve one optimizer update and full-batch normalization across chunks.
2. Broadcast int16 tensors through contiguous uint8 views of logical source and
   receive allocations. Preserve the existing descriptor, source tensor/device,
   upstream error propagation, PackedTensor behavior and strided inputs. This is
   broadcast-only, on the same-endian fleet. No int32 conversion allocation.
3. Recover the useful refit reproductions as regression tests of current APIs,
   using current CSCS image tooling. Retain generation pause and watchdog behavior.
   Modify synchronization only if a focused regression establishes a defect.

Keep dependency pins and lockfiles unchanged. Keep rollout PP1; trainer PP2/PP4
in refit probes is permitted. No M2N/NCCL-EP activation. Report host, distributed
CPU and GPU evidence separately; no throughput claim without a matched measure.
Existing issues: streaming #32, byte transport #39. Do not duplicate issues.
