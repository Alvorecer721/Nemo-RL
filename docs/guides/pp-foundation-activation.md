# PP foundation activation audit

The September 2026 foundation integrates NeMo-RL through `5368eff5f`, Bridge through `75d7b9eb`, and MCore through `b1fe7599`, while preserving the Apertus fork. The committed Bridge/MCore pins include the local compatibility fixes. Transformer Engine is `2.18.0+27486e03`; vLLM remains `0.26.0`.

This audit distinguishes features reached by NeMo-RL from features that only exist in its dependencies. The source inventory covered ten new NeMo commits, 60 Bridge commits, 79 MCore commits, and the preceding 16 local runtime/image commits. Validation results and exact image/source identities are recorded in the root `integration-ledger.md`.

## Settings activated in this integration

The [Apertus 70B recipe](../../examples/configs/recipes/llm/grpo-apertus1p5-70b-16n4g-tp2pp4-gsm8k-2k-windowed-optimized.yaml) now uses:

```yaml
policy:
  logprob_chunk_size: 256
  megatron_cfg:
    defer_fp32_logits: true
```

Both settings are required for the measured memory benefit. Megatron already produces BF16 logits before its default FP32 output promotion. Deferring that promotion moves the exact BF16-to-FP32 numeric conversion into the chunk loop; it does not add a forward downcast. Reinterpreting raw bits as FP32 would change the values and is inappropriate for logprob arithmetic. On GH200, with batch 1, sequence length 2048, vocabulary 266752 and TP2, logprob/gradient parity passed. Peak incremental allocation fell from 3.278 GB to 1.367 GB; the isolated forward/backward operation increased from 6.456 ms to 13.697 ms. Multiplying the additional 7.240 ms by 128 microbatches gives an estimated 0.927 seconds per optimizer update. This is an operation-count estimate, not a matched full-step A/B measurement.

The 58% reduction is specific to that temporary allocation, concentrated on the pipeline stage computing vocabulary logprobs. It does not establish a 58% reduction in whole-worker memory or extra KV-cache capacity on the separate rollout nodes. The requested microbatch 2 trial held global batch 768 fixed, but failed before its first optimizer update in TE Linear weight-gradient GEMM on a 95 GiB GPU. A 336 MiB allocation failed with 326.81 MiB free. Chunking alone is therefore insufficient for this MB2 setup, and the recipe retains MB1.

The fresh four-update MB1 control passed: warm policy time averaged 40.87 seconds, full-step time 49.88 seconds, and recorded allocated/reserved peaks were 78.20/87.80 GiB across 48 training ranks. Checkpoint writes were disabled in both arms. There is no MB2 speedup measurement because it did not complete an update. Further MB2 or 12k work needs additional measured activation-memory savings; reducing recomputation increases activation memory and is a separate MB1 performance experiment. Preserve the data, loss and optimizer settings and compare full-update time, peak memory and numerical behavior.

NeMo also now supplies its actual packing setting to Bridge's HybridEP safety helper. Bridge moved that safeguard to dataset configuration, which NeMo does not populate. The correction restores padding for eager packed HybridEP and retains the graph/backend guards. The regression produced two expected failures before the correction and all ten cases passed afterward. Dense Apertus does not exercise this MoE path.

## Existing active paths

No additional activation is needed for local HF parameter views used by `nccl_reshard`, retained rollout metrics, single-controller epoch limits, fully parallel checkpoint loading, asynchronous checkpoint saves/constant-structure caching, or concurrent single-controller trainer/vLLM initialization.

Upstream's legacy Megatron-inference initialization overlap does not create a new startup optimization for the single-controller vLLM path, which already initializes the two sides concurrently.

## TE graphs, attention and communication overlap

| Candidate | Current connection | Prerequisite before enabling |
|---|---|---|
| Local training CUDA graphs | NeMo forwards `policy.megatron_cfg.cuda_graph_impl: local`, graph modules, warmup and TE RNG settings; MCore's schedules reach capture. | Stable tensor shapes and packed metadata across training/logprob calls, plus capture/replay and gradient checks. Constant total packed width alone is insufficient. |
| FlashAttention 3 versus cuDNN fused attention | TE 2.18 exposes per-version eligibility switches. The qualified Megatron worker contains FA2 `2.8.1`; FA3/FA4 and `flash_attn_interface` are absent. | Install FA3 in the actual worker environment and rebuild its dependency image; select the backend explicitly and confirm TE's debug output before A/B timing. |
| TP communication/GEMM overlap | `policy.megatron_cfg.model_overrides.tp_comm_overlap` reaches Bridge's userbuffer initialization through NeMo's `initialize_megatron` call. | Match the exact flattened-token shape of the allocated userbuffers across every training/logprob pass, or implement appropriate phase-dependent buffer ownership. Profile exposed communication before tuning. |

[TE 2.18](https://docs.nvidia.com/deeplearning/transformer-engine/release-notes/) adds THD attention capture and further FP8 GroupedLinear capture support. These remove TE limitations but do not prove capture for NeMo's changing packed batches or GLM's separate DSA attention implementation. Dense Apertus does not use MoE GroupedLinear. The bounded foundation qualification uses unpacked 2k inputs; it is not the proposed packed-8k experiment.

`NVTE_FLASH_ATTN_V3=1` permits a version; it does not force TE to choose it. The [pinned selector](https://github.com/NVIDIA/TransformerEngine/blob/27486e03cfc1fa41f6932dcecdc47c71c47eac3e/transformer_engine/pytorch/attention/dot_product_attention/utils.py) prefers eligible cuDNN fused attention on Hopper. NeMo's `megatron_cfg.attention_backend` override also affects selection. Use `NVTE_DEBUG=1` and `NVTE_DEBUG_LEVEL=2` for one diagnostic run to verify the actual choice.

For TP overlap, Bridge allocates userbuffers from `model.seq_length * TrainingConfig.micro_batch_size / CP`. NeMo currently supplies training-config microbatch size 1, while real training/logprob microbatches and padding can differ; the provider sequence length can also be the HF maximum context. TE checks [exact buffer element counts](https://github.com/NVIDIA/TransformerEngine/blob/27486e03cfc1fa41f6932dcecdc47c71c47eac3e/transformer_engine/pytorch/csrc/extensions/comm_gemm_overlap.cpp), so allocating a larger maximum is not sufficient. Reinitializing the global userbuffer registry must also respect outstanding operations and captured graphs. This is a shape/lifecycle integration task, not a missing initialization call.

The focused worker probe reports PyTorch NCCL **2.28.9**, while 70B refit logs also contain native NCCL **2.30.7** banners. Neither observation alone identifies every communicator's loaded library. Build headers and installed wheel metadata are also insufficient to qualify a different transport. `NVTE_WITH_NCCL_EP=0` remains set; the newer M2N NCCL transport is not qualified by this image.

## Other available features that need wiring or qualification

| Feature | Current limitation / next gate |
|---|---|
| RNG checkpoint improvements | The focused MCore RNG/sampler tests pass, but Bridge has independent checkpoint code and NeMo sets `load_rng=false`. Port and expose the appropriate semantics before claiming exact RNG continuity. |
| FP32 optimizer-load memory helper | NeMo calls the loader outside Bridge's guarded training-setup context. Wire it with state-equality and peak-memory checks; importing Bridge alone does not enable it. |
| Fully parallel checkpoint selectors | Basic loading is active. Some group/exchange controls and object-loading options do not reach both wrappers; forward supported settings or reject unsupported combinations. |
| CPU shared-memory checkpoint staging | Requires coordinated NeMo and Bridge writer flags and cleanup behavior, followed by save/resume and memory checks. |
| Generation-fleet recovery | Disabled in the current recipe. Test both failure at a step boundary and a rank lost during refit; a lost NCCL context is not universally recoverable. |
| Keep-mode vLLM pause during refit | The legacy collector calls it; the single-controller path has different admission/abort handling. It needs surviving-replica targeting and in-flight failure tests. A KV-cache flag does not connect this path. |
| MTP-only backbone freezing | A model flag alone does not install MCore's training-loop freeze hook in NeMo; the base output head can remain trainable. Fully connect and validate this mode before using it. |
| MoE metric normalization | Bridge's training-logger correction does not update NeMo's independent metric aggregation. |
| Baked source for 70B startup | Baked worker interpreters are selected, but the validation launcher imports the explicit shared checkout. Separate runtime-source selection from provenance validation before changing it. |

The optimizer-resume checkpoint's actual tensor, tracker and train-state destinations are captured correctly. Bridge's asynchronous completion message and serialized `checkpoint.save` can show the earlier conversion-cache path after NeMo restores its temporary configuration override. NeMo reconstructs the load path explicitly on resume. The qualification records retain this reporting limitation; they do not infer a redirected write from that message.

## DP imbalance and packing

MCore's `sequence_packing_scheduler` belongs to its training/data loop and is not connected to NeMo-RL through a model override. NeMo packs each global batch chunk before assigning bins round-robin to DP ranks; it does not only pack within already assigned ranks.

In the current unpacked 70B recipe, dynamic batching is also disabled and dense inputs use a shared cross-DP padding target. Longer valid responses alone do not establish more dense compute. Measure padded shapes, microbatch counts, compute time and collective wait time per training rank before proposing token-balanced sharding. The existing `train/per_worker_token_counts` list comes from rollout workers; it is not that trainer-DP measurement.

Rollout PP2, GLM DSA/MTP execution, all historical checkpoint formats, training graphs, FA3, FP8 training and TP userbuffer overlap remain outside this bounded qualification. Generic bias-activation fusion is also inappropriate for Apertus XIELU, and fused-linear logprobs have a separate parameter-gather-overlap constraint.
