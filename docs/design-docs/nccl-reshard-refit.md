# NCCL Reshard Refit (Experimental)

> **Experimental**: `nccl_reshard_refit` is an experimental feature.

The default non-colocated transport broadcasts every **full** parameter tensor from the
training ranks to every generation rank. `nccl_reshard_refit` replaces that for the bulk
of the payload with a **shard-to-shard reshard**: each training rank sends only its local
shard, and each generation rank receives exactly the bytes of its own (differently
parallelized) shard. It avoids a full-tensor gather and broadcast: a destination
materializes only the weight shard required by its own layout.

## Enabling It

Add the config key (it is `NotRequired` in `PolicyConfig`, so use `+` when overriding
from the CLI):

```bash
uv run ./examples/run_grpo.py \
  --config <your_config>.yaml \
  policy.generation.colocated.enabled=false \
  policy.generation.refit_transport=nccl_reshard
```

At setup, `check_nccl_reshard_refit_support()` validates the configuration and raises a
single `ValueError` listing every violation. The current requirements are:

* **Non-colocated only** — `policy.generation.colocated.enabled=false`. The colocated
  path uses IPC and is unaffected by this feature.
* **Megatron training backend** — `policy.megatron_cfg.enabled=true` (the DTensor
  training backend is not supported yet.).
* **vLLM or Megatron generation backend** — `policy.generation.backend` must be
  `vllm` or `megatron` (SGLang and TRTLLM are not supported yet).
* Training-side Megatron supports expert tensor parallelism when the generation
  destination is Megatron. A vLLM destination still requires
  `expert_tensor_parallel_size: 1`. Custom PP layouts
  (`pipeline_model_parallel_layout`, virtual PP > 1, embedding/loss
  pipeline-split accounting) are not supported yet.
* **Generation-side ETP with `inference_optimized` is pinned to 1.** Those MoE
  layers do not implement expert tensor parallelism and raise whenever the
  *resolved* ETP exceeds 1 — and an omitted ETP resolves to TP, not 1. So
  `merged_inference_megatron_cfg` pins generation-side
  `expert_tensor_parallel_size` to 1 for that transformer implementation, which
  is what lets generation-side TP > 1 work; the reshard then handles the
  train-ETP → gen-ETP=1 gather. Other transformer implementations retain their
  configured ETP. An explicitly requested generation-side ETP > 1 with
  `inference_optimized` is rejected by config key name rather than surfacing as
  a raw MCore assert at model build.
* **Precision** for vLLM supports BF16 train ↔ BF16 gen, blockwise-FP8 train
  (`fp8_param=true` + blockwise recipe) ↔ FP8 gen, and BF16 train → MXFP8 gen
  (`vllm_cfg.precision=fp8`, `vllm_cfg.is_mx=true`). Blockwise-FP8 train →
  MXFP8 gen is not supported.
* Megatron generation accepts BF16 or supported Transformer Engine FP8 training
  parameter storage, including blockwise FP8 and MXFP8 with `fp8_param=true`.
  Quantized sources are materialized as logical BF16 for transport; the
  destination either stores BF16 or quantizes each complete local weight into
  MXFP8. Before every refit, an explicit parameter sync materializes optimizer
  updates that would otherwise wait for the next overlapped all-gather. When
  MXFP8 parameter all-gather reuses the gradient buffer, that aliased allocation
  stays GPU-resident across refit so
  persistent DDP/autograd views remain valid; ordinary gradient buffers and
  optimizer state are offloaded only when
  `policy.generation.mcore_generation_config.offload_policy_before_refit` is true.
* **The wire format is always BF16, even for MXFP8 train → MXFP8 gen.** This is
  forced by the upstream API, not a shortcut, and is worth stating because it
  means an MXFP8 trainer does *not* get a smaller refit (expect ~2x the
  theoretical MXFP8 wire size, plus a dequantize on the source and a re-quantize
  on the destination). Three reasons it cannot currently be otherwise:
  * TE MXFP8 and MCore MXFP8 are not byte-compatible. MCore itself dequantizes
    and re-quantizes when converting between them, deliberately, "to avoid any
    numerical differences between TE and mcore MXFP8 formats"
    (`megatron/core/inference/quantization/utils.py`).
  * `MXFP8Tensor`'s only data constructor is `from_bf16`; `copy_` delegates to
    `quantize_`, which calls `from_bf16`. There is no relayout entry point.
  * MCore stores *swizzled* scales, padded to multiples of 128 rows and 4
    columns, so a shard of the swizzled scales is not a shard of the logical
    scales. An alignment-aware MXFP8 transport would have to unswizzle,
    re-slice, and re-swizzle — most of the cost of a requantize anyway.
  The benefit of this path is capability (M-to-N reshard into a Megatron
  engine), not bandwidth.
* BF16 FlashInfer TRTLLM MoE is supported through vLLM's native
  layerwise-reload path. Its grouped expert weights must use expert-parallel
  destination sharding with linear expert placement; tensor-sharded expert
  destinations and round-robin placement are rejected. This path does not
  support an FP8 KV cache or a co-trained MTP drafter; setup rejects both
  combinations.
* vLLM expert parallelism is supported with the NeMo RL convention
  `expert_parallel_size == tensor_parallel_size`.
* Megatron generation supports expert parallelism; generation-side expert tensor
  parallelism is available only when `transformer_impl` is not `inference_optimized`.
* Megatron generation uses the same top-level selector as other backends:
  `refit_transport=null` selects NeMo-RL's packed collective,
  `refit_transport=mcore` selects Megatron Core's native refit, and
  `refit_transport=nccl_reshard` selects M-to-N. `refit_backend` is consulted
  only for `refit_transport=mcore`. Colocated Megatron generation requires
  `refit_transport=mcore`, because its refit is carried by the in-place
  wake-reshard; the other transports are rejected rather than silently ignored.
* **Generation-side pipeline parallelism is supported for Megatron training.**
  vLLM and Megatron generation can use different TP and PP layouts from the
  trainer. Setup discovers each generation rank's actual local weights and
  validates their ownership before installing the transfer plan on both sides.
  PP ranks that do not own a weight participate in communicator setup without
  allocating storage for that weight. Tied embeddings remain on the misc path
  so the backend's regular loader preserves their stage-specific semantics.
* **No ModelOpt real quantization** — `policy.generation.real_quant=false`. Real-quant
  rollouts refit through vLLM's layerwise-reload weight loaders, which the bulk
  `xferdtensor` writes bypass.

Operational knobs:

* `NRL_REFIT_NUM_STREAMS` (default `2`) — number of CUDA streams the generation side
  uses to overlap per-PP-stage bulk reshards. Having higher number can increase
  concurrency of the transportation when PP-size is large, but will have higher
  memory overhead.

## Design Overview

FFN layers are the dominant payload in the weight transfer. Our profiling shows
that the MoE FFN layers account for 97%-98% of the model weights. To balance
performance and software sustainability, we chose a dual-path strategy for the
nccl-reshard-refit implementation:

* **Bulk path** — the FFN projection weights (`gate_proj` / `up_proj` / `down_proj`
  `.weight`, dense MLP and MoE experts alike; see `is_nccl_reshard_param()`). These are
  resharded shard-to-shard with `xferdtensor` over dedicated NCCL communicators. For
  large models this covers the vast majority of the refit bytes. The fork also
  selects supported attention output projections and eligible untied vocabulary
  weights; the source catalog records the actual selection.
  Two FFN-named groups are explicitly excluded and ride the misc path instead:
  shared-expert weights (`*.shared_expert.*`, which fuse differently on the vLLM
  side) and co-trained MTP drafter weights (which vLLM keeps in a separate
  drafter module updated through `load_weights`). Co-trained MTP is not supported
  with BF16 FlashInfer TRTLLM; this routing applies to other supported backend
  combinations. MTP weights are recognized two ways: bare-`mtp.`-prefix HF names
  (NemotronH, Qwen3.5) via
  `is_nccl_reshard_param()`, and DeepSeek-style MTP exported as trailing
  `model.layers.N` indices via provenance — the Megatron-side name carries an
  `mtp.` module segment (bare for LM bridges, `language_model.mtp.*` for the VL
  and EXAONE bridges), so the worker excludes those HF layers when building the
  metadata (`_collect_mtp_hf_layer_names()`).
* **Misc path** — everything else (embeddings, attention projections, layernorms, the
  MoE router, `lm_head`, FP8 `_scale_inv` siblings, FP8 KV-cache scales, …). FP8
  KV-cache scales are supported only by backend combinations that allow an FP8 KV
  cache; BF16 FlashInfer TRTLLM rejects that configuration at setup. These tensors
  ride a packed broadcast (conventional `packed_tensor.py` implementation) over the
  shared `model_update_group` and are loaded on the generation side through the
  backend's regular `load_weights` machinery.

The feature is integrated into the `nemo_rl/weight_sync/` framework. For vLLM,
`create_weight_synchronizer(...)` returns an `NcclReshardWeightSynchronizer` directly.
For Megatron generation, the existing `MegatronWeightSynchronizer` retains ownership of
the inference-engine lifecycle and delegates only the transfer to an
`NcclReshardWeightSynchronizer`.

### Execution Flow: Setup Time

`NcclReshardWeightSynchronizer.init_communicator()` runs three steps once, before
training starts:

The generation backend declares whether it needs Bridge's physical export or logical
weights. The synchronizer passes that payload requirement to the source worker; the
source worker does not inspect or branch on the destination backend's name. Requesting
logical weights is a Megatron-inference-specific exception: vLLM keeps the universal
Bridge-export representation, while Megatron inference requests logical weights because
its destination storage is built by MCore rather than Bridge. Megatron workers are assigned
an explicit source or destination refit role and expose the same
`prepare_refit_info`, `build_hf_to_local_param_map`,
`prepare_nccl_reshard_refit_info`, and `nccl_reshard_refit` entry points in either
role.

1. **`init_collective()`** — creates the `model_update_group`, a NCCL group spanning all
   training and generation ranks. The bulk path does not use it; it carries the misc
   packed-broadcast, including FP8 KV-cache scales for backend combinations that support
   them, identical to the conventional collective transport.
2. **`init_nccl_reshard_comm_group()`** — creates the bulk-path communicator(s): **one
   NCCL group per training PP stage**, each spanning that stage's training ranks plus
   *all* generation ranks (non-PP is simply `pp_size == 1`, a single group over
   everything). Keeping the bulk path on its own communicators decouples it from the
   misc broadcast. Each rendezvous address comes from the actual source-stage
   leader's placement bundle, including when one Ray placement group spans several
   nodes or its bundle order differs from model-rank order.
3. **`prepare_nccl_reshard_refit_info()`** — the metadata exchange. The **training side
   builds a backend-agnostic description** of every bulk parameter
   (`build_nccl_reshard_refit_info()` in `nemo_rl/weight_sync/nccl_reshard_utils.py`),
   keyed strictly by **HuggingFace parameter names**, and ships it to the generation
   side. Before shipping, `make_nccl_reshard_refit_info_wire_safe()` converts the
   `MeshInfo` rank tensors and `Shard`/`Replicate` placements into plain dicts/lists —
   Megatron patches torch's storage unpickler, so raw tensor pickles would require
   `import megatron` inside the vLLM worker. The generation side rebuilds the objects
   with `restore_refit_info_placements()`.

   For generation PP > 1, the destination first reports local canonical weight
   names, logical shard shapes and dtypes, and its actual PP/TP/DP/EP coordinates.
   `finalize_nccl_reshard_refit_info()` checks complete topology and exactly one
   owning PP stage for every decoder weight. Its destination mesh includes all TP
   shards and DP replicas of that stage. Some backends also allocate vocabulary
   weights on several PP stages; each such stage must report a complete copy,
   and those copies share the mesh's replica axis. Partial copies are rejected.
   The source and destination install the same validated plan. A membership
   rebuild repeats discovery and validation over the surviving complete engines.
   Generation PP = 1 retains the existing setup path.

The derived metadata (`nccl_reshard_refit_info`) contains, per parameter:

* `name` — the HF parameter name (per-expert MoE weights are grouped into a single
  `...experts.{gate,up,down}_proj.weight` entry of shape `[num_experts, ...]`, tagged
  with `grouped_expert_proj`);
* `global_shape` and `dtype` of the full, unsharded tensor;
* `src_mesh_info` / `src_placements` — the training-side rank mesh (`MeshInfo`) and
  DTensor-style `Shard`/`Replicate` placements, derived from the training parallelism
  (TP/EP/PP; experts live on an EP mesh, everything else on a TP mesh);
* `dst_mesh_info` / `dst_placements` — the same for the generation side (TP mesh, or an
  EP mesh for experts when vLLM expert parallelism is enabled);
* `pp_stage` — which training PP stage owns the parameter (present when `pp_size > 1`),
  used to route it to the right per-stage communicator.

Alongside it, `misc_meta` (an **ordered** dict of `name -> {shape, dtype}`) describes
every misc parameter; the order is load-bearing because producer and consumer walk it in
lockstep during the packed broadcast.

Finally, both sides build their `hf_to_local_param_map`: a mapping from each bulk HF
parameter name to a `LocalParamSpec(base, pre, post)` describing how that parameter is
realized **locally**:

* On the **training side**, a direct parameter's `base` is the live TP/EP-local shard
  (sent as-is); grouped MoE experts get a `pre` hook that stacks this rank's per-expert
  views into a `[num_local_experts, ...]` tensor fresh at each refit.
* On the **generation side**, a direct parameter's `base` is the live vLLM parameter
  (received into in place). Conventional fused parameters use `pre`/`post` hooks to
  receive a component and copy it into the appropriate local region. BF16
  FlashInfer TRTLLM grouped experts instead receive into canonical EP-local staging
  tensors; `post` loads each logical expert with its global expert ID through vLLM's
  native weight loader.

### Execution Flow: Refit Time

Every training step (with in-flight weight updates, concurrently with generation),
`NcclReshardWeightSynchronizer.sync_weights()` triggers both sides:

* `pre` contains a function that should be executed in-flight before the refit.
* `post` contains a function that should be executed in-flight after the refit.

* The **training side** walks `per_layer_params`, skipping parameters owned by other PP
  stages. For each parameter it resolves the `LocalParamSpec`, runs `pre` (expert
  stacking) if present, wraps the local shard in a `DTensorRef` (which reports the
  *global* shape while holding only the local tensor), and calls
  `xferdtensor(src, src_mesh, src_placements, None, dst_mesh, dst_placements, group,
  stream)`.
* The **generation side** walks the same metadata in the same order — every rank in a
  comm group must issue the same sequence of transfers. Per-PP-stage parameter groups
  are distributed across `NRL_REFIT_NUM_STREAMS` CUDA streams so different stages'
  reshards overlap. For each locally owned parameter it runs `pre`
  (receive-buffer allocation), calls
  `xferdtensor(None, ..., dst, ..., group, stream)`, then `post` (copy back into the
  fused parameter or load staged TRTLLM experts). After every transfer completes, the
  TRTLLM path finalizes vLLM's native layerwise reload once to restore the packed runtime
  layout. A rank on another generation PP stage passes a metadata-only destination
  descriptor and skips both hooks. It still enters the transfer call because
  communicator splitting and native transport setup can involve the full parent
  group.

### The Misc Path

After the bulk reshard completes, the misc parameters are transferred.
This part is reusing the same code implementation as the conventional packed_tensor refit.

## Decoupling Backend-Agnostic Parts and Backend-Dependent Parts

To facilitate backend extension, the implementation cleanly separates backend-agnostic
components from backend-dependent ones. As a result, extending to a new backend only
requires implementing the backend-dependent components.

**Backend-agnostic** (no knowledge of Megatron or vLLM):

* `nemo_rl/weight_sync/nccl_reshard_utils.py` — the metadata builder
  (`build_nccl_reshard_refit_info`), mesh/placement derivation (`build_mesh_info`,
  `get_placements`, `MeshInfo`), the bulk-path whitelist (`is_nccl_reshard_param`),
  per-expert grouping into HF-convention grouped entries, the config validator, and the
  `LocalParamSpec`, `RefitCtx`, and `HFToLocalParamMap` contracts. All parameter sharding
  required by the different types of parallelism is handled by this utility.
* `nemo_rl/weight_sync/xferdtensor.py` — the transfer entry point and its transport
  dispatch (see below).
* `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py` and the factory routing —
  the lifecycle orchestration.

The glue that makes this work across backends is the **HF naming convention**: the
training side must describe its parameters using HF names and global shapes, and the
generation side maps those HF names onto whatever its own storage layout is.

**Backend-dependent**:

* **Training side** (`megatron_policy_worker.py`): producing the HF-named state-dict
  metadata; building `hf_to_local_param_map` — resolving each HF name to the local
  Megatron tensor view and providing the grouped-MoE `pre` stacking hook; the
  `init_collective` / `init_nccl_reshard_comm_group` bootstrap methods; the
  `nccl_reshard_refit()` send loop; the misc packed-broadcast producer.
* **Generation side** (`vllm_backend.py`): building `hf_to_local_param_map` — mapping HF
  names onto vLLM's fused parameters (`qkv_proj`, `gate_up_proj`, grouped-expert
  `w13_weight`/`w2_weight`) with `pre`/`post` hooks for slice regions or canonical
  TRTLLM staging, which is deliberately **shape-driven** so the same code handles
  supported generation parallelism; the comm bootstrap methods; the
  `nccl_reshard_refit()` receive loop; the misc consumer feeding `load_weights`; and
  backend-specific finalization after all weights arrive.
* **Megatron generation side** (`megatron_worker.py`): mapping the same canonical
  HF FFN shards to local fused dense/expert views. BF16 destinations receive in
  place; MXFP8 destinations use short-lived BF16 staging buffers and quantize into
  their persistent MCore storage. Misc weights continue through Megatron Bridge's
  packed-broadcast import path.

**To extend to a new backend**, provide a destination map from canonical HF weights
to that backend's local storage. Both backends implement this as
`build_hf_to_local_param_map`; Megatron derives its targets from Bridge conversion tasks.
Everything else follows the fixed transport contract.

**The one backend-specific implementation — `build_hf_to_local_param_map`:** resolve
each bulk HF name to your local storage as a `LocalParamSpec` — `base` for tensors
sent/received as-is, and `pre`/`post` hooks wherever your layout requires staging
(fused/merged tensors, layout conversions, grouped-expert stacking). Backends that
rebuild runtime storage may also need one transport-level finalizer after all specs have
run. These are the only places the backend's parameter layout is encoded; all cross-mesh
byte movement is already handled by the shared metadata and `xferdtensor`.

(A new *training* backend additionally has to produce the HF-named metadata — names,
global shapes, dtypes, and the parallelism description the agnostic builder consumes —
inside its `prepare_nccl_reshard_refit_info`, since only the backend knows how to read
its own weights. A new *generation* backend simply consumes the shipped metadata.)

**Copy-paste boilerplate** (identical in shape to the existing backend; only
names/attributes change):

1. `prepare_nccl_reshard_refit_info` — restore the shipped metadata and call
   `build_hf_to_local_param_map` once.
2. The communicator bootstrap (`init_collective`, `init_nccl_reshard_comm_group`) — the
   same `StatelessProcessGroup` setup; the only requirement is the rank convention:
   training ranks first (per-stage-local for the bulk groups), generation ranks after.
3. The `nccl_reshard_refit()` loop — walk `per_layer_params` in metadata order (grouped
   by `pp_stage` across `NRL_REFIT_NUM_STREAMS` streams), resolve each `LocalParamSpec`,
   run `pre`, call `xferdtensor`, run `post`. It only touches the generic spec/metadata
   contracts, never your layout.
4. The misc producer/consumer — reuses the conventional packed-broadcast path.

## `xferdtensor` Transports

`xferdtensor()` (in `nemo_rl/weight_sync/xferdtensor.py`) is the single entry point both
workers call. It has the 8-argument signature

```python
xferdtensor(src_tensor, src_mesh, src_placement,
            dst_tensor, dst_mesh, dst_placement,
            process_group, stream=None)
```

and dispatches to one of three implementations:

* **Native NCCL M2N** — the reshard operation provided by the **nccl4py
  wrapper** (`nccl.m2n.reshard`). This is selected when the package is available
  and the communicator reports device API support. The pinned M2N mesh API also
  requires at most two mesh axes and contiguous ranks in row-major order:
  the local shards, mesh rank grids, and placements are handed to the NCCL library,
  which executes the cross-mesh redistribution natively.
* **`xferdtensor_python_impl`** (`nemo_rl/weight_sync/xferdtensor_python.py`) — a pure
  Python + nccl4py-collectives **backup implementation** for environments without a
  proper NCCL / nccl4py reshard installation. It computes the exact shard overlaps
  between the source and destination layouts, moves each destination region once via
  batched point-to-point (with striped receives across replica groups), and fans out to
  replicas with cached split-communicator broadcasts. It is a drop-in with the same
  signature and is selected automatically when `nccl.m2n` is not importable or
  the communicator lacks device API support, or a mesh is outside the native
  API's supported geometry. A successful single-node native
  test does not establish native support across nodes. This fallback still uses
  NCCL for the shard transfers and does not gather and broadcast each full weight.
* **`xferdtensor_golden`** (`nemo_rl/weight_sync/xferdtensor.py`) — a pure function-only
  implementation intended for debugging. This implementation simply broadcasts the full
  tensor to the destination ranks, which then discard the unused parts. While not performant,
  it guarantees functionally correct outputs.

Both transports honor the `stream` argument so the transfer is ordered with the caller's
`pre`/`post` staging work on one CUDA stream.

## Generation PP Validation

`tests/functional/nccl_reshard_pp.py` exercises a local BF16 dense/MoE Qwen3 or Apertus
checkpoint with a Megatron trainer. It changes source parameters before every
warm refit, compares every bulk shard with an independently updated HF tensor,
and generates between updates. For vLLM it also checks every local model parameter
directly, including fused QKV, norms, and tied vocabulary aliases. Storage checks,
mutations, and generation are outside the timed refit.
MoE checks reconstruct grouped tensors from numerically ordered HF expert keys
and check vLLM's fused expert storage using its actual expert placement. The
current MoE raw-storage oracle selects the Triton BF16 backend.

`--update-mode optimizer` replaces artificial scaling with real distributed
Adam updates. After each update, all training ranks participate in an independent
full Megatron-Bridge HF export. Destination weights must match that export and
must change from the previous refit. Training and reference export are outside
the timer. `--train-dp`, `--train-ep`, `--train-etp`, `--gen-ep` and `--gen-etp`
allow different expert layouts on either side. The driver runs the production
configuration validator before allocating actors. vLLM tests default source ETP
to 1, as required by that validator; MoE training with TP > 1 enables sequence
parallelism.
For large checkpoints, `--reference-workers N` overlaps independent CPU
checkpoint checks. Every tensor is still compared in full; memory scales with
the number of concurrent checks. Partial timings are written before inspection,
and the report gains `passed: true` only after all checks finish.

The optional PP1 reference checks identical tokens and generation logprobs at
`atol=1e-5, rtol=0` with the same backend and TP. Cross-backend BF16 logprob
differences are reported separately: kernel differences can exceed 0.2 after
repeated artificial scaling even in the PP1 control.

For example, with the matching worker environments and enough GPUs available:

```bash
# Four GPUs: reference with training TP2 and generation TP2/PP1.
uv run tests/functional/nccl_reshard_pp.py \
  --backend vllm --model /path/to/Qwen3-0.6B \
  --train-tp 2 --gen-tp 2 --gen-pp 1 --output pp1.json

# Eight GPUs: different training/generation PP, with a PP1 output oracle.
uv run tests/functional/nccl_reshard_pp.py \
  --backend vllm --model /path/to/Qwen3-0.6B \
  --train-tp 1 --train-pp 4 --gen-tp 2 --gen-pp 2 \
  --reference-report pp1.json --output pp2.json
```

Set `NRL_XFERDTENSOR_PYTHON=1` to qualify the Python NCCL path explicitly.
Otherwise the runtime logs the selected implementation and device API capability.
The separate four-GPU `tests/functional/nccl_reshard_pp_transport.py` gate accepts
`--mode python` or `--mode native`; native mode fails if native M2N is unavailable.

On 2026-09-20, GH200 tests with the fork's vLLM 0.29.0 / Torch 2.13.0 image gave
the following Qwen3-0.6B BF16 results. Each row includes five changed warm refits;
setup and the first refit are excluded. Both nodes have four GPUs in the
two-node rows.

| Generation backend | Nodes | Train TP/PP → Gen TP/PP | Transport | Warm median | Warm min–max |
|---|---:|---|---|---:|---:|
| vLLM | 1 | 2/1 → 2/1 | Python NCCL | 108 ms | 106–109 ms |
| vLLM | 1 | 2/1 → 1/2 | Python NCCL | 108 ms | 105–152 ms |
| vLLM | 1 | 2/1 → 1/2 | Native M2N | 111 ms | 109–115 ms |
| vLLM | 2 | 1/4 → 2/2 | Python NCCL | 142 ms | 128–148 ms |
| Megatron | 2 | 2/2 → 2/2 | Python NCCL | 261 ms | 261–271 ms |

All bulk checks passed. The vLLM TP2/PP2 run matched its TP2/PP1 reference with
zero logprob difference at updates 0, 1, and 5. The Megatron timing includes its
inference-engine pause/resume lifecycle; vLLM's non-colocated refit call measures
the transfer and weight loading. These rows are not a comparison of generation
throughput or an extrapolation to larger models.

The two-node runtime reported `device_api_support=False` and automatically used
Python NCCL. Native M2N was verified on one node only. No dependency upgrades or
image-fingerprint bypasses were needed.

Apertus-1.5-70B BF16 on two nodes, training TP1/PP4 to vLLM TP2/PP2,
completed three changed warm refit calls in **3.212, 3.212 and 3.239 seconds**
(median **3.212 seconds**); the cold refit took 3.334 seconds. The logged payload
was 123.14 GiB bulk plus 12.50 GiB misc. This also used Python NCCL. All four
iterations passed all 484 bulk-shard and 1,606 local parameter-tensor checks;
generation/source-policy comparisons passed at updates 0, 1 and 3. The receipt
also preserves an earlier run that ended during its final CPU check as partial.

Qwen3-Coder-30B-A3B-Instruct BF16 also passed with expert parallelism:

| Generation | Nodes | Training TP/PP/DP/EP/ETP → Generation TP/PP/EP/ETP | Updates | Transport | Warm refits |
|---|---:|---|---|---|---:|
| vLLM | 1 | 1/1/2/2/1 → 1/2/1/1 | Two deterministic BF16 updates | Native M2N | 0.462, 0.459 s |
| vLLM | 3 | 1/2/4/4/1 → 2/2/2/1 | Two real distributed Adam updates | Python NCCL | 0.866, 0.869 s |
| vLLM | 3 | 2/2/2/4/1 → 2/2/2/1 | One real distributed Adam update | Python NCCL | 0.916 s |
| Megatron | 3 | 1/2/4/4/1 → 2/2/2/1 | Two real distributed Adam updates | Python NCCL | 0.967, 0.974 s |

Every initial and updated bulk shard matched the independent HF reference;
vLLM also passed every raw-parameter check. Generation passed after each refit.
Both TP1-source Adam steps changed all 392 observed vLLM bulk shards and all
388 Megatron bulk shards; the TP2-source step changed all 384 bulk shards.
Source vocabulary padding makes that TP2 case use the misc path for vocabulary
weights, which are included in the full vLLM raw-parameter check. The native row uses contiguous 1D source
and destination meshes on one NVLink node. These different topologies do not
isolate the speed difference between native and Python implementations. The
logged TP1-source 30B payload was 55.91 GiB bulk plus 0.96 GiB misc; the
TP2-source case used 54.75 GiB bulk plus 2.12 GiB misc. Full reference export
and all-weight hashing are outside the reported timings.

Fault probes use `--gen-dp 2 --rebuild-after-stage-loss` to remove a whole engine
after one PP stage exits, then check a changed refit and generation on the
survivor. `--abort-during-refit --refit-timeout 20` exits a stage after its engine
enters receive. The latter requires bounded failure and never serves or attempts
to reuse a lost CUDA context. The final two-node recovery rebuilt in 3.36 seconds;
the in-flight two-node probe returned the explicit fatal-context error after
21.28 seconds. This exercises
the transport contract; it does not replace full SingleController availability
testing.

These are local GH200 qualification results, not upstream CI results.
The functional probes above reproduce the checks; measured timings depend on
the model, topology, transport and installed dependencies.
The recorded MoE vLLM gates use Triton BF16 expert storage. Quantized PP,
processed TRTLLM MoE storage, and MTP remain outside these measured gates.

## Expected Performance

| Platform | Model | Precision | Train → Gen mapping | XferDTensor fraction | Refit time |
|--|---|---|---|---:|---:|
| H100 | QWEN3 4B (dense) | BF16 | DP8 → TP8 | 66.9% | 0.21–0.34s |
| H100 | QWEN3 30B | BF16 | EP8×PP2 → TP8×DP2 | 95.0% | 0.74–1.00s |
| H100 | QWEN3 30B | FP8 | EP8×DP2 → TP2×DP8 | 93.0% | 2.90–4.00s |
| H100 | DSV3 | BF16 | PP16×EP16 → TP32×DP8 | 97.6% | 2.39s-2.83s |
| H100 | QWEN3.5 397B | BF16 | TP8xPP8xEP32 -> TP16xDP16 | 97.4% | 1.97s-2.17s |
| GB200 | DSV3 | BF16 | PP16×EP16 → TP32×DP8 | 97.6% | 1.93s-2.59s |
| GB200 | Nemotron Ultra-v3 | BF16 | TP8xEP32xPP2 -> TP8xDP8 | 93.8% | 2.32s |

The feature supports both dense and MoE models. The table above shows the `XferDTensor fraction`, which is the proportion of the refit payload that utilizes the high-performance `bulk` transfer path. As the model size increases, this fraction becomes higher, which is the key to provide a scalable refit time to large models. For FP8 models, the efficiency is currently lower compared to BF16 models.
