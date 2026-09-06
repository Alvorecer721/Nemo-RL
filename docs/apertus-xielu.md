# xIELU on the Apertus stack: kernel, consumers, and the measurements behind the configuration

Apertus's activation is xIELU (arXiv:2411.13010). This page is the single reference for
how it executes on each side of the RL loop, where the CUDA kernel comes from, why
generation deliberately does **not** use it, and the measurement discipline that decision
came out of. The short version:

- **Training (Megatron workers): CUDA kernel, on purpose.** Eager execution is where a
  hand-fused kernel genuinely wins (measured ~8× over eager Python at decode shapes,
  near-roofline at 3.1 TB/s, with a hand-written backward).
- **Generation (vLLM): fused-Python, on purpose.** Inside a `torch.compile`d,
  CUDA-graphed engine, inductor's generated kernel and the hand-written one are a
  measured tie (0.24–0.26 ms/token in every configuration), so the kernel injection
  carried integration complexity for zero benefit — and was removed.
- **Consistency is a measured contract, not an implementation-matching aesthetic.** The
  Generation KL Error gauge reads 0.0003 in matched and mixed implementation
  configurations alike, and implementation matching has previously *masked* a real bug
  (see the traps page).

## Provenance

The kernel is **`nathanrchn/kernels`** (Nathan Ranchin, ETH), vendored as the
`3rdparty/kernels` submodule via the fork `Alvorecer721/kernels`. It is **not**
`nickjbrowning/XIELU` — that project merely defines the interface vLLM's activation
layer probes for (`import xielu.ops`, then `torch.classes.xielu.XIELU()`), and vLLM's
log messages name it. Our fork's `XIELUInference` binding implements that same
interface on top of Nathan's kernel; no nickjbrowning code is involved anywhere.

Fork deltas vs upstream (kept deliberately small): the forward-path `.contiguous()`
correctness fix (upstream's vectorized `uint4` loads index raw memory linearly — a
non-contiguous input silently reads wrong elements), the Autograd-key registration,
and the inference torchbind class. The first is PR-ready upstream; the binding is
parked until something consumes it.

## The two consumers

One `.cu` kernel, two entry points, opposite verdicts:

| | Training (Megatron) | Generation (vLLM) |
|---|---|---|
| Import path | `from xielu import xielu` (autograd op) in the bridge's `xielu_activation.py` | none — workers' inherited `PYTHONPATH` is explicitly blanked (they inherit the driver env, which carries `XIELU_SITE` for training) |
| Execution regime | **eager** — no compiler in the loop | `torch.compile` + CUDA graphs |
| Kernel's value there | real: one fused fwd kernel vs ~8 eager launches, hand-written bwd | none: inductor fuses the Python version to the same cost |
| Fallback if kernel absent | `@jit_fuser` compiled-Python (same math) | vLLM's built-in Python xIELU (same math) |
| Delivery | `XIELU_SITE` on the driver/worker `PYTHONPATH` (all launchers) | — |

The launcher variable `XIELU_SITE` therefore **feeds training only**. Removing the
generation injection took two moves, not one: deleting the injecting `env_vars` line
AND explicitly blanking the vLLM workers' `PYTHONPATH` — they inherit the driver
environment, so the training-side site otherwise leaks back in (and did, reproducing
the original compile crash the moment the safety plugin was gone). The variable, the
wheelhouse, and the fork all stay.

## The measurements

All numbers from paired same-node A/Bs (jobs 3045293/3045454, nid007126, 15 steps per
arm, fresh compiles, per-arm kernel-presence attested in-log) plus an isolated
microbenchmark; per-token figures are generation-phase seconds normalized by generated
tokens, stall-steps trimmed.

**End-to-end, steady state (ms/token, batch 32, TP2×2 engines):**

| | python | CUDA kernel |
|---|---|---|
| capstor-served image | 0.247 | 0.246 |
| iopsstor-served image | 0.243 | 0.241 |

A tie, replicated across image tiers. (The ~2% row difference is the image-tier
footnote in the Slurm README — small, consistent, irrelevant at step scale.)

**Isolated op (bf16, GH200, `[16,1,21504]` decode / `[4096,1,21504]` prefill):**

| Mode | decode | prefill |
|---|---|---|
| eager Python | 112 µs | 1984 µs |
| compiled Python (standalone) | 38 µs | 176 µs |
| CUDA kernel | 14 µs | 112 µs (3.1 TB/s) |

The kernel is genuinely 1.6–2.6× faster than standalone-compiled Python and ~8× faster
than eager Python. Both facts coexist with the end-to-end tie because inside the
compiled engine the activation is fused into neighboring kernels (≈ zero marginal
cost) and CUDA graphs erase the launch overheads the kernel exists to avoid — while
the kernel, as an opaque custom op, cannot participate in either. Its superiority is
an *eager-regime* property — which is exactly why training keeps it. Other eager
regimes (`enforce_eager=True` serving, HF-transformers inference, debugging) would
benefit in the same direction; no end-to-end figure is claimed for them because none
was measured and nothing we run is eager on the inference side.

**The retracted "+27% generation throughput"** (168→224 tok/s): both runs behind that
claim generated at the same true speed (0.247 vs 0.259 ms/token). The quoted numbers
were vLLM's windowed throughput snapshots, which divide tokens by wall-clock windows
that include idle/training time — in a colocated loop with a ~26% generation duty
cycle they under-report burst speed (~4,000 tok/s aggregate) by up to an order of
magnitude and mostly measure window alignment. The 27% delta was one job compiling
fresh (119 s first step) vs one loading a compile cache (77 s): a *startup-warmth*
difference laundered into a throughput claim by the metric. Never compare tok/s
snapshot lines across runs; use the per-step `generation:` phase timers.

## Operational notes

- **Builds are ABI-bound to (python, torch, CUDA, arch) — not to vLLM.** One build per
  container generation via `infra/slurm/cscs/build_xielu.slurm`, published as
  `MLLM/wheelhouse/aarch64/xielu-site-<ver>-<abi>` with the `xielu-site-current`
  symlink the launchers follow.
- **Compile-cache hygiene**: vLLM's compile-cache key does not record kernel presence;
  graphs traced under one activation implementation are silently loaded by runs using
  the other. With generation permanently kernel-free this trap is dormant, but purge
  `~/.cache/vllm*/torch_compile_cache` if the kernel is ever re-injected (details on
  the traps page).
- **Why removal also helps correctness observability**: vLLM's kernel path captures
  `beta`/`eps` at `__init__`, while the Python path reads the live engine-owned,
  non-persistent buffers. The latter keeps unintended mutation of those constants
  visible to the KL gauge (traps page, item 1).
- **Re-enabling** (should a future measured regime justify it): the kernel-free
  default lives as `policy.generation.vllm_cfg.env_vars.PYTHONPATH: ""` in the recipe
  family root (`grpo-apertus1p5-8b-1n4g-megatron-probe.yaml`), so it holds on every
  launcher; override it per-run by exporting `VLLM_XIELU_SITE=<site>` to the fixgate
  (what `bench/xielu_ab.slurm`'s kernel arm does), purge compile caches, and re-run
  the paired same-node A/B before believing any number.
