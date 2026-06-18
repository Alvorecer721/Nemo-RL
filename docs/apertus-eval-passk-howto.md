# Apertus pass@8 + Verifier Evaluation — How-To

Self-contained runbook for an agent that has an Apertus checkpoint and wants to
**measure failure-mode rates** on the same 10K-prompt blend the DPO pilot
trained against. Outputs are directly comparable to the SFT baseline numbers
quoted in `apertus-dpo-pilot-data.md`, so before/after deltas (e.g. "did DPO
fix the doom-loop rate?") are one diff away.

This doc is meant to be read end-to-end by an agent with no prior context.
If you're a human, jump to **TL;DR** at the bottom.

---

## What this evaluates

For a given Apertus checkpoint (the SFT base, or any DPO/GRPO descendant):

| Metric | What it measures | SFT baseline (from `apertus-dpo-pilot-data.md`) |
|---|---|---|
| `thinking_emitted` | model uses the `<|inner_prefix|>` block at all | 86.2% |
| `thinking_closed` | structurally well-formed inner block | 62.1% (37.9% violate chat template) |
| `is_correct` (format) | closed thinking + stop + TTR ≥ 0.20 + 5-gram < 10 | 47.6% |
| `finish_reason == "length"` | hit the 8K cap (runaway) | 16.9% |
| `content_verifiable` ∧ `content_correct` (per-agent) | answer passes the hard rule-based verifier | 10.7% of verifiable rows; 19.2% IFEval, 14.4% mcqa, 11.7% structured_outputs, 3.5% code_gen, 2.0% tool_use |

Per-prompt aggregations (`pass@1`, `pass@8`, "any-doom rate" etc.) all derive
from these per-row flags.

---

## Prerequisites — verify before you start

You need to be running **inside this repo** on a CSCS Clariden GH200 cluster
with a Slurm allocation you can submit from.

```bash
# 1. You are in the repo root (working tree of NVIDIA-NeMo/RL fork @ main).
ls examples/run_grpo.py 3rdparty/Megatron-LM-workspace/Megatron-LM    # both should exist

# 2. Submodules are initialised. If any of these are empty, run
#    `git submodule update --init --recursive` first — the rollouts script
#    needs Megatron-Bridge on PYTHONPATH.
for d in 3rdparty/Megatron-LM-workspace/Megatron-LM \
         3rdparty/Megatron-Bridge-workspace/Megatron-Bridge \
         3rdparty/Gym-workspace/Gym \
         3rdparty/Automodel-workspace/Automodel; do
  [ -n "$(ls -A $d 2>/dev/null)" ] && echo "OK   $d" || echo "MISS $d"
done

# 3. The blend's 10K-prompt JSONL exists (this is the stratified sample we use
#    everywhere — same 10K each time so the evals are comparable).
ls -la logs/rollouts_blend_apertus_think_8k.jsonl
#  -rw-r-----+ ... 234M ... logs/rollouts_blend_apertus_think_8k.jsonl   ← target

# 4. Source-row HF cache is present (used by the rollout script to look up
#    `tools=` per prompt — without this, tool-use prompts render with
#    "Tool Capabilities: disabled" and you get 0% tool-call emission).
ls /iopsstor/scratch/cscs/nathanrchn/.cache/huggingface/hub/datasets--nvidia--Nemotron-RL-Ultra-Training-Blends/snapshots/*/ifbench.jsonl

# 5. Apertus tokenizer (canonical, tools-fixed snapshot).
ls /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed_tools_fixed/chat_template.jinja
```

If you fail any check, fix it before submitting jobs. Step 3 missing → re-run
the single-shot blend rollout from `apertus-dpo-pilot-data.md` § Stage 1. Step
4 missing → the tool-index build will simply skip those subsets; not fatal,
but tool-use prompts will render without tools.

> **Canonical Apertus tokenizer:** always point `policy.tokenizer.name` /
> `AutoTokenizer.from_pretrained(...)` at
> `/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed_tools_fixed`.
> Some files in this repo still hard-code the older `…snapshot-20260611` path
> — see the banner in `apertus-dpo-pilot-data.md` for the migration list.

---

## Pick which checkpoint to evaluate

The default in every rollout script is the SFT base:

```
/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200
```

To eval a different checkpoint, you must override `--model`. The path can be:

- A **HuggingFace-format directory** (a dir containing `config.json` +
  `model.safetensors*`). Most exported checkpoints look like this.
- A **single .safetensors file** is *not* supported by vLLM. You need the dir.

For a **DPO-trained checkpoint** produced by this repo (e.g.
`results/dpo-apertus-pilot/step_300/`), the checkpoint is in **Megatron
distributed format** (`iter_0000300/`-style sharding), not HuggingFace.
You have to **convert it to HF format first** using
`nemo_rl.models.megatron.converters` (see `examples/converters/` in the repo —
or use the dump_hf path inside the DPO script if you saved with `dump_hf=True`).
Once you have an HF dir, the eval below treats it identically to the SFT base.

> Concrete migration recipe (when you have a DPO Megatron checkpoint):
> 1. Run `uv run --locked --extra mcore python -m nemo_rl.models.megatron.converters.megatron_to_hf --in=<megatron-ckpt> --out=<hf-dir> --model=apertus`
> 2. Verify `<hf-dir>/config.json` is readable.
> 3. Pass `MODEL=<hf-dir>` in the env when sbatching the eval job (see below).
> If your DPO script saved a "policy" HF snapshot (`results/.../hf/`) directly,
> step 1 is unnecessary — point `MODEL` at that dir.

---

## Run pass@8 (Stage A)

**Goal:** produce per-row format/structural flags for 10,000 prompts × 8 samples
on your checkpoint.

### Submit

```bash
# From the repo root, on a CSCS compute node (or login node — see incantation
# notes below for the difference).

# Optional override — defaults are fine for the SFT base.
MODEL=${MODEL:-/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200}
TOKENIZER=${TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed_tools_fixed}
OUT_BASE=${OUT_BASE:-logs/rollouts_passk_apertus_eval}   # change this for each new eval

# Submission incantation when sitting on a compute node (the dev session uses
# this — the env-unsets strip the inherited Pyxis container plugin so sbatch
# from inside CE doesn't choke; harmless from a login node).
env -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts \
    LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:$LD_LIBRARY_PATH \
    OUT_BASE=$OUT_BASE \
    sbatch infra/slurm/cscs/rollouts_blend_passk_apertus_10k.slurm
```

The slurm script handles 4 nodes × TP=4 (16 GH200), stratified-samples 10,000
prompts from `logs/rollouts_blend_apertus_think_8k.jsonl` (deterministic seed
42), runs vLLM `n=8` per prompt at `temperature=1.0 top_p=1.0 max_tokens=8192`
with `enable_thinking=True`, threads `tools=` from the source rows so tool-use
prompts actually exercise their tool defs, and concatenates the 4 per-rank
JSONLs into one combined file at the end.

> **Replace the model path:** the slurm wrapper *currently hardcodes* the SFT
> base via the `DEFAULT_MODEL` constant in
> `nemo_rl_apertus/rollouts_blend_passk_offline.py`. To eval a different
> checkpoint, either (a) override the `--model` CLI arg by editing the slurm
> bash (the relevant line is the `uv run --locked --extra vllm python -m
> nemo_rl_apertus.rollouts_blend_passk_offline ... \` block — add
> `--model $MODEL`), or (b) export `MODEL=` and patch the script to honour it.
> The safer one-shot is just (a).

### Wait

Expected wall time on 4 GH200 nodes with TP=4: **~1h 20min** (~83 min) for
80,000 rollouts. The reservation `SD-69241-apertus-1-5-0` (if active for you)
short-circuits the queue. Without it, queue waits on `normal` can be hours.

```bash
# Monitor: substitute the JOBID returned by sbatch.
squeue -j JOBID -o "%i %P %j %T %M %D %R %S"
tail -F logs/rollouts_blend_passk_apertus_10k_JOBID.err
```

### Inspect

```bash
# Output: 80,000 rows.
wc -l logs/rollouts_passk_apertus_eval.jsonl
ls -la logs/rollouts_passk_apertus_eval*.jsonl

# Per-row schema:
head -1 logs/rollouts_passk_apertus_eval.jsonl | python3 -m json.tool
# fields: prompt_idx, sample_idx, config, prompt, generation_text,
#         generation_text_full, n_output_tokens, finish_reason,
#         thinking_emitted, thinking_closed, ttr, top_5gram_count, is_correct
```

`is_correct` here is **format only** (closed thinking + stop + TTR ≥ 0.20 +
5-gram < 10). It does **not** know whether the answer is correct.

---

## Run hard-verifier sweep (Stage B)

**Goal:** add `content_correct` per row for the ~26% of prompts that have a
rule-based verifier (IFEval / mcqa / structured_outputs / code_gen /
tool_use). LLM-judge agents are intentionally skipped.

### Submit

```bash
# The verifier slurm reads PASSK_JSONL and writes OUT_JSONL.
env -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts \
    LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:$LD_LIBRARY_PATH \
    PASSK_JSONL=logs/rollouts_passk_apertus_eval.jsonl \
    OUT_JSONL=logs/rollouts_passk_apertus_eval_verified.jsonl \
    sbatch infra/slurm/cscs/verify_passk_apertus.slurm
```

This is a **1 node CPU-bound** job (code_gen subprocess work is the long pole).
Wall time on 80K rows: **~25 min** in the `debug` partition (90 node-minute
QoS, fits comfortably). First run on a fresh node will also install the
verifier deps (`verifiable_instructions` from Google's IFEval repo,
`jsonschema`, `pyyaml`) into a persistent overlay dir at
`/iopsstor/scratch/cscs/nathanrchn/.cache/python_verify_deps` — subsequent
runs reuse the install.

### Inspect

```bash
# Headline summary is printed at the bottom of the .out log:
grep -A 20 "SUMMARY" logs/verify_passk_apertus_JOBID.out

# Schema additions (the per-row record now also carries):
head -1 logs/rollouts_passk_apertus_eval_verified.jsonl | python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
for k in ("agent_ref", "content_verifiable", "content_correct", "verifier_reason"):
    print(f"  {k}: {r.get(k)!r}")
'
```

Expected summary shape (numbers from the SFT baseline on the 10K sample):

```
verifiable rows : 20624 / 80000  (25.8%)
content_correct :  2215 / 20624  (10.7%)

Per-agent (sorted by verified count):
  instruction_following_simple_agent     verified 7960/7960  correct 1528/7960 (19.2%)
  single_step_tool_use_with_argument...   verified 3904/5744  correct   78/3904 ( 2.0%)
  code_gen_simple_agent                   verified 3696/3696  correct  129/3696 ( 3.5%)
  mcqa_simple_agent                       verified 1984/1984  correct  286/1984 (14.4%)
  ... etc
```

---

## Produce the headline numbers (Stage C — no Slurm)

The two output files above are everything you need. Run these on a compute
node (the analyses are pure-Python, CPU-only).

> **Python version footgun**: the compute-node system `/usr/bin/python3` is
> 3.6.15. Most of the analysis scripts here use stdlib only and run on 3.6,
> but anything with `from __future__ import annotations` (some of the heavier
> ad-hoc scripts) needs 3.7+. If a script errors with `future feature
> annotations is not defined`, run it from inside the container instead:
> `uv run --locked --extra vllm python3 ...`. The Slurm scripts already do
> this.

### Top-line format/structural metrics

```bash
python3 << 'PY'
import json
from collections import Counter
n=0; closed=0; emit=0; iscor=0; tok_sum=0; fr=Counter()
for line in open("logs/rollouts_passk_apertus_eval.jsonl"):
    if not line.strip(): continue
    r = json.loads(line)
    n += 1
    if r.get("thinking_emitted"): emit+=1
    if r.get("thinking_closed"): closed+=1
    if r.get("is_correct"): iscor+=1
    tok_sum += r.get("n_output_tokens",0)
    fr[r.get("finish_reason","?")] += 1
print(f"n                 : {n}")
print(f"thinking_emitted  : {emit}/{n} ({emit/n*100:.1f}%)")
print(f"thinking_closed   : {closed}/{n} ({closed/n*100:.1f}%)  (target: ≥95% post-DPO)")
print(f"is_correct format : {iscor}/{n} ({iscor/n*100:.1f}%)")
print(f"finish=length     : {fr.get('length',0)}/{n} ({fr.get('length',0)/n*100:.1f}%)  (target: <5%)")
print(f"mean output toks  : {tok_sum/n:.0f}")
PY
```

### Per-agent content correctness

```bash
python3 << 'PY'
import json
from collections import defaultdict
agg = defaultdict(lambda: {"n":0, "v":0, "c":0})
for line in open("logs/rollouts_passk_apertus_eval_verified.jsonl"):
    if not line.strip(): continue
    r = json.loads(line)
    a = r.get("agent_ref") or "<unmatched>"
    agg[a]["n"] += 1
    if r.get("content_verifiable"):
        agg[a]["v"] += 1
        if r.get("content_correct"): agg[a]["c"] += 1
total_v = sum(d["v"] for d in agg.values())
total_c = sum(d["c"] for d in agg.values())
print(f"verifiable rows : {total_v} / {sum(d['n'] for d in agg.values())} ({total_v/sum(d['n'] for d in agg.values())*100:.1f}%)")
print(f"content_correct : {total_c} / {total_v} ({total_c/max(1,total_v)*100:.1f}%)")
print()
rows = [(a, d["n"], d["v"], d["c"]) for a,d in agg.items() if d["v"] > 0]
rows.sort(key=lambda r: -r[2])
print(f"{'agent':<60} {'n':>6} {'verified':>9} {'correct':>9} {'%':>6}")
for a,n,v,c in rows:
    print(f"{a:<60} {n:>6} {v:>9} {c:>9} {c/max(1,v)*100:>5.1f}%")
PY
```

### Doom-loop classification (matches the morning report bucketing)

```bash
python3 << 'PY'
import json, re
from collections import Counter
n=0; severe=0; degen=0; pattern=0; cjk_drift=0
CJK = re.compile(r"[一-鿿]")
for line in open("logs/rollouts_passk_apertus_eval.jsonl"):
    if not line.strip(): continue
    r = json.loads(line)
    n += 1
    ttr = r["ttr"]; top5 = r["top_5gram_count"]; ntok = r["n_output_tokens"]
    if ttr < 0.10 and ntok >= 50: severe += 1
    elif 0.10 <= ttr < 0.20 and ntok >= 50: degen += 1
    if top5 >= 10: pattern += 1
    g = r["generation_text"]; p = r["prompt"]
    if CJK.search(g) and not CJK.search(p): cjk_drift += 1
print(f"SEVERE doom (TTR<0.10):   {severe:>5} ({severe/n*100:5.2f}%)")
print(f"DEGENERATE (0.10≤TTR<0.20): {degen:>5} ({degen/n*100:5.2f}%)")
print(f"PATTERN-LOCK (top 5g≥10×): {pattern:>5} ({pattern/n*100:5.2f}%)")
print(f"CJK DRIFT (non-zh→zh):     {cjk_drift:>5} ({cjk_drift/n*100:5.2f}%)")
PY
```

### Per-prompt pass@8

```bash
python3 << 'PY'
import json
from collections import defaultdict
by_prompt = defaultdict(list)
for line in open("logs/rollouts_passk_apertus_eval_verified.jsonl"):
    if not line.strip(): continue
    r = json.loads(line)
    if r.get("content_verifiable"):
        by_prompt[r["prompt_idx"]].append(bool(r.get("content_correct")))
verifiable = {p: s for p,s in by_prompt.items() if s}
n = len(verifiable)
p1 = sum(1 for p,s in verifiable.items() if s[0]) / n
p8 = sum(1 for p,s in verifiable.items() if any(s)) / n
allp = sum(1 for p,s in verifiable.items() if all(s)) / n
allf = sum(1 for p,s in verifiable.items() if not any(s)) / n
mix = 1 - allp - allf
print(f"verifiable prompts : {n}")
print(f"pass@1   : {p1*100:.2f}%")
print(f"pass@8   : {p8*100:.2f}%")
print(f"all-pass : {allp*100:.2f}%   ← model already nails")
print(f"all-fail : {allf*100:.2f}%   ← model can't solve in 8 tries")
print(f"MIXED    : {mix*100:.2f}%   ← preference-learning gold")
PY
```

---

## Compare two checkpoints (before/after DPO)

This is the actual reason you ran the eval. Two output files:

```
logs/rollouts_passk_apertus_eval_BASE_verified.jsonl     ← from the SFT base run
logs/rollouts_passk_apertus_eval_DPO_verified.jsonl      ← from your new checkpoint
```

Compute per-metric deltas:

```bash
python3 << 'PY'
import json

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    closed = sum(1 for r in rows if r.get("thinking_closed")) / n
    is_correct = sum(1 for r in rows if r.get("is_correct")) / n
    fr_length = sum(1 for r in rows if r.get("finish_reason") == "length") / n
    # Content-correct rate over verifiable
    v = [r for r in rows if r.get("content_verifiable")]
    cc = sum(1 for r in v if r.get("content_correct")) / max(1, len(v))
    return dict(n=n, closed=closed, is_correct=is_correct, fr_length=fr_length, cc=cc, nv=len(v))

base = load("logs/rollouts_passk_apertus_eval_BASE_verified.jsonl")
dpo  = load("logs/rollouts_passk_apertus_eval_DPO_verified.jsonl")

def fmt(b, d, name, lo_is_good=False):
    delta = d - b
    sign = "▲" if delta > 0 else ("▼" if delta < 0 else "·")
    good = (delta < 0 if lo_is_good else delta > 0)
    color = "↑↑" if good and abs(delta) > 0.01 else ("↓" if not good and abs(delta) > 0.01 else "")
    print(f"  {name:<28} base {b*100:6.2f}%  →  DPO {d*100:6.2f}%  ({sign}{abs(delta)*100:+.2f}pp) {color}")

print(f"BASE n={base['n']} (verifiable {base['nv']}), DPO n={dpo['n']} (verifiable {dpo['nv']})")
print()
fmt(base["closed"],     dpo["closed"],     "thinking_closed",            lo_is_good=False)
fmt(base["is_correct"], dpo["is_correct"], "is_correct (format)",        lo_is_good=False)
fmt(base["fr_length"],  dpo["fr_length"],  "finish=length (runaway)",    lo_is_good=True)
fmt(base["cc"],         dpo["cc"],         "content_correct (verified)", lo_is_good=False)
PY
```

For the doom-loop fix to be considered a win, you want:

- `thinking_closed`: **up** from 62% → ideally >95%
- `is_correct` format: **up** from 48% → ideally >80%
- `finish=length`: **down** from 17% → ideally <5%
- `content_correct`: **flat or up** — DPO can hurt content quality if the data
  was format-biased; if this drops sharply, you over-trained.

---

## Known pitfalls (the agent that wrote this hit them)

1. **`tools=` was missing in v1.** The first `rollouts_blend_passk_offline.py`
   didn't pass `tools=` to `apply_chat_template`. Tool-use prompts then
   rendered with `Tool Capabilities: disabled` and the model emitted **zero**
   `<|tools_prefix|>` blocks → 0% tool-use content correctness, falsely.
   The fix is upstream now; if you start from a fresh fork, confirm the
   normalisation+thread-through is present near the `apply_chat_template`
   call.

2. **Trap #3 (double BOS).** Apertus's chat template emits `{{ bos_token }}`
   on line 142. If you push rendered text through any path that re-tokenises
   with `add_special_tokens=True` (vLLM's `vllm_content` flow, some Gym
   re-encode paths), you get double BOS and silently catastrophic logprob
   comparisons. The rollout flow used here pushes **token ids**, not text, so
   this trap doesn't fire — but if you're inspired to wrap the eval into a
   Gym agent, re-read `apertus-traps-and-invariants.md` § 3.

3. **`debug` partition QoS limit.** Total **node-minutes ≤ 90 per job**. The
   verifier sweep at 1 node × 60 min fits; the rollout sweep at 4 nodes ×
   22 min fits a smaller pass (1280 prompts) but cannot fit the full 10K
   (4 nodes × 80 min ≫ 90 node-min). Use `normal` partition for the rollout
   sweep, optionally with the reservation `SD-69241-apertus-1-5-0` to skip
   the queue.

4. **OOM on TP=2 with seq=8K.** Apertus's vocab-parallel logits exp() needs
   ~8.7 GB per GH200 at `seq=8192, TP=2` — blows through the 95 GB GH200. The
   pass@8 rollout script uses **TP=4**, which quarters this to ~2.2 GB.
   Don't drop TP unless you also drop the sequence length.

5. **CSCS compute node Python is 3.6.15.** Most ad-hoc analyses in this doc
   are stdlib-only and run on 3.6; any script using `from __future__ import
   annotations` or `tuple[…]` type syntax (PEP 585) will choke. Run those
   inside the container: `uv run --locked --extra vllm python3 ...`.

6. **Verifier deps install.** The first time the verifier slurm runs on a new
   compute-side overlay, it does `pip install --target=$DEPS_DIR
   verifiable_instructions @ git+https://github.com/abukharin-nv/verifiable-instructions.git
   jsonschema pyyaml` — adds ~1-2 min cold start. Subsequent jobs cache-hit.

7. **Tool-use is fundamentally hard for Apertus 1.5 SFT.** Even with the
   `tools=` fix, only ~2% of tool-use rollouts produce a content-correct
   call (right tool name + arg match). Don't read a 0% tool-use rate post-DPO
   as a regression unless you compare against the corrected baseline (2-4%,
   not 0%).

---

## Submission incantation quick-reference

From a compute node (inside a CSCS CE container — most agent sessions
are here):

```bash
env -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts \
    LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:$LD_LIBRARY_PATH \
    sbatch [--reservation=SD-69241-apertus-1-5-0] <script>
```

From a login node: plain `sbatch <script>` (the `env -u` and
`LD_LIBRARY_PATH` are harmless no-ops, but you don't need them).

---

## TL;DR

```bash
# 0. Confirm prereqs (above).
#    Set MODEL to your checkpoint dir (HF format). If none, defaults to SFT base.

# 1. Pass@8 — 4 nodes × ~80 min on normal partition.
env -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts \
    LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:$LD_LIBRARY_PATH \
    OUT_BASE=logs/rollouts_passk_eval \
    sbatch --reservation=SD-69241-apertus-1-5-0 \
           infra/slurm/cscs/rollouts_blend_passk_apertus_10k.slurm

# 2. Wait for it. Then verify — 1 node × ~25 min on debug.
env -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_writable \
    -u SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_container_mounts \
    LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:$LD_LIBRARY_PATH \
    PASSK_JSONL=logs/rollouts_passk_eval.jsonl \
    OUT_JSONL=logs/rollouts_passk_eval_verified.jsonl \
    sbatch infra/slurm/cscs/verify_passk_apertus.slurm

# 3. Print the headline metrics (see "Produce the headline numbers" above).

# 4. If you also have a BASE-verified file, run the before/after diff cell.
```

Outputs are deterministic given the same seeds, so two checkpoints scored
this way are directly comparable.
