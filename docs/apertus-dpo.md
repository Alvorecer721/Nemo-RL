# DPO on Apertus: offline & online (CSCS GH200)

Direct Preference Optimization for Apertus 1.5 8B, in two flavors: **offline DPO** (static preference pairs) and **online DPO** (policy rollouts ranked on the fly by an LLM-as-judge). Both reuse the stock NeMo-RL `DPOLossFn`; online DPO adds the judge + a GRPO-style generation loop. For the runtime/image setup and certification gates see [apertus-quickstart.md](apertus-quickstart.md) and [apertus-traps-and-invariants.md](apertus-traps-and-invariants.md); for Slurm submission knobs see [infra/slurm/cscs/README.md](../infra/slurm/cscs/README.md).

## How it works

**DPO loss (both flavors).** `L = w_p·(−log σ(β·(r_chosen − r_rejected))) + w_sft·L_sft`, where the implicit reward `r = Σ_t (log π_θ − log π_ref)` is summed over the response tokens (length-normalized when `preference_average_log_probs: true`) and `β = reference_policy_kl_penalty`. The reference π_ref is a frozen CPU snapshot of the policy. (MPO — `examples/run_mpo_apertus.py`, `nemo_rl_apertus/mpo_loss.py` — is the same machinery plus a BCO quality term.)

**Offline DPO** (`examples/run_dpo_apertus.py` — registers the tools/thinking processor + runtime guard; stock `examples/run_dpo.py` also works for plain `chosen`/`rejected` data): a static dataset supplies `chosen`/`rejected` per prompt; each step computes the DPO loss on those fixed pairs. The reference is frozen for the whole run.

**Online DPO** (`examples/run_online_dpo_apertus.py`, driver `nemo_rl_apertus/online_dpo.py`): there is **no** preference dataset — pairs are manufactured each step:

1. repeat each prompt `R` times (`grpo.num_generations_per_prompt`) and generate `R` rollouts with the **current** policy (GRPO machinery, reused unchanged);
2. score every rollout with the **judge** (an environment that returns a scalar reward per rollout);
3. per prompt: **chosen** = argmax score, **rejected** = argmin (a group whose `max−min ≤ tie_eps`, or with a missing assistant turn, is *degenerate* → masked out; rollouts that hit the token cap (`truncated`) are masked too **by default** — set `online_dpo.train_on_truncated: true` to instead keep them, so a low judge score makes a truncated/incomplete rollout the *rejected*);
4. apply the stock `DPOLossFn` on the on-the-fly pairs;
5. optionally refresh the reference to the current policy every `reference_update_freq` steps (`-1`/`0` = frozen = stock DPO).

`online_dpo.setup` calls `grpo.setup` verbatim (which already builds the policy + reference model, the vLLM engine, the clusters, and the prompt dataloader) and swaps the loss to `DPOLossFn` — only the loss differs from GRPO.

## Recipes, launchers & probes

| Recipe (`examples/configs/recipes/llm/`) | Algo | Data | What it does |
|---|---|---|---|
| `dpo-apertus1p5-8b-maxmin-megatron.yaml` | offline DPO | MaxMin binary-preference parquet (`chosen`/`rejected`) | reference-faithful full-set DPO (β 25, length-normalized, lr 1e-6), validation every 200 steps |
| `probe-dpo-apertus1p5-8b-toolthinking-1n4g-megatron.yaml` | offline DPO | synthetic tools/thinking matrix (generated) | 1-node / 4-GPU smoke of `ToolThinkingPreferenceProcessor` — thinking × tools × multi/single-turn; data from `tools/build_offline_dpo_apertus_testset.py` |
| `probe-online-dpo-apertus1p5-8b-1n4g-megatron.yaml` | online DPO | DeepScaler prompts (smoke) | 1-node / 4-GPU, 3-step smoke of the fused path; single-node judge; the default probe |
| `online-dpo-apertus1p5-8b-maxmin-megatron.yaml` | online DPO | MaxMin **prompt-only** parquet | online DPO on the reference prompt set via the additive `prompt_only` loader |

Launchers (`infra/slurm/cscs/`): **offline** → `submit_nemo_rl_dpo.slurm` (+ `submit_nemo_rl_dpo_multinode.slurm`), which run `run_dpo_apertus.py` with `RECIPE` defaulting to the MaxMin offline recipe; the tools/thinking smoke probe is `probe_dpo_toolthinking_apertus.slurm` (regenerates the synthetic set in a prologue, then runs the probe recipe and asserts the loss is finite). **Online** → a general engine `online_dpo_launcher.sh` (sets the judge + recipe from env, then sbatches the orchestrator, which brings up the judge, discovers its URL, health-checks it, then sbatches training) with **two thin presets** over it: `probe_online_dpo_1n_1judge.sh` (smoke: DeepScaler + single-node judge) and `launch_online_dpo_maxmin.sh` (reference MaxMin + Qwen judge). Lower-level pieces: `serve_judge.slurm`, `online_dpo_orchestrator.sh`, `submit_online_dpo.slurm`. See the [Slurm README](../infra/slurm/cscs/README.md) for the full knob list.

## Quick start

**Offline DPO** (1 node / 4 GPUs):

```bash
cd <repo> && mkdir -p logs
sbatch infra/slurm/cscs/submit_nemo_rl_dpo.slurm        # RECIPE defaults to dpo-...-maxmin-megatron.yaml
```

**Online DPO** (two-job: a judge server + the trainer):

```bash
cd <repo> && mkdir -p logs
# default smoke recipe (single-node judge):
JUDGE_SERVE_MODEL=/path/to/judge-model JUDGE_API_KEY=$KEY \
    infra/slurm/cscs/probe_online_dpo_1n_1judge.sh
# reference MaxMin prompt set:
JUDGE_SERVE_MODEL=/path/to/judge-model JUDGE_API_KEY=$KEY \
    RECIPE=examples/configs/recipes/llm/online-dpo-apertus1p5-8b-maxmin-megatron.yaml \
    infra/slurm/cscs/probe_online_dpo_1n_1judge.sh
# …or the MaxMin launcher with the reference's Qwen3.6-27B judge pre-pinned (just the key):
JUDGE_API_KEY=$KEY infra/slurm/cscs/launch_online_dpo_maxmin.sh
```

`launch_online_dpo_maxmin.sh` mirrors the SPIN reference `launch_online_50k.sh`: it pins the same judge model the reference uses (Qwen3.6-27B, served single-node at TP=4 by default; set `MODEL_LAUNCH_DIR` + `SERVER_WORKERS=8` to reproduce the reference's router-balanced 8-replica serving) and the MaxMin recipe, then **hands off (via `exec`) to the general `online_dpo_launcher.sh`** (the same engine the smoke probe uses) — i.e. it adds no launch logic of its own; it only pre-sets the Qwen judge + MaxMin config and lets that launcher do the actual work (sbatch the orchestrator → serve the judge → discover its URL → health-check → sbatch the trainer). The judge *methodology* already matches the reference exactly (UltraFeedback, `aspects=[helpfulness]`, `max_tokens=1`, `temperature=0`, `top_logprobs=20`, `enable_thinking=false`); training HP in the recipe are probe-scale (the reference ran bs=256, R=8, prompt 8192).

## The judge (online DPO)

The judge is served over an **OpenAI-compatible HTTP endpoint** (two-job architecture). The training job reaches it via `JUDGE_BASE_URL` / `JUDGE_MODEL` / `JUDGE_API_KEY`; it is wrapped as a Ray `JudgeEnvironment` actor (`nemo_rl_apertus/online_judge_env.py`) so it slots into the rollout like any reward environment.

### Configuration (`env.online_dpo_judge`)

```yaml
env:
  online_dpo_judge:
    type: ultrafeedback            # selects the Judge backend via build_judge()
    aspects: [helpfulness]         # subset of {instruction_following, honesty, truthfulness, helpfulness, thinking_appropriateness, thinking_formatting}; score = (weighted) mean over aspects
    max_concurrency: 512           # concurrent judge HTTP calls per batch
    # base_url / api_key / model resolve from JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL (or pin here)
    # system_prompt: "..."         # override the UltraFeedback judge system prompt
    # aspect_prompts: {helpfulness: "...{prompt}...{completion}..."}   # per-aspect rubric templates
    # aspect_weights: {helpfulness: 1.0, thinking_appropriateness: 0.3, thinking_formatting: 0.2}  # per-aspect weights for the cross-aspect mean (default: equal)
    # request_timeout: 60.0        # per-call HTTP timeout (s)
    # top_logprobs: 20             # first-token top-logprobs the score expectation reads
    # enable_thinking: false       # passed as chat_template_kwargs (e.g. disable Qwen thinking) — the JUDGE's own reasoning, not the policy's
```

**Scoring (`ultrafeedback`):** the judge emits a single `1`–`5` token; we read its first-token top-logprobs, softmax over `{"1".."5"}`, and take the expected value `Σ tokenᵢ·pᵢ ∈ [1, 5]`, averaged over the enabled aspects. All calls in a batch fire concurrently; an API/parse failure scores `0.0` (non-fatal — the prompt group degenerates and is masked). The cross-aspect mean is **equal-weight** by default; set **`aspect_weights`** (`{aspect: float}`; aspects omitted default to `1.0`) to make a shaping axis count for less than a primary one — handy when adding the reasoning axes below alongside `helpfulness`.

**Reasoning axes (for thinking models).** Two opt-in aspects judge the reasoning trace itself (in the Apertus template the trace is wrapped in the special thinking tokens `<|inner_prefix|>` … `<|inner_suffix|>`, or `<think>` … `</think>` with the model's bundled template — both rubrics point the judge at that span). They are normal aspects (rubrics in `DEFAULT_ASPECT_PROMPTS`, overridable via `aspect_prompts`). The rollout decodes the assistant content with `skip_special_tokens=True`, which would strip those delimiters, so when either aspect is enabled the entry point hands the judge env the **policy tokenizer** (`env.online_dpo_judge.completion_tokenizer`, set automatically) and the env re-decodes the scored completion with `skip_special_tokens=False` — the only extra plumbing, and it is automatic:

- **`thinking_appropriateness`** — is the reasoning the right *length for the problem's difficulty*, *straight / goal-oriented* (no padding, rambling, circular or backtracking steps), **and** does the *final answer faithfully follow from the reasoning* (no trace↔answer contradiction, work not ignored)? Judges calibration, quality, and reasoning→answer consistency — **not** whether the answer is objectively correct (the other aspects cover that).
- **`thinking_formatting`** — is the reasoning correctly *delimited*: a matching opening **and** closing thinking token present, the reasoning inside and the final answer outside? A response with no reasoning at all is treated as correctly formatted (nothing to delimit). Judges only the delimiting, not the content.

Pair them with the policy thinking toggle below (e.g. `mode: random`) so the model sees both reasoning and non-reasoning rollouts, and usually with sub-unity `aspect_weights` so they shape rather than dominate the reward. Both rubrics rank reasoning that **runs until truncation without ever producing an answer** (or never closes its thinking block) at the bottom; to turn that into an active training signal rather than a discard, set **`online_dpo.train_on_truncated: true`** (see *How it works* — by default truncated rollouts are masked from the loss).

### What the judge receives

For each rollout, `judge_inputs_from_conversation` splits the conversation at the **last assistant turn**:

- **completion** = that last assistant turn's content — the *exact* span the DPO loss trains (via `only_unmask_final`);
- **prompt** = everything before it (earlier user/assistant turns kept as context). Multi-turn safe.

### Plain vs chat-template-rendered prompt

By default the judge sees the rollout's prompt turns, whose `content` is the **chat-template-rendered** string (BOS, role tags, etc.). To hand the judge a **clean/plain** prompt instead, the data processor stashes the raw prompt in `extra_env_info["judge_prompt"]` (a `str` or `[{role, content}]` list), which `judge_inputs_from_conversation` prefers over the rendered turns:

- the **`online_prompt_processor`** (used by the `prompt_only` loader / MaxMin recipe) **sets** `judge_prompt` → the judge sees the clean prompt;
- the inherited **`math_hf_data_processor`** (DeepScaler probe) does **not** → the judge sees the rendered string (fallback).

`extra_env_info["judge_images"]` (optional) attaches image URLs/`data:` URIs for multimodal judging.

### Bring your own judge

Implement the `Judge` protocol in `nemo_rl_apertus/online_judge.py` — `score(prompts, completions, images=None) -> list[float]` (higher = better) — register it in `JUDGE_REGISTRY` under a `type` key, and select it with `env.online_dpo_judge.type`. `build_judge(cfg)` dispatches; the driver and `JudgeEnvironment` are judge-agnostic. (A pairwise or reward-model backend that needs all `R` completions together, or token ids, should extend the protocol signature — see the note in `online_judge.py`.)

## Datasets

### Offline preference data (and bringing your own)

Set `data.train.dataset_name` + `data_path`. Stock loaders: **`BinaryPreferenceDataset`** (rows with `chosen`/`rejected` message lists) and **`PreferenceDataset`** (`context` + `completions:[{rank, completion}]`, lower rank = preferred). The default processor is the stock `preference_preprocessor`; `setup_preference_data` now also honors a registry **`data.processor`** (a small flagged Apertus edit) and a dataset's `preprocessor`. So to add a new on-disk format, add a `RawDataset` subclass whose `format_data`/`preprocessor` normalizes your rows into one of those canonical schemas, or register a custom processor and select it via `data.processor`.

For a richer or **pretokenized** processor, `nemo_rl_apertus/data_processors.py` provides a `PreferenceDataProcessor` ABC (Template Method: `parse` → `build_message_log` → `_assemble`) with two concretes — `RankedCompletionsPreferenceProcessor` (text, parity with the stock processor) and `PretokenizedPreferenceProcessor` (consumes per-turn `token_ids`, no tokenizer call). Register an instance with `.register("name")` and select it via `data.processor`.

**Tool calls & thinking (offline).** Use **`ToolThinkingPreferenceProcessor`** (`data.processor: apertus_tool_thinking_preference`) via the **`examples/run_dpo_apertus.py`** entry, which registers it (`register_offline_dpo_processors()`). The `probe_dpo_toolthinking_apertus.slurm` launcher exercises this path; `submit_nemo_rl_dpo.slurm` defaults to the MaxMin recipe, which uses the stock `BinaryPreferenceDataset` processor instead. It renders the **whole conversation** once with `apply_chat_template` so the developer block is correct, then splits into `[prompt, response]` at the final assistant turn.

- **Detection** — scanned across **both** chosen and rejected (so the pair shares one identical prompt/developer block). *Tools:* schemas are **not** inferred — you supply them in `datum["tools"]` (list or JSON string; see *Row format* below); tool *usage* (a `tool` turn, a `tool_calls` field, or a `tool_calls` content block) is auto-detected only as a **guard** — usage with no `datum["tools"]` raises. *Thinking:* `enable_thinking` = `datum["enable_thinking"]` if set, else auto-detected from a `thoughts` content block (inline `<|inner_prefix|>` strings baked into plain text are **not** auto-detected — set the flag for those).
- **Developer-block assembly** (by the Apertus template): a system message is auto-injected if absent (the default *"You are Apertus 1.5 Omni …"* Omni prompt) — set **`policy.tokenizer.disable_default_system_prompt: true`** to emit an empty `<|system_start|><|system_end|>` block instead (it prepends an explicit empty system turn when none is present; conversations that already carry a system turn are untouched). This applies across all data paths + rollout prompts; the MaxMin offline/online and tools/thinking recipes enable it. The developer block always renders `Deliberation: enabled/disabled` (from `enable_thinking`) and `Tool Capabilities:` + definitions / disabled (from `datum["tools"]`). The same `tools` + `enable_thinking` feed both renders, so chosen/rejected differ only in the final response. Schemas must be complete OpenAI function specs (`name` + `description` + typed `parameters`) or `apply_chat_template` itself raises.
- **Tokenization:** `add_special_tokens=False` — the template emits its own BOS (**exactly one**, no `<s><s>`) and `<|assistant_end|>` is the terminator (**no spurious `</s>`**); the **prompt is verified to be an exact token-prefix** of the full render, so nothing is duplicated/dropped at the boundary.
- **Scope:** the completion must be a single final assistant turn (which may bundle `tool_calls`/thinking blocks); a multi-turn tool **loop** across separate messages fuses into one assistant block (no clean prompt/response split) → it raises — use `PretokenizedPreferenceProcessor` for those.

**Row format (data requirements).** Base schema is `PreferenceDataset` — `{"context": [turns], "completions": [{"rank", "completion": [turns]}, {…}]}` (lower rank = chosen; both completions share `context`). Tool/thinking specifics:

- **`datum["tools"]`** — complete OpenAI function specs, as a **list or a JSON string** (use the string for parquet/JSONL: Arrow can't unify varied `parameters` schemas across rows). Per tool: `{"type":"function","function":{"name","description","parameters": <JSON-Schema with "type"/"properties"/"required">}}` — `description` + typed `parameters` are mandatory or `apply_chat_template` raises.
- **In-turn tool use** — an assistant turn calls a tool via a message-level `"tool_calls": [{"type":"function","function":{"name","arguments": "<JSON string>"}}]`; a tool result is a `{"role":"tool","content":"…"}` turn.
- **Keep every turn's `content` a string** (use inline `<|inner_prefix|>…<|inner_suffix|>` for thinking) so a single JSONL/parquet stays Arrow-loadable (mixing string and `{blocks}` content in one file breaks the loader).
- **`datum["enable_thinking"]`** — `true`/`false`; or omit it and express thinking as a `{"type":"thoughts"}` content block (auto-detected). Inline `<|inner_prefix|>` strings are **not** auto-detected.
- **Final completion** — exactly one assistant turn (see *Scope* above): a tool round-trip belongs in `context`, closed by a `user` turn before the final reply.

**Tokenizer compatibility.** The recipes configure **`…_thinking_token_fixed_tools_fixed`** (tool results render as dedicated `<|tool_output_start/end|>` tokens). It shares the exact vocab/token-IDs of the `…_thinking_token_fixed`(`.snapshot-20260611`) variant — IDs 73/74 are the same reserved special slots, just *named* there — so the model's embeddings are unaffected by the choice. Variants:

| tokenizer | tool defs/calls accepted | tool-output render | thinking |
|---|---|---|---|
| `…_thinking_token_fixed` (+ `.snapshot`) | OpenAI-nested **or** flat | brackets `[ … ]` | `<\|inner_prefix\|>` |
| `…_thinking_token_fixed_tools_fixed` **(configured)** | OpenAI-nested or flat | `<\|tool_output_start/end\|>` tokens | `<\|inner_prefix\|>` |
| model checkpoint's bundled template | **flat only** | `<\|tool_output_start/end\|>` tokens | `<think>` |

The tool-**output** delimiter (`[ … ]` vs dedicated tokens) is a **rendering** difference only — the *input* data (`role:"tool"` turns / `tool_outputs` blocks) is identical and all *fixed* variants share one vocab (same token IDs), so switching among them needs **no** data change. The real data-format break is the **checkpoint's bundled template**: no `tool.function` unwrap (needs **flat** `{name, description, parameters}` specs, not the OpenAI-nested form here) and `<think>` thinking markers — which is why the recipes pin a fixed tokenizer and `default_tokenizer_to_model` warns when falling back to the checkpoint.

To smoke-test this path: `tools/build_offline_dpo_apertus_testset.py` emits a tiny preference set covering the full thinking × tools × multi/single-turn matrix (thinking rows set `enable_thinking` explicitly; no-think rows alternate explicit/omitted to exercise auto-detect; `tools` as a JSON string), and `infra/slurm/cscs/probe_dpo_toolthinking_apertus.slurm` runs it as a 1-node GPU probe via the recipe above.

### Online prompt data (and bringing your own)

Online DPO needs a **prompt-only** set. The additive **`prompt_only`** loader (`nemo_rl_apertus/online_data.py`, registered at startup by the entry point) reads a parquet/jsonl/HF set:

```yaml
data:
  train:
    dataset_name: prompt_only
    data_path: /path/to/prompts.parquet
    prompt_key: prompt             # column holding a string or a list of {role, content} turns
    processor: online_prompt_processor
    split_validation_size: 0.0     # >0 carves a held-out val split (see Validation)
  default:
    env_name: online_dpo_judge
    processor: online_prompt_processor
```

To use a different source, point `data_path`/`prompt_key` at it, or add a `RawDataset` subclass that emits `{messages, task_name}` and register it. Unlike offline DPO, online DPO's tokenization processor **is** registry-configurable via `data.processor` (the `online_prompt_processor` tokenizes the full prompt conversation with a generation prompt and sets the clean `judge_prompt`).

### Policy thinking (online rollouts)

Whether the **policy** reasons during a rollout is controlled per prompt by the `online_dpo.thinking` block (resolved in `online_prompt_processor`, so it applies only when a recipe uses that processor — the MaxMin `prompt_only` set — **not** the DeepScaler probe's inherited `math_hf_data_processor`). This is the Apertus `enable_thinking` chat-template kwarg (→ the developer block's `Deliberation: enabled/disabled`); it is **distinct** from `env.online_dpo_judge.enable_thinking`, which governs the *judge model's* own reasoning.

```yaml
online_dpo:
  thinking:
    mode: random          # default (chat-template default) | on | off | random
    probability: 0.5      # P(thinking on) per prompt when mode == random
    # seed: 42            # RNG seed for mode == random (defaults to grpo.seed)
```

- The decision is made **once per prompt**, *before* the driver repeats it into `R` rollouts, so a prompt's `R` rollouts share one thinking mode and stay an apples-to-apples preference group (the judge then ranks them on response quality / reasoning calibration, not on whether they happened to think).
- `mode: random` is a deterministic `Bernoulli(probability)` keyed by the sample index (+ seed), so it is **reproducible** across runs/resumes; `mode: default` omits the kwarg entirely (the prior behavior).
- A per-row override wins over the mode: give the prompt parquet an `enable_thinking` column (configurable via the loader's `thinking_key`) to force specific prompts on/off — mirroring the offline `ToolThinkingPreferenceProcessor`.
- The resolved flag travels in `extra_env_info` and is dumped per rollout as the `thinking` column of `online_dpo_rollouts_step<N>.jsonl`. Combine with the judge's `thinking_appropriateness` / `thinking_formatting` aspects (above) to reward well-calibrated, well-formed reasoning. (Do **not** also set `policy.tokenizer.chat_template_kwargs.enable_thinking` when using this — that global binding would collide with the per-prompt kwarg.)

## Logging & rollout inspection (online DPO)

W&B / TensorBoard are stock: enable via the recipe `logger.wandb_enabled: true` + `logger.wandb: {project, name, entity}` (and/or `logger.tensorboard_enabled`). The DPO launchers **pass `WANDB_API_KEY` through the environment** (never on a command line) — supply it with `sbatch --export=ALL,WANDB_API_KEY=$WANDB_API_KEY,…`. Whether W&B actually runs is decided by the recipe's `logger.wandb_enabled` (the MaxMin recipe defaults it **on**; the probes default **off**); the env knob **`WANDB_DISABLED=true`** force-disables it regardless of the recipe. Other `WANDB_*` vars (e.g. `WANDB_MODE=offline` for nodes without outbound HTTPS) are inherited too (the launchers set `SLURM_EXPORT_ENV=ALL`). Online DPO additionally **dumps each step's rollouts** to `<log_dir>/online_dpo_rollouts_step<N>.jsonl` — per rollout: `prompt`, `response` (the judge-scored spans), `judge_score`, `thinking` (the per-prompt reasoning toggle the processor resolved — `true`/`false`, or `null` when no thinking config is active), `group`/`rollout_in_group`, and `selection` (`chosen` / `rejected` / `degenerate` / `unused`). Cap with **`online_dpo.num_logged_rollouts`** (`-1` = all [default], `0` = off, `N` = first N). Aggregate metrics logged under `train/`: `pairs/*` (`judge_score_mean`, `num_pairs`, `num_degenerate_pairs`, `frac_valid_pairs`, chosen/rejected reward), `rollout/*` (generation/total token lengths + turn stats; the per-rollout reward is surfaced via `pairs/judge_score_mean`), and `loss`/`preference_loss`/`accuracy`.

## Validation

**Online DPO (opt-in, off by default).** A held-out **judge** evaluation — no DPO loss is computed (self-generated val pairs would be circular). Each validation generates `R` rollouts per held-out prompt and judges them, logging `validation/judge_score_mean` + `frac_valid_pairs`. Activate with the stock `grpo.val_period` / `val_at_start` / `val_at_end`; provide held-out prompts via **`data.train.split_validation_size > 0`** (carved from train; honored by the `prompt_only` loader, **not** `DeepScaler`) **or** a separate `data.validation` block. The val batch size is `grpo.val_batch_size`; cap the number of val batches with `online_dpo.val_batches` (absent = all). Each validation refits the generation engine, forcing a refit on the next train step. With all three flags off (the probe default) it is a complete no-op.

**Offline DPO.** Computes the DPO loss/accuracy on a **held-out preference set**. Activate with `dpo.val_period` / `val_batches` / `val_global_batch_size` / `val_at_start` plus a `data.validation` block (a held-out chosen/rejected parquet) — see the MaxMin offline recipe.
