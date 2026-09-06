# Apertus Omni GRPO+ALP consumer

Status: implemented in this fork; end-to-end runtime certification is still
required.

This note describes the code that currently consumes the pretokenized
`rl_prompt` dataset. It is an implementation record, not evidence that the
configured multi-node workload has completed successfully.

## Data contract

Each split consists of:

- `{split}.bin` and `{split}.idx`: prompt-only MMIDIDX documents containing
  generation-ready Omni token IDs.
- `index_{split}.parquet`: one sidecar row per MMIDIDX document, in the same
  order, with `answer` and `answer_variants` grading metadata.

The consumer checks that the document and sidecar row counts match. It passes
the stored prompt IDs through unchanged; applying a chat template or tokenizing
them again would corrupt the inline image-token representation.

`answer_type` is not part of the consumer contract. The verifier grades against
the gold set formed from `answer` and `answer_variants`.

## Implemented path

| Stage | Implementation |
|---|---|
| Dataset | `nemo_rl/data/datasets/response_datasets/rl_prompt_dataset.py` reads and aligns MMIDIDX documents with the Parquet sidecar. |
| Processor | `mmididx_grpo_data_processor` in `nemo_rl/data/processors.py` creates a `DatumSpec` with pretokenized `message_log` content and verifier metadata. |
| Generation | The existing vLLM `prompt_token_ids` path consumes the prompt IDs; inline image tokens remain opaque token IDs. |
| Verifier | `nemo_rl/environments/single_turn_verifier_environment.py` compares the extracted response with every accepted gold answer. |
| Reward shaping | Existing GRPO ALP shaping uses the verifier's binary reward as its pass-rate signal. |
| Recipes | `grpo-apertus-omni-reasoning-alp-smoke.yaml` and `grpo-apertus-omni-reasoning-alp-8n.yaml`. |

This is an Omni GRPO path, not a pixel-VLM environment: images have already
been represented as token IDs by the producer before NeMo-RL reads the data.

## Verifier behavior

The verifier returns `1.0` when any accepted gold matches and `0.0` otherwise.
It attempts symbolic comparison, numeric comparison, and normalized exact-text
comparison. Rewards must remain in `[0, 1]` so ALP pass rate retains its
intended meaning.

The intended response-extraction contract is, in priority order:

1. the final `\\boxed{...}` answer;
2. the final `Answer: ...` line;
3. the final `<answer>...</answer>` block;
4. the final non-empty response line.

The implementation follows this priority and uses the final occurrence within
each format. Direct extraction tests cover every format, final-occurrence
selection, and cross-format priority.

## Current production configuration

The 8-node recipe currently configures:

- `cluster.num_nodes: 8`;
- `max_total_sequence_length: 8192`;
- `generation.max_new_tokens: 6000`;
- `vllm_cfg.max_model_len: 8192`;
- `reward_shaping.max_response_length: 6000`;
- stop token IDs `2`, `68`, and `72`;
- eight single-turn verifier workers.

These are configuration values, not a certification claim. The repository does
not contain recorded evidence of a completed 8-node Omni training run, so the
recipe must not be described as validated at that scale.

## Validation status

`tests/unit/data/test_rl_prompt_dataset.py` uses a fake indexed store to verify
document/sidecar alignment, prompt-ID pass-through, metadata propagation,
oversized-prompt handling, and dataset registration.

The remaining validation gaps are:

- a real MMIDIDX-plus-Parquet integration fixture;
- direct end-to-end verifier tests for every comparator;
- a recorded end-to-end smoke run proving generation, reward flow, ALP shaping,
  and checkpointing;
- a recorded multi-node completion before making an 8-node validation claim.

## Invariants

- Never re-tokenize or apply a chat template to stored prompt IDs.
- Require exactly one sidecar row per indexed document.
- Preserve `answer_variants` through collation and rollout metadata slicing.
- Keep verifier rewards in `[0, 1]`.
- Keep ALP `max_response_length` consistent with the generation output limit.
- Treat configured topology and sequence length as unverified until backed by a
  completed run and retained logs.
