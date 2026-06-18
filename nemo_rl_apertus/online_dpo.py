# Copyright (c) 2026, the Apertus project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Online DPO driver for Apertus (GRPO rollouts + judge ranking + DPO loss).

Online DPO has no offline chosen/rejected dataset. Each step:

1. generate ``R`` rollouts per prompt with the *current* policy (GRPO machinery),
2. score every rollout with a pluggable judge (``task_to_env`` → the judge env
   returns ``total_reward``; see :mod:`nemo_rl_apertus.online_judge`),
3. within each prompt's ``R`` rollouts pick best = chosen / worst = rejected,
4. apply the stock :class:`~nemo_rl.algorithms.loss.loss_functions.DPOLossFn` on
   those on-the-fly pairs,
5. optionally refresh the reference model to the current policy every ``N`` steps.

It is an **additive** layer: :func:`setup` calls the stock GRPO ``setup`` verbatim
(which already builds the policy with a reference model, the vLLM generation
engine, the clusters, the prompt-only dataloader, and the refit handshake), then
swaps the returned ``ClippedPGLossFn`` for a ``DPOLossFn`` — mirroring how
``run_mpo_apertus.py`` swaps the loss after ``dpo.setup``. The recipe is a GRPO
recipe plus an ``online_dpo`` block; the GRPO ``loss_fn`` block is built and
discarded.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, NotRequired, Optional, TypedDict

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    _default_grpo_save_state,
    refit_policy_generation,
)
from nemo_rl.algorithms.grpo import setup as grpo_setup
from nemo_rl.algorithms.loss import DPOLossFn
from nemo_rl.data.collate_fn import preference_collate_fn
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.utils.timer import Timer
from nemo_rl_apertus.online_judge import (
    judge_inputs_from_conversation,
    last_assistant_index,
)


class ThinkingConfig(TypedDict):
    """Per-prompt policy reasoning toggle for online rollouts (``online_dpo.thinking``).

    Resolved per prompt in ``online_prompt_processor`` (so it applies only when a recipe
    selects that processor — e.g. the MaxMin ``prompt_only`` set — not the DeepScaler
    probe's inherited ``math_hf_data_processor``) and shared across that prompt's ``R``
    rollouts. See :func:`nemo_rl_apertus.online_data._resolve_enable_thinking`.
    """

    mode: str  # "default" (chat-template default) | "on" | "off" | "random"
    probability: NotRequired[float]  # P(thinking on) per prompt when mode == "random"
    seed: NotRequired[int]  # RNG seed for mode == "random" (entry point defaults it to grpo.seed)


class OnlineDPOConfig(TypedDict):
    """The ``online_dpo`` config block (DPO-specific deltas on top of a GRPO recipe).

    Rollout knobs (``num_generations_per_prompt`` = R, ``num_prompts_per_step``,
    ``max_num_steps``, ``max_rollout_turns``, ``seed``) live in the standard
    ``grpo`` block; this block holds the preference-loss params and the online
    extras.
    """

    reference_update_freq: int  # <=0: frozen reference (stock DPO); N>0: refresh every N steps
    tie_eps: float  # a group whose (max-min) judge score <= tie_eps is masked
    reference_policy_kl_penalty: float  # DPO beta (stock DPOLossConfig key name)
    preference_loss_weight: float
    sft_loss_weight: float
    preference_average_log_probs: bool
    sft_average_log_probs: NotRequired[bool]
    num_logged_rollouts: NotRequired[int]  # rollouts/step dumped to JSONL: <0/None/absent = all, 0 = off, N = first N
    val_batches: NotRequired[Optional[int]]  # held-out judge-val batches per eval: None/absent = all
    thinking: NotRequired[ThinkingConfig]  # per-prompt rollout reasoning toggle (online_prompt_processor only)
    train_on_truncated: NotRequired[bool]  # False (default): mask truncated rollouts (GRPO parity); True: keep them so a low judge score makes them the rejected


def _build_dpo_loss_config(online_dpo_cfg: OnlineDPOConfig) -> dict[str, Any]:
    """Select the ``DPOLossConfig`` subset from the ``online_dpo`` block.

    The loss-param keys use the stock ``DPOLossConfig`` names verbatim (so a stock
    DPO recipe ports over directly); this just drops the online-only extras
    (``reference_update_freq``, ``tie_eps``).
    """
    return {
        "reference_policy_kl_penalty": online_dpo_cfg["reference_policy_kl_penalty"],
        "preference_loss_weight": online_dpo_cfg["preference_loss_weight"],
        "sft_loss_weight": online_dpo_cfg["sft_loss_weight"],
        "preference_average_log_probs": online_dpo_cfg["preference_average_log_probs"],
        "sft_average_log_probs": online_dpo_cfg.get(
            "sft_average_log_probs", online_dpo_cfg["preference_average_log_probs"]
        ),
    }


# =======================================================
# Setup
# =======================================================
def setup(
    master_config: dict[str, Any],
    tokenizer: Any,
    dataset: Any,
    val_dataset: Optional[Any],
    processor: Optional[Any] = None,
) -> tuple[Any, ...]:
    """Build the online-DPO run by reusing GRPO ``setup`` and swapping in ``DPOLossFn``.

    Returns the same 10-tuple shape as GRPO ``setup``, except the loss is a
    ``DPOLossFn``.
    """
    grpo_config = master_config["grpo"]
    policy_config = master_config["policy"]

    num_generations = grpo_config["num_generations_per_prompt"]
    assert num_generations >= 2, (
        f"online_dpo needs num_generations_per_prompt >= 2 to form a chosen/rejected "
        f"pair, got {num_generations}"
    )
    assert grpo_config["num_prompts_per_step"] == policy_config["train_global_batch_size"], (
        "online_dpo expects train_global_batch_size == grpo.num_prompts_per_step "
        "(one preference pair per prompt; the loss doubles the batch internally), "
        f"got {policy_config['train_global_batch_size']} and {grpo_config['num_prompts_per_step']}"
    )
    # DPOLossFn has no importance-sampling correction, so generation must be on-policy
    # in dtype terms: fp8 rollouts would bias the implicit reward. Restrict to bf16/fp16 for v1.
    gen_precision = policy_config["generation"].get("vllm_cfg", {}).get("precision")
    assert gen_precision != "fp8", (
        "online_dpo v1 does not support fp8 rollout generation (the DPO loss has no "
        "importance-sampling correction); use bf16/fp16."
    )

    # Megatron LR-schedule caveat: we reuse grpo.setup, which sets train_iters =
    # max_num_steps (1 optimizer step/step). Each DPO step consumes 2x the samples
    # (chosen+rejected), so a non-constant LR schedule would decay ~2x too fast
    # (reach min-lr at the halfway point). grpo.setup builds the policy in-call, so
    # we can't double train_iters after the fact (as offline dpo.setup does). The
    # shipped recipe uses lr_decay_style=constant, where this is a no-op; warn loudly
    # for any other decay style so the horizon can be set to account for the 2x batch.
    megatron_cfg = policy_config.get("megatron_cfg", {})
    if megatron_cfg.get("enabled", False):
        decay_style = megatron_cfg.get("scheduler", {}).get("lr_decay_style", "constant")
        if decay_style != "constant":
            warnings.warn(
                f"online_dpo + megatron with lr_decay_style={decay_style!r}: the LR "
                "schedule horizon assumes GRPO's 1x sample accounting, but each DPO "
                "step consumes 2x samples (chosen+rejected), so the LR will decay ~2x "
                "too fast. Use lr_decay_style=constant, or set the scheduler decay "
                "iters/samples to account for the doubled effective batch.",
                stacklevel=2,
            )

    # DPOLossFn is SEQUENCE_LEVEL: each chosen+rejected pair must reach the loss in ONE call so
    # split_output_tensor ([::2]/[1::2]) can form rewards_delta. Sequence packing feeds the loss
    # one sequence at a time (SequencePackingLossWrapper) and dynamic batching reorders rows —
    # either makes rewards_rejected empty, so preference_loss collapses to 0 with NO gradient
    # (grad_norm=0) and the policy never learns. Offline dpo.setup asserts both off for this
    # reason; we reuse grpo.setup (GRPO enables packing for its token-level loss), so force them
    # off here — before the policy is built — instead of editing the shared GRPO recipe chain.
    for _key in ("sequence_packing", "dynamic_batching"):
        _sub = policy_config.get(_key)
        if _sub is not None and _sub.get("enabled"):
            warnings.warn(
                f"online_dpo: disabling policy.{_key} — incompatible with the sequence-level "
                "DPO loss (it needs each chosen/rejected pair in one loss call; otherwise the "
                "gradient is zero and the policy does not learn).",
                stacklevel=2,
            )
            _sub["enabled"] = False

    # Reuse GRPO setup verbatim; discard its ClippedPGLossFn (we build DPOLossFn below).
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        _discarded_clipped_pg_loss,
        logger,
        checkpointer,
        save_state,
        master_config,
    ) = grpo_setup(master_config, tokenizer, dataset, val_dataset, processor)

    use_linear_ce_fusion = policy_config["megatron_cfg"]["enabled"] and policy_config[
        "megatron_cfg"
    ].get("use_linear_ce_fusion_loss", False)
    loss_fn = DPOLossFn(
        _build_dpo_loss_config(master_config["online_dpo"]),
        use_linear_ce_fusion=use_linear_ce_fusion,
    )

    return (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        save_state,
        master_config,
    )


# =======================================================
# Pair construction from rollouts
# =======================================================
def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep only the keys the preference collate uses (role/content/token_ids).

    Drops rollout-only keys (e.g. ``generation_logprobs``) and the judge env's
    extra fields so every message in a log has a uniform key set — exactly the
    shape offline DPO's message logs have, so ``preference_collate_fn`` behaves
    identically.
    """
    return {
        "role": message["role"],
        "content": message["content"],
        "token_ids": message["token_ids"],
    }


def _trim_to_last_assistant(
    message_log: LLMMessageLogType,
) -> Optional[list[dict[str, Any]]]:
    """Return prompt + the (last) assistant turn, dropping the judge's trailing env observation.

    The judge environment appends an ``environment`` observation message after the
    assistant turn; ``only_unmask_final`` in the collate would otherwise unmask
    that instead of the response. Returns ``None`` if there is no assistant turn.
    Uses the same ``last_assistant_index`` the judge uses, so the trained span and
    the judged span are defined identically.
    """
    last_assistant = last_assistant_index(message_log)
    if last_assistant < 0:
        return None
    return [_clean_message(m) for m in message_log[: last_assistant + 1]]


def _mask_truncated_rollouts(
    repeated_batch: BatchedDataDict[Any], train_on_truncated: bool
) -> None:
    """Fold the generation-truncation flag into per-rollout validity (in place).

    A rollout that hit ``max_new_tokens`` mid-generation (``truncated=True``) is, by default,
    masked out of the loss (``loss_multiplier -> 0``) so any pair whose chosen/rejected is
    truncated is dropped — parity with GRPO, which does not train on truncated sequences.

    With ``train_on_truncated=True`` this is a **no-op**: truncated rollouts stay eligible, so a
    judge that scores them low (e.g. the ``thinking_appropriateness`` / ``thinking_formatting``
    aspects penalize reasoning that runs until it is cut off) makes a truncated/incomplete
    rollout the *rejected* of its pair — actively training the policy to converge before it
    truncates, instead of silently discarding the sample.
    """
    if train_on_truncated or "truncated" not in repeated_batch:
        return
    keep = (~repeated_batch["truncated"].bool()).to(
        repeated_batch["loss_multiplier"].dtype
    )
    repeated_batch["loss_multiplier"] = repeated_batch["loss_multiplier"] * keep


def select_pairs_from_scores(
    scores: list[float], num_generations: int, tie_eps: float
) -> list[tuple[int, int, bool]]:
    """For each contiguous group of ``R`` rollouts, pick (best_idx, worst_idx, is_degenerate).

    Indices are global (into the repeated batch). A group whose best/worst scores
    differ by <= ``tie_eps`` is degenerate (no usable preference signal) → masked.
    """
    pairs: list[tuple[int, int, bool]] = []
    num_groups = len(scores) // num_generations
    for g in range(num_groups):
        start = g * num_generations
        group = scores[start : start + num_generations]
        best = max(range(num_generations), key=lambda i: group[i])
        worst = min(range(num_generations), key=lambda i: group[i])
        degenerate = (group[best] - group[worst]) <= tie_eps
        pairs.append((start + best, start + worst, degenerate))
    return pairs


def build_preference_batch(
    repeated_batch: BatchedDataDict[Any],
    scores: list[float],
    num_generations: int,
    tie_eps: float,
    tokenizer: Any,
    make_sequence_length_divisible_by: int,
) -> tuple[BatchedDataDict[Any], dict[str, float]]:
    """Assemble the interleaved chosen/rejected DPO batch from judge-ranked rollouts.

    Degenerate or malformed pairs are kept (so the batch stays
    ``2 * num_prompts`` rows for the no-partial-batch invariant) but masked via
    ``loss_multiplier = 0``. Reuses the stock ``preference_collate_fn`` so the
    output contract matches offline DPO exactly.
    """
    pairs = select_pairs_from_scores(scores, num_generations, tie_eps)
    message_logs = repeated_batch["message_log"]
    sample_validity = repeated_batch["loss_multiplier"]

    data_batch: list[dict[str, Any]] = []
    num_degenerate = 0
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    for g, (chosen_idx, rejected_idx, degenerate) in enumerate(pairs):
        chosen_log = _trim_to_last_assistant(message_logs[chosen_idx])
        rejected_log = _trim_to_last_assistant(message_logs[rejected_idx])
        prompt_valid = (
            float(sample_validity[chosen_idx]) > 0
            and float(sample_validity[rejected_idx]) > 0
        )
        keep = (
            not degenerate
            and chosen_log is not None
            and rejected_log is not None
            and prompt_valid
        )
        if not keep:
            num_degenerate += 1
            # Fall back to raw (cleaned) logs so the collate still has valid tensors.
            if chosen_log is None:
                chosen_log = [_clean_message(m) for m in message_logs[chosen_idx]]
            if rejected_log is None:
                rejected_log = [_clean_message(m) for m in message_logs[rejected_idx]]
        else:
            chosen_scores.append(scores[chosen_idx])
            rejected_scores.append(scores[rejected_idx])

        data_batch.append(
            {
                "message_log_chosen": chosen_log,
                "message_log_rejected": rejected_log,
                "length_chosen": sum(len(m["token_ids"]) for m in chosen_log),
                "length_rejected": sum(len(m["token_ids"]) for m in rejected_log),
                "loss_multiplier": 1.0 if keep else 0.0,
                "idx": g,
            }
        )

    batch = preference_collate_fn(
        data_batch,
        tokenizer=tokenizer,
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
        add_loss_mask=True,
    )
    metrics = {
        "num_pairs": float(len(pairs)),
        "num_degenerate_pairs": float(num_degenerate),
        "frac_valid_pairs": (len(pairs) - num_degenerate) / max(len(pairs), 1),
        "judge_score_mean": float(np.mean(scores)) if scores else 0.0,
        "chosen_reward_mean": float(np.mean(chosen_scores)) if chosen_scores else 0.0,
        "rejected_reward_mean": float(np.mean(rejected_scores))
        if rejected_scores
        else 0.0,
    }
    return batch, metrics


def build_rollout_log(
    repeated_batch: BatchedDataDict[Any],
    scores: list[float],
    num_generations: int,
    tie_eps: float,
    num_logged_rollouts: Optional[int],
    tokenizer: Optional[Any] = None,
) -> dict[str, list[Any]]:
    """Build a per-rollout record (prompt/response/judge_score/selection) for JSONL dumping.

    ``num_logged_rollouts`` caps how many rollouts (rows of the repeated batch) are
    recorded: ``< 0`` or ``None`` = all, ``0`` = none, ``N`` = the first ``N``. ``selection`` marks
    each rollout ``chosen``/``rejected`` (its group's best/worst), ``degenerate`` (a masked
    tie — for an exact tie best==worst, so it collapses to that one index), or ``unused``.
    ``prompt``/``response`` are the exact spans the judge scored (the clean ``judge_prompt``
    when the processor supplied one). ``thinking`` is the per-prompt reasoning toggle the
    processor resolved (``True``/``False``, or ``None`` when no thinking config was active).
    ``tokenizer`` (set on the reasoning-aspect path) makes ``response`` the same special-token-
    preserving re-decode the judge scored, so the log matches the judged completion.
    Returns a dict of equal-length columns for ``Logger.log_batched_dict_as_jsonl``.
    """
    message_logs = repeated_batch["message_log"]
    extra_env_info = repeated_batch.get("extra_env_info")
    total = len(message_logs)
    n = (
        total
        if num_logged_rollouts is None or num_logged_rollouts < 0
        else min(num_logged_rollouts, total)
    )

    selection: dict[int, str] = {}
    for chosen_idx, rejected_idx, degenerate in select_pairs_from_scores(
        scores, num_generations, tie_eps
    ):
        if degenerate:
            selection[chosen_idx] = "degenerate"
            selection[rejected_idx] = "degenerate"
        else:
            selection[chosen_idx] = "chosen"
            selection[rejected_idx] = "rejected"

    log: dict[str, list[Any]] = {
        "group": [],
        "rollout_in_group": [],
        "selection": [],
        "judge_score": [],
        "thinking": [],
        "prompt": [],
        "response": [],
    }
    for i in range(n):
        meta = extra_env_info[i] if extra_env_info is not None else None
        prompt_text, response_text, _ = judge_inputs_from_conversation(
            message_logs[i], meta, tokenizer=tokenizer
        )
        log["group"].append(i // num_generations)
        log["rollout_in_group"].append(i % num_generations)
        log["selection"].append(selection.get(i, "unused"))
        log["judge_score"].append(float(scores[i]))
        log["thinking"].append((meta or {}).get("enable_thinking"))
        log["prompt"].append(prompt_text)
        log["response"].append(response_text)
    return log


# =======================================================
# Validation (held-out judge metrics; opt-in)
# =======================================================
def _should_run_validation(
    val_dataloader: Optional[StatefulDataLoader],
    val_period: int,
    total_steps: int,
    is_last_step: bool,
    val_at_end: bool,
) -> bool:
    """Periodic / end-of-run validation gate (``val_at_start`` is handled separately).

    Returns ``False`` when there is no val dataloader, so a cadence flag set without
    validation data is a safe no-op (deactivated). ``total_steps`` is the just-completed
    (already incremented) step, so ``total_steps % val_period`` matches GRPO's pre-increment
    ``(total_steps + 1) % val_period``.
    """
    if val_dataloader is None:
        return False
    return (val_period > 0 and total_steps % val_period == 0) or (
        val_at_end and is_last_step
    )


def validate_online_dpo(
    policy_generation: Any,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: Any,
    val_task_to_env: Optional[dict[str, Any]],
    num_generations: int,
    tie_eps: float,
    max_seq_len: int,
    max_rollout_turns: int,
    max_val_batches: Optional[int],
) -> dict[str, float]:
    """Generate + judge on held-out prompts; return held-out judge metrics (no optimizer step).

    Online DPO has no static preference set, so validation measures generalization via the
    judge: for each held-out prompt it generates ``R`` rollouts, scores them, and reports the
    mean judge score plus the fraction of prompts that yield a usable (non-degenerate)
    preference pair. No DPO loss is computed — freshly self-generated val pairs would make a
    val loss circular; the held-out judge score is the meaningful generalization signal.
    Caps at ``max_val_batches`` batches (``None`` = all); returns ``{}`` if there is no
    validation dataloader. The caller owns the generation-engine prepare/finish.
    """
    if val_dataloader is None:
        return {}
    all_scores: list[float] = []
    all_lengths: list[float] = []
    num_pairs = 0
    num_nondegenerate = 0
    for batch_idx, val_batch in enumerate(val_dataloader):
        if max_val_batches is not None and batch_idx >= max_val_batches:
            break
        repeated = val_batch.repeat_interleave(num_generations)
        repeated, gen_metrics = run_multi_turn_rollout(
            policy_generation=policy_generation,
            input_batch=repeated,
            tokenizer=tokenizer,
            task_to_env=val_task_to_env,
            max_seq_len=max_seq_len,
            max_rollout_turns=max_rollout_turns,
            greedy=False,
        )
        scores = repeated["total_reward"].tolist()
        all_scores.extend(scores)
        all_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])
        for _, _, degenerate in select_pairs_from_scores(scores, num_generations, tie_eps):
            num_pairs += 1
            if not degenerate:
                num_nondegenerate += 1
    return {
        "judge_score_mean": float(np.mean(all_scores)) if all_scores else 0.0,
        "frac_valid_pairs": (num_nondegenerate / num_pairs) if num_pairs else 0.0,
        "num_pairs": float(num_pairs),
        "num_samples": float(len(all_scores)),
        "mean_gen_tokens_per_sample": float(np.mean(all_lengths)) if all_lengths else 0.0,
    }


def _validate_and_log(
    policy: Any,
    policy_generation: Any,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: Any,
    val_task_to_env: Optional[dict[str, Any]],
    num_generations: int,
    tie_eps: float,
    max_seq_len: int,
    max_rollout_turns: int,
    max_val_batches: Optional[int],
    colocated_inference: bool,
    need_refit: bool,
    stale: bool,
    logger: Any,
    step: int,
) -> None:
    """Engine-managed validation: refit/prepare generation → validate → finish → log.

    Always calls ``finish_generation`` (which discards vLLM weights under sleep_level>=2), so
    the caller MUST set ``POLICY_GENERATION_STALE = True`` afterward to force a refit before the
    next train generation. Validation timings are logged under ``timing/validation``.
    """
    val_timer = Timer()
    print(f"▶ Running online-DPO validation at step {step}...", flush=True)
    with val_timer.time("total_validation_time"):
        if need_refit and stale:
            refit_policy_generation(
                policy, policy_generation, colocated_inference, timer=val_timer
            )
        elif colocated_inference:
            policy.offload_after_refit()
            policy_generation.prepare_for_generation()
        else:
            policy_generation.prepare_for_generation()
        val_metrics = validate_online_dpo(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            num_generations,
            tie_eps,
            max_seq_len,
            max_rollout_turns,
            max_val_batches,
        )
        policy_generation.finish_generation()
    logger.log_metrics(val_metrics, step, prefix="validation")
    logger.log_metrics(
        val_timer.get_timing_metrics(reduction_op="sum"), step, prefix="timing/validation"
    )
    if val_metrics:
        print(
            f"  • val judge_score_mean: {val_metrics['judge_score_mean']:.4f} "
            f"| frac_valid_pairs: {val_metrics['frac_valid_pairs']:.3f} "
            f"| samples: {int(val_metrics['num_samples'])}",
            flush=True,
        )


# =======================================================
# Training loop
# =======================================================
def online_dpo_train(
    policy: Any,
    policy_generation: Optional[Any],
    dataloader: StatefulDataLoader,
    tokenizer: Any,
    loss_fn: DPOLossFn,
    task_to_env: dict[str, Any],
    logger: Any,
    checkpointer: Any,
    save_state: GRPOSaveState,
    master_config: dict[str, Any],
    val_dataloader: Optional[StatefulDataLoader] = None,
    val_task_to_env: Optional[dict[str, Any]] = None,
) -> None:
    """Run online DPO: generate → judge → pair → DPO step → (optional) ref refresh.

    Validation (held-out judge metrics) is opt-in via the standard ``grpo`` val cadence
    (``val_period`` / ``val_at_start`` / ``val_at_end``); it needs a ``val_dataloader``
    (provide val data, e.g. ``data.train.split_validation_size > 0``). See
    :func:`validate_online_dpo`.
    """
    timer = Timer()
    grpo_config = master_config["grpo"]
    online_dpo_config = master_config["online_dpo"]
    policy_config = master_config["policy"]

    if save_state is None:
        save_state = _default_grpo_save_state()

    num_generations = grpo_config["num_generations_per_prompt"]
    max_num_steps = grpo_config["max_num_steps"]
    max_num_epochs = grpo_config["max_num_epochs"]
    max_rollout_turns = grpo_config["max_rollout_turns"]
    max_seq_len = policy_config["max_total_sequence_length"]
    train_gbs = policy_config["train_global_batch_size"]
    train_mbs = policy_config["train_micro_batch_size"]
    make_seq_div_by = policy_config["make_sequence_length_divisible_by"]
    colocated_inference = policy_config["generation"]["colocated"]["enabled"]
    tie_eps = online_dpo_config["tie_eps"]
    ref_update_freq = online_dpo_config["reference_update_freq"]
    # Rollout dump cap: <0 / None (absent) = all rollouts/step, 0 = off, N = first N.
    num_logged_rollouts = online_dpo_config.get("num_logged_rollouts")
    # Truncated rollouts (hit max_new_tokens mid-generation): masked from the loss by default
    # (GRPO parity); True keeps them eligible so a low judge score makes them the rejected.
    train_on_truncated = online_dpo_config.get("train_on_truncated", False)
    # When the judge re-decodes the completion with special tokens (reasoning aspects; the
    # entry point sets env.online_dpo_judge.completion_tokenizer), log the same re-decode so
    # the dumped `response` matches the judged span. Otherwise log the (stripped) content.
    judge_redecodes_completion = bool(
        master_config.get("env", {})
        .get("online_dpo_judge", {})
        .get("completion_tokenizer")
    )
    rollout_log_tokenizer = tokenizer if judge_redecodes_completion else None
    # Validation cadence (reuses the stock grpo knobs); a no-op unless a val_dataloader exists.
    val_at_start = grpo_config["val_at_start"]
    val_at_end = grpo_config["val_at_end"]
    val_period = grpo_config["val_period"]
    val_batches = online_dpo_config.get("val_batches", None)  # None = all val batches

    NEED_REFIT = True
    if policy_generation is None:
        policy_generation = policy
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True

    current_step = save_state["current_step"]
    total_steps = save_state["total_steps"]
    current_epoch = save_state["current_epoch"]
    consumed_samples = save_state["consumed_samples"]

    if val_at_start and total_steps == 0 and val_dataloader is not None:
        _validate_and_log(
            policy,
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            num_generations,
            tie_eps,
            max_seq_len,
            max_rollout_turns,
            val_batches,
            colocated_inference,
            NEED_REFIT,
            POLICY_GENERATION_STALE,
            logger,
            total_steps,
        )
        POLICY_GENERATION_STALE = True  # finish_generation may have slept-discarded weights

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")
        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/"
                f"{min(len(dataloader), max_num_steps)} {'=' * 25}",
                flush=True,
            )
            metrics: dict[str, Any] = {}
            with timer.time("total_step_time"):
                # ---- 1. repeat prompts R times ----
                repeated_batch = batch.repeat_interleave(num_generations)

                # ---- 2. refit generation engine to the current policy, then generate ----
                with timer.time("prepare_for_generation"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy, policy_generation, colocated_inference, timer=timer
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
                        policy_generation.prepare_for_generation()

                with timer.time("generation"):
                    repeated_batch, rollout_metrics = run_multi_turn_rollout(
                        policy_generation=policy_generation,
                        input_batch=repeated_batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=max_seq_len,
                        max_rollout_turns=max_rollout_turns,
                        greedy=False,
                    )
                    policy_generation.finish_generation()

                # ---- 3. judge scores (per rollout) -> 4. pick chosen/rejected ----
                scores = repeated_batch["total_reward"].tolist()
                with timer.time("data_processing"):
                    # Mask truncated rollouts from the loss (GRPO parity) unless
                    # train_on_truncated keeps them as eligible (low-scored) rejecteds.
                    _mask_truncated_rollouts(repeated_batch, train_on_truncated)
                    train_batch, pair_metrics = build_preference_batch(
                        repeated_batch,
                        scores,
                        num_generations,
                        tie_eps,
                        tokenizer,
                        make_seq_div_by,
                    )

                # ---- optional: dump this step's rollouts (prompt/response/score/selection) ----
                # Logged every step (incl. all-degenerate ones, to debug why) to
                # <log_dir>/online_dpo_rollouts_step<N>.jsonl. Online DPO has no validation
                # loop, so this is the primary way to inspect generations + judge scores.
                if num_logged_rollouts != 0:
                    logger.log_batched_dict_as_jsonl(
                        build_rollout_log(
                            repeated_batch,
                            scores,
                            num_generations,
                            tie_eps,
                            num_logged_rollouts,
                            rollout_log_tokenizer,
                        ),
                        f"online_dpo_rollouts_step{total_steps + 1}.jsonl",
                    )

                num_valid_pairs = float(train_batch["sample_mask"].sum().item())
                if num_valid_pairs == 0:
                    # No judge signal this step: skip the ref-logprobs + optimizer step.
                    # The policy weights are unchanged, but finish_generation() already ran
                    # this step and under sleep_level>=2 it discards the generation engine's
                    # weights — so force a refit before the next generation (else it would
                    # run on freed/garbage weights). Mirrors the post-validation guard.
                    POLICY_GENERATION_STALE = True
                    warnings.warn(
                        "online_dpo: all preference pairs degenerate this step "
                        "(no judge signal); skipping the optimizer step.",
                        stacklevel=2,
                    )
                else:
                    # ---- 5. reference logprobs (lp-inference mode), then the DPO step ----
                    with timer.time("logprob_inference_prep"):
                        policy.prepare_for_lp_inference()
                    with timer.time("reference_logprobs"):
                        reference_logprobs = policy.get_reference_policy_logprobs(
                            train_batch, micro_batch_size=train_mbs * 2
                        )["reference_logprobs"]
                        # roll left by one so index t holds the logprob of the next token
                        train_batch["reference_policy_logprobs"] = torch.roll(
                            reference_logprobs, -1, dims=-1
                        )

                    with timer.time("training_prep"):
                        policy.prepare_for_training()
                        POLICY_GENERATION_STALE = True

                    with timer.time("policy_training"):
                        train_results = policy.train(
                            train_batch,
                            loss_fn,
                            eval_mode=False,
                            gbs=train_gbs * 2,
                            mbs=train_mbs * 2,
                            timer=timer,
                        )
                    metrics["loss"] = train_results["loss"].numpy()
                    metrics["grad_norm"] = train_results["grad_norm"].numpy()
                    for k, v in train_results["all_mb_metrics"].items():
                        metrics[k] = (
                            np.mean(v).item()
                            if k in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}
                            else np.sum(v).item()
                        )

                    # ---- 6. optional reference refresh (after this step consumed its ref) ----
                    # NOTE: this lives in the non-degenerate branch, so an all-degenerate
                    # step (skipped optimizer step) also skips a scheduled refresh — the
                    # reference only advances on steps that actually trained.
                    if ref_update_freq > 0 and (total_steps + 1) % ref_update_freq == 0:
                        print("▶ Refreshing reference model to current policy...", flush=True)
                        policy.update_reference_model()

            # ---- logging ----
            metrics.update({f"pairs/{k}": v for k, v in pair_metrics.items()})
            metrics.update({f"rollout/{k}": v for k, v in rollout_metrics.items()})
            timing_metrics = timer.get_timing_metrics(reduction_op="sum")
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")
            print("\n📊 Online DPO step results:")
            # loss/accuracy first, then the gradient + loss-normalization diagnostics
            # (grad_norm tells if the step actually learned; num_valid_samples vs
            # global_valid_seqs/toks shows what the sequence-level loss is normalized by).
            # np.mean() collapses array-valued metrics (e.g. grad_norm) to a scalar safely.
            for key in (
                "loss", "preference_loss", "accuracy", "sft_loss", "grad_norm",
                "rewards_chosen_mean", "rewards_rejected_mean",
                "num_valid_samples", "global_valid_seqs", "global_valid_toks",
            ):
                if key in metrics:
                    print(f"  • {key}: {float(np.mean(metrics[key])):.4f}")
            # sample_mask carries one entry per chosen+rejected row, so it double-counts
            # pairs; report the pair count from the (un-doubled) pair metrics instead.
            valid_pairs = int(pair_metrics["num_pairs"] - pair_metrics["num_degenerate_pairs"])
            print(
                f"  • pairs valid/total: "
                f"{valid_pairs}/{int(pair_metrics['num_pairs'])} "
                f"| judge_score_mean: {pair_metrics['judge_score_mean']:.4f}"
            )
            timer.reset()

            # ---- checkpoint ----
            consumed_samples += grpo_config["num_prompts_per_step"]
            current_step += 1
            total_steps += 1
            is_last_step = total_steps >= max_num_steps or (
                current_epoch + 1 == max_num_epochs and current_step == len(dataloader)
            )

            # ---- validation (held-out judge metrics; no optimizer step) ----
            if _should_run_validation(
                val_dataloader, val_period, total_steps, is_last_step, val_at_end
            ):
                _validate_and_log(
                    policy,
                    policy_generation,
                    val_dataloader,
                    tokenizer,
                    val_task_to_env,
                    num_generations,
                    tie_eps,
                    max_seq_len,
                    max_rollout_turns,
                    val_batches,
                    colocated_inference,
                    NEED_REFIT,
                    POLICY_GENERATION_STALE,
                    logger,
                    total_steps,
                )
                POLICY_GENERATION_STALE = True  # finish_generation may have slept-discarded weights

            should_save = master_config["checkpointing"]["enabled"] and (
                is_last_step
                or total_steps % master_config["checkpointing"]["save_period"] == 0
            )
            if should_save:
                _save_checkpoint(
                    checkpointer,
                    policy,
                    dataloader,
                    save_state,
                    master_config,
                    total_steps=total_steps,
                    current_step=current_step % len(dataloader),
                    current_epoch=current_epoch,
                    consumed_samples=consumed_samples,
                    timer=timer,
                )

            if total_steps >= max_num_steps:
                print("Max number of steps reached, stopping training.", flush=True)
                return

        current_epoch += 1
        current_step = 0


def _save_checkpoint(
    checkpointer: Any,
    policy: Any,
    dataloader: StatefulDataLoader,
    save_state: GRPOSaveState,
    master_config: dict[str, Any],
    *,
    total_steps: int,
    current_step: int,
    current_epoch: int,
    consumed_samples: int,
    timer: Timer,
) -> None:
    """Persist policy weights/optimizer, dataloader state, and the save-state."""
    # NOTE: GRPOSaveState also carries total_valid_tokens, which online DPO does not
    # track (it's a throughput counter, not needed for resume); it stays at its init 0.
    save_state["total_steps"] = total_steps
    save_state["current_step"] = current_step
    save_state["current_epoch"] = current_epoch
    save_state["consumed_samples"] = consumed_samples
    with timer.time("checkpointing"):
        print(f"Saving checkpoint for step {total_steps}...", flush=True)
        checkpoint_path = checkpointer.init_tmp_checkpoint(
            total_steps, save_state, master_config
        )
        policy.save_checkpoint(
            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
            optimizer_path=os.path.join(checkpoint_path, "policy", "optimizer")
            if checkpointer.save_optimizer
            else None,
            tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
            checkpointing_cfg=master_config["checkpointing"],
        )
        torch.save(
            dataloader.state_dict(),
            os.path.join(checkpoint_path, "train_dataloader.pt"),
        )
        checkpointer.finalize_checkpoint(checkpoint_path)
