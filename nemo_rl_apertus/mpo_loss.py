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
"""MPO (Mixed Preference Optimization) loss for NeMo-RL DPO training.

Implements the InternVL MPO objective (arXiv:2411.10442) as an additive
extension of the stock :class:`nemo_rl.algorithms.loss.loss_functions.DPOLossFn`:

    L = w_p * L_preference (DPO)  +  w_q * L_quality (BCO)  +  w_g * L_generation (SFT)

The preference and generation (SFT) terms are the stock DPOLossFn machinery,
reused verbatim via ``super().__call__``. This module adds only the BCO quality
term (paper Eq. 5-7; reported weights w_p=0.8, w_q=0.2, w_g=1.0, beta=0.1):

    r        = beta * (log pi_theta(y|x) - log pi_0(y|x))   # per-sample implicit reward
    L_q^+    = -log sigmoid( r_chosen   - delta)
    L_q^-    = -log sigmoid(-(r_rejected - delta))
    L_quality = L_q^+ + L_q^-   (per preference pair)

where ``delta`` is the running mean of ALL implicit rewards (chosen and
rejected) of valid samples seen so far — TRL ``BCOTrainer`` semantics
(``trl/trainer/bco_trainer.py::bco_loss`` + ``RunningMoments``): the current
batch is folded into the running mean FIRST, then ``delta`` is read for the
loss; the merge is count-weighted (``mean += (xs_mean - mean) * n / (count + n)``),
no EMA/momentum. Variance/std tracking is omitted because BCO consumes only the
mean.

Contracts
---------
1. **Bitwise-stock at w_q = 0**: with ``quality_loss_weight == 0.0`` the
   returned loss is the *same tensor object* produced by the stock
   ``DPOLossFn.__call__`` — bitwise identical training. Quality metrics and the
   delta stream are still computed (cheap, sequence-level) so delta can be
   monitored before enabling the term.
2. **Reward semantics**: the implicit reward mirrors
   ``DPOLossFn._dpo_loss`` exactly (token_mask[:, 1:], ref logprobs[:, :-1],
   masked diff sum, optional ``preference_average_log_probs`` division), then
   scales by ``reference_policy_kl_penalty`` (beta) — matching TRL's
   ``chosen_rewards = beta * chosen_logratios``.
3. **Normalization**: each side of the quality term is normalized with
   ``masked_mean(..., global_normalization_factor=global_valid_seqs / 2)`` —
   the identical per-pair convention the stock preference term uses. The
   reported quality loss is therefore *per pair* (chosen + rejected), i.e.
   2x TRL's per-sample mean.
4. **delta state residency**: ``loss_fn`` is re-pickled into the policy workers
   on every ``policy.train`` call, so instance attributes cannot carry running
   state across steps. The running mean lives in a module-level registry
   (``_DELTA_STATES``) inside the worker process; pickled ``MPOLossFn``
   instances are stateless views onto it (keyed by ``quality_delta_state_id``).
5. **Distributed correctness**: rewards are replicated across tensor-parallel
   ranks; the delta update all-reduces (sum, count) over the *data-parallel*
   group only (``megatron.core.parallel_state.get_data_parallel_group()`` in
   the Megatron worker, where the loss executes on the last PP stage). When
   torch.distributed is initialized but Megatron parallel state is not
   (DTensor path), it falls back to the world group: TP/CP replicas then
   contribute duplicate copies, which leaves the all-reduced *mean* exact
   (numerator and denominator scale together) and only inflates the absolute
   count by the replication factor. With torch.distributed uninitialized
   (unit tests) the update is local-only.
6. **Train-only updates**: both stock workers run eval under
   ``torch.no_grad()``, so ``next_token_logprobs.requires_grad`` discriminates
   train from validation microbatches. delta updates (and their collectives) run
   only on train microbatches — mirroring TRL's ``do_train`` gate and keeping
   validation side-effect-free. (Assumes a trainable policy, which DPO requires.)
7. **Persistence**: when ``quality_delta_state_path`` is set, rank 0 of the
   reduce group atomically rewrites a sidecar JSON ``{mean, count}`` after each
   update — the analog of TRL's ``running.json``. On worker (re)start the
   registry seeds from, in priority order: sidecar file > config
   (``quality_delta_init`` / ``quality_delta_resume_count``).

STEP-1 INVARIANT (sanity anchor for smoke tests)
------------------------------------------------
At policy == reference, log-ratios vanish, so every implicit reward r == 0:

* preference: -log sigmoid(beta * (0 - 0)) = -log(1/2) = ln 2 ~= 0.6931
* delta update folds a batch of zeros: delta = delta_init * c / (c + n)
  (== 0 when delta_init = 0)
* quality (per pair): -log sigmoid(0 - delta) - log sigmoid(-(0 - delta))
  = -log sigmoid(-delta) - log sigmoid(delta); at delta = 0 this is
  2 ln 2 ~= 1.3863
* total = w_pref * 0.6931 + w_q * 1.3863 + w_sft * (chosen NLL)

Config keys (:class:`MPOConfig` fields, an extension of the stock ``DPOLossConfig``)
------------------------------------------------------------------------------------
* ``quality_loss_weight``       (float, default 0.0) — w_q
* ``quality_delta_init``        (float, default 0.0) — delta seed at (re)start
* ``quality_delta_resume_count`` (float, default 0)  — count seed at (re)start
* ``quality_delta_state_path``  (str | None, default None) — sidecar JSON path
* ``quality_delta_state_id``    (str, default "default") — registry key

Returned metrics: everything DPOLossFn returns, plus ``quality_loss``,
``delta``, ``delta_count``, ``implicit_rewards_chosen_mean``,
``implicit_rewards_rejected_mean``. NOTE: the stock driver sums microbatch
metrics across microbatches x DP ranks before logging, so the *logged*
delta/delta_count are aggregates for monitoring; the sidecar holds the exact
values.
"""

import json
import os
from typing import Any, Optional

import torch
from pydantic import BaseModel
from torch import Tensor

from nemo_rl.algorithms.loss.loss_functions import (
    DPOLossConfig,
    DPOLossDataDict,
    DPOLossFn,
)
from nemo_rl.algorithms.utils import masked_mean
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

DELTA_SIDECAR_NAME = "mpo_delta.json"

# TRL RunningMoments epsilon: a fresh state adopts the first batch mean exactly.
_COUNT_EPS = 1e-24


class MPOConfig(DPOLossConfig):
    """The stock DPO loss fields plus the MPO quality-term keys (module docstring).

    ``extra="allow"`` is inherited, so the full ``dpo`` config section validates
    directly: ``MPOConfig.model_validate(master_config.dpo.model_dump())``.
    """

    quality_loss_weight: float = 0.0
    quality_delta_init: float = 0.0
    quality_delta_resume_count: float = 0.0
    quality_delta_state_path: Optional[str] = None
    quality_delta_state_id: str = "default"


class RunningMeanState:
    """Count-weighted running mean (TRL ``RunningMoments`` mean/count semantics)."""

    __slots__ = ("mean", "count")

    def __init__(self, mean: float = 0.0, count: float = _COUNT_EPS):
        self.mean = float(mean)
        self.count = float(count)

    @torch.no_grad()
    def update(
        self,
        local_sum: Tensor,
        local_count: Tensor,
        group: Optional["torch.distributed.ProcessGroup"],
    ) -> None:
        """Fold one batch of (already masked) per-sample rewards into the mean.

        ``local_sum`` / ``local_count`` are this rank's contributions; they are
        all-reduced over ``group`` so every member folds identical global batch
        statistics and the running state stays replica-identical.
        ``group is None`` means no reduction (torch.distributed uninitialized).
        """
        stats = torch.stack((local_sum.detach().float(), local_count.detach().float()))
        if group is not None:
            torch.distributed.all_reduce(stats, group=group)
        xs_sum, xs_count = float(stats[0]), float(stats[1])
        if xs_count <= 0:
            return  # fully-masked batch: nothing to fold (TRL never hits this)
        tot_count = self.count + xs_count
        # TRL RunningMoments.update: mean += (xs_mean - mean) * xs_count / tot_count
        self.mean += (xs_sum / xs_count - self.mean) * xs_count / tot_count
        self.count = tot_count


# Worker-resident registry (contract 4): survives across per-step re-pickled
# MPOLossFn copies because the worker process keeps this module loaded.
_DELTA_STATES: dict[str, RunningMeanState] = {}


def reset_delta_states() -> None:
    """Drop all worker-resident delta state (unit-test isolation hook)."""
    _DELTA_STATES.clear()


def delta_sidecar_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, DELTA_SIDECAR_NAME)


def read_delta_sidecar(path: str) -> Optional[dict[str, float]]:
    """Load ``{mean, count}`` from a sidecar JSON; None if absent/unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return {"mean": float(payload["mean"]), "count": float(payload["count"])}
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _write_delta_sidecar(path: str, state: RunningMeanState) -> None:
    """Atomically persist the running state (TRL ``running.json`` analog).

    Tmp name is pid-unique so concurrent writers (DP-group rank 0 of each
    tensor-parallel coordinate — their states are replica-identical) cannot
    collide; ``os.replace`` keeps readers from ever seeing a torn file.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"mean": state.mean, "count": state.count}, f)
    os.replace(tmp, path)


def _resolve_delta_group() -> Optional["torch.distributed.ProcessGroup"]:
    """Reduce group for delta statistics (contract 5)."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    try:
        from megatron.core import parallel_state

        if parallel_state.model_parallel_is_initialized():
            return parallel_state.get_data_parallel_group()
    except ImportError:
        pass
    # Non-Megatron (DTensor) fallback: mean stays exact under TP/CP replication;
    # only the absolute count is inflated by the replication factor.
    return torch.distributed.group.WORLD


def _is_delta_writer(group: Optional["torch.distributed.ProcessGroup"]) -> bool:
    if group is None:
        return True
    return torch.distributed.get_rank(group) == 0


class MPOLossFn(DPOLossFn):
    """Mixed Preference Optimization loss: stock DPO + BCO quality term.

    See the module docstring for the objective, contracts, config keys, the
    delta (running reward anchor) lifecycle, and the STEP-1 invariant.
    """

    def __init__(self, cfg: MPOConfig, use_linear_ce_fusion: bool = False):
        if not isinstance(cfg, MPOConfig):
            cfg = MPOConfig.model_validate(
                cfg.model_dump() if isinstance(cfg, BaseModel) else dict(cfg)
            )
        super().__init__(cfg, use_linear_ce_fusion=use_linear_ce_fusion)
        self.quality_loss_weight = cfg.quality_loss_weight
        self.quality_delta_init = cfg.quality_delta_init
        self.quality_delta_resume_count = cfg.quality_delta_resume_count
        self.quality_delta_state_path = cfg.quality_delta_state_path
        self.quality_delta_state_id = cfg.quality_delta_state_id

    def _delta_state(self) -> RunningMeanState:
        """Acquire the worker-resident running state (contracts 4 and 7)."""
        state = _DELTA_STATES.get(self.quality_delta_state_id)
        if state is None:
            sidecar = (
                read_delta_sidecar(self.quality_delta_state_path)
                if self.quality_delta_state_path is not None
                else None
            )
            if sidecar is not None:
                state = RunningMeanState(sidecar["mean"], sidecar["count"])
            else:
                state = RunningMeanState(
                    self.quality_delta_init,
                    self.quality_delta_resume_count + _COUNT_EPS,
                )
            _DELTA_STATES[self.quality_delta_state_id] = state
        return state

    def _implicit_rewards(
        self,
        next_token_logprobs: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
    ) -> Tensor:
        """Per-sample implicit reward r = beta * (log pi_theta - log pi_0).

        Mirrors the reward computation in ``DPOLossFn._dpo_loss`` exactly
        (``nemo_rl/algorithms/loss/loss_functions.py`` in this checkout), then
        applies beta — TRL bco_loss's ``chosen/rejected_rewards`` (contract 2).
        """
        token_mask = data["token_mask"][:, 1:]
        ref_logprobs = data["reference_policy_logprobs"][:, :-1]
        diff = (next_token_logprobs - ref_logprobs) * token_mask
        logratios = diff.sum(-1)
        if self.preference_average_log_probs:
            logratios = logratios / token_mask.sum(-1).clamp(min=1)
        return self.reference_policy_kl_penalty * logratios

    def __call__(  # type: ignore[override]
        self,
        next_token_logprobs: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        dpo_loss, metrics = super().__call__(
            next_token_logprobs, data, global_valid_seqs, global_valid_toks
        )

        sample_mask = data["sample_mask"]
        rewards = self._implicit_rewards(next_token_logprobs, data)

        state = self._delta_state()
        if next_token_logprobs.requires_grad:
            # Train microbatch (eval runs under torch.no_grad in both workers;
            # contract 6). TRL order: fold the batch FIRST, then read delta.
            detached = rewards.detach()
            group = _resolve_delta_group()
            state.update((detached * sample_mask).sum(), sample_mask.sum(), group)
            if self.quality_delta_state_path is not None and _is_delta_writer(group):
                _write_delta_sidecar(self.quality_delta_state_path, state)
        delta = state.mean  # python float: no gradient flows through the anchor

        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards)
        quality_chosen = -torch.nn.functional.logsigmoid(rewards_chosen - delta)
        quality_rejected = -torch.nn.functional.logsigmoid(-(rewards_rejected - delta))
        ## per-pair normalization, identical to the stock preference term (contract 3)
        quality_loss = masked_mean(
            quality_chosen,
            sample_mask[::2],
            global_normalization_factor=global_valid_seqs / 2,
        ) + masked_mean(
            quality_rejected,
            sample_mask[1::2],
            global_normalization_factor=global_valid_seqs / 2,
        )

        if self.quality_loss_weight != 0.0:
            mpo_loss = dpo_loss + self.quality_loss_weight * quality_loss
        else:
            mpo_loss = dpo_loss  # contract 1: the stock tensor itself, untouched

        with torch.no_grad():
            implicit_rewards_chosen_mean = masked_mean(
                rewards_chosen,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            )
            implicit_rewards_rejected_mean = masked_mean(
                rewards_rejected,
                sample_mask[1::2],
                global_normalization_factor=global_valid_seqs / 2,
            )

        metrics.update(
            {
                "loss": mpo_loss.item(),
                "quality_loss": quality_loss.item(),
                "delta": float(state.mean),
                "delta_count": float(state.count),
                "implicit_rewards_chosen_mean": implicit_rewards_chosen_mean.item(),
                "implicit_rewards_rejected_mean": implicit_rewards_rejected_mean.item(),
            }
        )
        return mpo_loss, metrics
