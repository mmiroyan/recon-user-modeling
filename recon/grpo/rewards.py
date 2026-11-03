"""Reward functions for reasoning synthesis."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from recon.grpo.action_generator import generate_action
from recon.grpo.judges.dim_judge import _DIMS, judge_responses
from recon.grpo.judges.sanity_check import check_verbatim
from recon.grpo.prompt_format import PromptConfig
from recon.grpo.rollout_parse import normalize_reasoning_trace


logger = logging.getLogger(__name__)


def _group_indices(prompts: list[str], kwargs: dict[str, Any]) -> list[list[int]]:
    session_ids = kwargs.get("session_id")
    turn_idx = kwargs.get("turn_idx")
    keys: list[Any] = (
        list(zip(session_ids, turn_idx))
        if session_ids is not None and turn_idx is not None
        else list(prompts)
    )
    groups_by_key: dict[Any, list[int]] = {}
    order: list[Any] = []
    for i, key in enumerate(keys):
        if key not in groups_by_key:
            groups_by_key[key] = []
            order.append(key)
        groups_by_key[key].append(i)
    return [groups_by_key[k] for k in order]


class _SanityJudge:
    """Runs sanity checks for a batch and caches results.

    SanityReward populates the cache via run(); DimJudgeReward reads from it
    so check_verbatim is called exactly once per rollout.
    """

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_retries: int,
        concurrency: int,
        api_base: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.api_base = api_base
        self._results: dict[tuple[str, str], Optional[bool]] = {}

    def run(self, gt_actions: list[str], traces: list[str]) -> None:
        """Compute and cache sanity results for the batch (called by SanityReward)."""
        self._results = {}
        pairs: list[tuple[str, str]] = list(dict.fromkeys(zip(gt_actions, traces)))

        def _one(pair: tuple[str, str]) -> tuple[tuple[str, str], Optional[bool]]:
            gt, trace = pair
            if not trace:
                return pair, False
            res = check_verbatim(
                gt_action=gt,
                reasoning_trace=trace,
                model=self.model,
                temperature=self.temperature,
                max_retries=self.max_retries,
                api_base=self.api_base,
            )
            # None → API failure; True → passed clean; False → contaminated
            return pair, (None if res is None else not res.found)

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for pair, result in ex.map(_one, pairs):
                self._results[pair] = result

    def get(self, gt_action: str, trace: str) -> Optional[bool]:
        """True = passed (clean), False = contaminated, None = API failure."""
        return self._results.get((gt_action, trace))


class SanityReward:
    __name__ = "sanity_reward"

    def __init__(
        self,
        *,
        sanity_judge: _SanityJudge,
        sanity_failure_reward: float = -1.0,
    ) -> None:
        self.sanity_judge = sanity_judge
        self.sanity_failure_reward = sanity_failure_reward

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        completion_ids: Optional[list[list[int]]] = None,
        log_extra=None,
        log_metric=None,
        **kwargs: Any,
    ) -> list[Optional[float]]:
        gt_actions = kwargs.get("gt_action")
        if gt_actions is None:
            raise KeyError("SanityReward requires dataset column 'gt_action'.")

        traces = [normalize_reasoning_trace(c) for c in completions]
        self.sanity_judge.run(gt_actions, traces)

        rewards: list[Optional[float]] = []
        for gt, trace in zip(gt_actions, traces):
            result = self.sanity_judge.get(gt, trace)
            if result is None:
                rewards.append(0.0)  # sanity API failure → zero reward
            elif result:  # passed clean
                rewards.append(0.0)
            else:  # contaminated
                rewards.append(self.sanity_failure_reward)

        if log_extra is not None:
            log_extra("sanity_reward", rewards)
        if log_metric is not None:
            valid = [r for r in rewards if r is not None]
            if valid:
                log_metric("mean_sanity_reward", sum(valid) / len(valid))
        return rewards


class DimJudgeReward:
    __name__ = "dim_judge_reward"

    def __init__(
        self,
        *,
        sanity_judge: _SanityJudge,
        action_generator_model: str,
        dim_judge_model: str,
        prompt_config: PromptConfig,
        dim_judge_max_retries: int = 3,
        judge_concurrency: int = 8,
        action_temperature: float = 0.7,
        dim_judge_temperature: float = 0.0,
        action_generator_api_base: Optional[str] = None,
        dim_judge_api_base: Optional[str] = None,
    ) -> None:
        self.sanity_judge = sanity_judge
        self.action_generator_model = action_generator_model
        self.dim_judge_model = dim_judge_model
        self.prompt_config = prompt_config
        self.dim_judge_max_retries = dim_judge_max_retries
        self.judge_concurrency = max(1, judge_concurrency)
        self.action_temperature = action_temperature
        self.dim_judge_temperature = dim_judge_temperature
        self.action_generator_api_base = action_generator_api_base
        self.dim_judge_api_base = dim_judge_api_base

    def _process_group(
        self,
        group_indices: list[int],
        *,
        completions: list[str],
        contexts: list[str],
        gt_actions: list[str],
    ) -> dict[str, Any]:
        k = len(group_indices)
        context = contexts[group_indices[0]]
        gt_action = gt_actions[group_indices[0]]
        traces = [normalize_reasoning_trace(completions[i]) for i in group_indices]

        rewards: list[Optional[float]] = [None] * k
        generated_actions: list[str] = [""] * k
        per_dim_scores: list[dict[str, float]] = [{dim: 0.0 for dim in _DIMS} for _ in range(k)]
        judge_payloads: list[dict[str, Any] | None] = [None] * k

        survivor_local_idx: list[int] = []
        for local_i, trace in enumerate(traces):
            s = self.sanity_judge.get(gt_action, trace)
            if s is None:
                rewards[local_i] = 0.0   # sanity API failure → zero reward
            elif not s:
                rewards[local_i] = 0.0   # contaminated → 0, no dim judge call
            else:
                survivor_local_idx.append(local_i)

        if not survivor_local_idx:
            return dict(rewards=rewards, generated_actions=generated_actions,
                        per_dim_scores=per_dim_scores, judge_payloads=judge_payloads)

        def _act(local_i: int) -> Optional[str]:
            return generate_action(
                context=context,
                reasoning_trace=traces[local_i],
                model=self.action_generator_model,
                prompt_config=self.prompt_config,
                temperature=self.action_temperature,
                max_retries=self.dim_judge_max_retries,
                api_base=self.action_generator_api_base,
            )

        with ThreadPoolExecutor(max_workers=self.judge_concurrency) as ex:
            actions_raw = list(ex.map(_act, survivor_local_idx))

        valid_actions: list[str] = []
        valid_local_idx: list[int] = []
        for local_i, action in zip(survivor_local_idx, actions_raw):
            if action is not None and action.strip():
                generated_actions[local_i] = action.strip()
                valid_actions.append(action.strip())
                valid_local_idx.append(local_i)
            else:
                rewards[local_i] = 0.0  # action gen failed → zero reward

        if not valid_actions:
            return dict(rewards=rewards, generated_actions=generated_actions,
                        per_dim_scores=per_dim_scores, judge_payloads=judge_payloads)

        judge_out = judge_responses(
            context=context,
            ground_truth_response=gt_action,
            responses=valid_actions,
            model=self.dim_judge_model,
            prompt_config=self.prompt_config,
            max_retries=self.dim_judge_max_retries,
            temperature=self.dim_judge_temperature,
            api_base=self.dim_judge_api_base,
        )
        if judge_out is None:
            for local_i in valid_local_idx:
                rewards[local_i] = 0.0  # dim judge failed → zero reward so advantages are 0
            return dict(rewards=rewards, generated_actions=generated_actions,
                        per_dim_scores=per_dim_scores, judge_payloads=judge_payloads)

        payload, avg_scores = judge_out
        for pos, local_i in enumerate(valid_local_idx, start=1):
            judge_payloads[local_i] = payload[str(pos)]
            per_dim_scores[local_i] = {
                dim: float(payload[str(pos)][dim]["alignment_score"]) for dim in _DIMS
            }
            rewards[local_i] = avg_scores[pos - 1]

        return dict(rewards=rewards, generated_actions=generated_actions,
                    per_dim_scores=per_dim_scores, judge_payloads=judge_payloads)

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        completion_ids: Optional[list[list[int]]] = None,
        log_extra=None,
        log_metric=None,
        **kwargs: Any,
    ) -> list[Optional[float]]:
        contexts = kwargs.get("context")
        gt_actions = kwargs.get("gt_action")
        if gt_actions is None or contexts is None:
            raise KeyError(
                "DimJudgeReward requires dataset columns 'gt_action' and 'context'."
            )

        rewards: list[Optional[float]] = [None] * len(completions)
        generated_actions: list[str] = [""] * len(completions)
        judge_payloads: list[dict[str, Any] | None] = [None] * len(completions)
        per_dim_scores: list[dict[str, float]] = [{dim: 0.0 for dim in _DIMS} for _ in completions]

        for group in _group_indices(prompts, kwargs):
            g = self._process_group(
                group,
                completions=completions,
                contexts=contexts,
                gt_actions=gt_actions,
            )
            for local_i, i in enumerate(group):
                rewards[i] = g["rewards"][local_i]
                generated_actions[i] = g["generated_actions"][local_i]
                judge_payloads[i] = g["judge_payloads"][local_i]
                per_dim_scores[i] = g["per_dim_scores"][local_i]

        if log_extra is not None:
            log_extra("generated_action", generated_actions)
            log_extra("dim_judge_reward", rewards)
            log_extra("dim_judge_payload", judge_payloads)
            for dim in _DIMS:
                log_extra(f"{dim}_score", [scores[dim] for scores in per_dim_scores])
        if log_metric is not None:
            valid = [r for r in rewards if r is not None]
            if valid:
                log_metric("mean_dim_judge_reward", sum(valid) / len(valid))

        return rewards
