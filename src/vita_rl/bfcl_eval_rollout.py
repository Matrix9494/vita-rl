"""BFCL held-out evaluation for the Dressage whitebox environment.

Slime passes ``evaluation=True`` to a custom generator during normal
evaluation.  Dressage's whitebox generator deliberately rejects that mode:
the environment itself owns the complete multi-turn rollout.  This module is
the small compatibility layer: it keeps Slime's normal dataset construction
and reporting contract, but invokes the whitebox generator as an ordinary
environment episode.
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.sglang_rollout import EVAL_PROMPT_DATASET, generate_and_rm
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample


async def _evaluate_dataset(args: Any, dataset_cfg: Any) -> dict[str, dict[str, Any]]:
    """Evaluate one dataset without forwarding Slime's evaluation flag."""
    eval_multimodal_keys = (
        dataset_cfg.multimodal_keys
        if dataset_cfg.multimodal_keys is not None
        else args.multimodal_keys
    )
    apply_template = (
        dataset_cfg.apply_chat_template
        if dataset_cfg.apply_chat_template is not None
        else args.apply_chat_template
    )
    template_kwargs = (
        dataset_cfg.apply_chat_template_kwargs
        if dataset_cfg.apply_chat_template_kwargs is not None
        else args.apply_chat_template_kwargs
    )
    cache_key = dataset_cfg.cache_key + (
        args.hf_checkpoint,
        apply_template,
        json.dumps(eval_multimodal_keys, sort_keys=True)
        if eval_multimodal_keys is not None
        else None,
        json.dumps(template_kwargs, sort_keys=True)
        if template_kwargs is not None
        else None,
    )
    if cache_key not in EVAL_PROMPT_DATASET:
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=load_tokenizer(args.hf_checkpoint, trust_remote_code=True),
            processor=load_processor(args.hf_checkpoint, trust_remote_code=True),
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=eval_multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=apply_template,
            apply_chat_template_kwargs=template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]
    params = {
        "temperature": dataset_cfg.temperature,
        "top_p": dataset_cfg.top_p,
        "top_k": dataset_cfg.top_k,
        "max_new_tokens": dataset_cfg.max_response_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": (
            dataset_cfg.skip_special_tokens
            if dataset_cfg.skip_special_tokens is not None
            else args.rollout_skip_special_tokens
        ),
        "no_stop_trim": dataset_cfg.no_stop_trim
        if dataset_cfg.no_stop_trim is not None
        else True,
        "spaces_between_special_tokens": False,
    }
    if dataset_cfg.repetition_penalty is not None:
        params["repetition_penalty"] = dataset_cfg.repetition_penalty

    tasks = []
    index = 0
    for prompt_sample in dataset.samples:
        for sample_number in range(dataset_cfg.n_samples_per_eval_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.index = index
            index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.custom_rm_path = dataset_cfg.custom_rm_path
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sample_params = params.copy()
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sample_params["sampling_seed"] = args.rollout_seed + sample_number
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(args, sample, sample_params, evaluation=False)
                )
            )

    samples = []
    for task in asyncio.as_completed(tasks):
        result = await task
        # Dressage returns prefix-chain segments for an agent trajectory.  BFCL
        # reward is defined only for the completed terminal episode, so retain
        # its final segment rather than scoring intermediate prefixes.
        samples.append(result[-1] if isinstance(result, list) else result)
    samples.sort(key=lambda sample: sample.index)
    # A transport/runtime failure yields Dressage's aborted sample rather than
    # a reward.  Treat that failed episode as the only valid terminal fallback
    # (zero); this keeps the evaluator's reward vector strictly binary and
    # prevents one failed request from invalidating all held-out metrics.
    for sample in samples:
        if sample.reward is None:
            sample.reward = 0.0
    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [
                sample.reward if not reward_key else sample.reward[reward_key]
                for sample in samples
            ],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in samples],
            "samples": samples,
        }
    }


async def _evaluate(args: Any) -> RolloutFnEvalOutput:
    results = await asyncio.gather(
        *[_evaluate_dataset(args, cfg) for cfg in (getattr(args, "eval_datasets", None) or [])]
    )
    data: dict[str, dict[str, Any]] = {}
    for result in results:
        data.update(result)
    return RolloutFnEvalOutput(data=data)


def generate_rollout(
    args: Any, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Slime rollout entrypoint; training remains delegated to its standard path."""
    if not evaluation:
        from slime.rollout.sglang_rollout import generate_rollout as standard_generate_rollout

        return standard_generate_rollout(args, rollout_id, data_source, evaluation=False)
    del rollout_id, data_source
    return run(_evaluate(args))
