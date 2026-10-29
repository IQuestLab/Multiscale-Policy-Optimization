"""Claw-compatible training dumps using Search's actual rollout tensors.

Field names follow claw-eval-ppo/verl/verl/trainer/ppo/ray_trainer.py's
_dump_token_level_data. Missing backend-specific metadata is omitted.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_jsonl(path, entries):
    """Stream rows and publish only a complete step file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for entry in entries:
                stream.write(json.dumps(json_safe(entry), ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# Only names already emitted by Claw; do not serialize all Search extra_fields.
_METADATA_FIELDS = {
    "data_source": "data_source",
    "num_turns": "__num_turns__",
    "source_sample_index": "__source_sample_index__",
    "rollout_sample_id": "__rollout_sample_id__",
    "status_code": "status_code",
    "termination_reason": "termination_reason",
    "skip_reason": "skip_reason",
    "tool_calls": "tool_calls",
    "tool_results": "tool_results",
    "tool_error": "tool_error",
    "tool_error_info": "tool_error_info",
}
_TOKEN_FIELDS = {
    "value": "values",
    "advantage": "advantages",
    "return": "returns",
    "old_log_probs": "old_log_probs",
    "rollout_log_probs": "rollout_log_probs",
}


def token_level_entries(batch, tokenizer, step, reward_extra_infos=None):
    """Yield one row per training segment without changing the training batch.

    Tool/observation tokens stay in every aligned array, even though their
    response_mask is zero. Only attention-mask padding is removed. Values and
    advantages are the tensors consumed by training, not recomputed diagnostics.
    """
    tensors = batch.batch
    non_tensors = batch.non_tensor_batch
    responses = tensors["responses"]
    size = responses.shape[0]
    response_attention = tensors["attention_mask"][:, tensors["prompts"].shape[-1]:].bool()
    if response_attention.shape != responses.shape or tensors["response_mask"].shape != responses.shape:
        raise ValueError("Rollout dump requires matching response tokens and masks")
    for field in _TOKEN_FIELDS.values():
        if field in tensors and tensors[field].shape != responses.shape:
            raise ValueError(f"Rollout dump {field} must align with responses")

    def metadata(key, row, default=None):
        column = non_tensors.get(key)
        return default if column is None else json_safe(column[row])

    for row in range(size):
        active = response_attention[row]
        ids = responses[row][active].detach().cpu().tolist()
        action_mask = tensors["response_mask"][row][active].int().cpu().tolist()
        prompt_uid = metadata("uid", row)
        request_id = metadata("request_id", row)
        rollout_uid = request_id if request_id is not None else prompt_uid
        reward_model = metadata("reward_model", row, {})
        if isinstance(reward_model, str):
            reward_model = json.loads(reward_model)
        prompt_mask = tensors["attention_mask"][row, :tensors["prompts"].shape[-1]].bool()
        prompt_ids = tensors["prompts"][row][prompt_mask].detach().cpu().tolist()
        entry = {
            "step": step,
            "uid": str(rollout_uid) if rollout_uid is not None else None,
            "prompt_uid": str(prompt_uid) if prompt_uid is not None else None,
            "index": row,
            "length": len(ids),
            "num_assistant_tokens": sum(action_mask),
            "tokens": tokenizer.convert_ids_to_tokens(ids),
            "token_ids": ids,
            "response_mask": action_mask,
            "input": tokenizer.decode(prompt_ids, skip_special_tokens=True),
            "output": tokenizer.decode(ids, skip_special_tokens=True),
            "score": tensors["token_level_scores"][row].sum().item(),
            "gts": reward_model.get("ground_truth") if isinstance(reward_model, dict) else None,
            "response_length": len(ids),
        }
        for name, source in _TOKEN_FIELDS.items():
            entry[name] = tensors[source][row][active].float().cpu().tolist() if source in tensors else None
        for name, source in _METADATA_FIELDS.items():
            value = metadata(source, row)
            if value is not None:
                entry[name] = value
        # Claw also extends rows with per-sample reward outputs (Search: acc).
        # Prefer the batch copy, which follows DP reordering and dummy unpadding.
        for name, column in (reward_extra_infos or {}).items():
            if name in entry:
                continue
            if name in non_tensors:
                entry[name] = metadata(name, row)
            elif isinstance(column, (list, tuple, np.ndarray)) and len(column) == size:
                entry[name] = json_safe(column[row])
        yield entry


def dump_token_level_data(batch, tokenizer, step, dump_path, reward_extra_infos=None):
    path = Path(dump_path) / f"{step}.jsonl"
    write_jsonl(path, token_level_entries(batch, tokenizer, step, reward_extra_infos))
    return path
