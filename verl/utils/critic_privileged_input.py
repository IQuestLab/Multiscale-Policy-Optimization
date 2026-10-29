"""Qwen first-user edits and isolated critic tensors, independent of Megatron."""

import json
from typing import NamedTuple

import torch


class CriticPromptEdit(NamedTuple):
    """Coordinates relative to the original sequence after its leading padding."""

    task_id: str
    body_start: int
    body_end: int
    replacement_ids: tuple[int, ...]
    added_tokens: int


class QwenCriticPromptEditor:
    """Append task criteria to the first user body, including summary/subtask prompts."""

    def __init__(self, tokenizer):
        backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
        if not callable(getattr(backend, "to_str", None)):
            raise ValueError("Qwen critic editing requires a fast tokenizer")
        spec = json.loads(backend.to_str())
        spec.update(truncation=None, padding=None)
        self._encoder = type(backend).from_str(json.dumps(spec, ensure_ascii=False))
        self._start = self._special_id("<|im_start|>")
        self._end = self._special_id("<|im_end|>")
        self._user_header = (self._start, *self._encode("user\n"))

    def _encode(self, text):
        return tuple(self._encoder.encode(text, add_special_tokens=False).ids)

    def _special_id(self, text):
        ids = self._encode(text)
        if len(ids) != 1 or self._encoder.decode(list(ids), skip_special_tokens=False) != text:
            raise ValueError(f"Unsupported chat template token: {text}")
        return ids[0]

    def prepare(self, prompt_ids, *, task_id: str, entry, max_added_tokens: int) -> CriticPromptEdit:
        if task_id != entry.task_id:
            raise ValueError(f"Context task_id mismatch: expected {task_id}, got {entry.task_id}")
        if type(max_added_tokens) is not int or max_added_tokens <= 0:
            raise ValueError("max_added_tokens must be a positive integer")
        ids = tuple(prompt_ids)
        # The first user may contain the task, a compacted summary or a subtask.
        # Its original text is preserved; task_id selects the cached reference.
        for start in range(len(ids)):
            if ids[start:start + len(self._user_header)] != self._user_header:
                continue
            body_start = start + len(self._user_header)
            try:
                body_end = ids.index(self._end, body_start)
            except ValueError as exc:
                raise ValueError(f"Incomplete first user message: {task_id}") from exc
            if self._start in ids[body_start:body_end]:
                raise ValueError(f"Invalid first user boundary: {task_id}")
            body = self._encoder.decode(list(ids[body_start:body_end]), skip_special_tokens=False)
            break
        else:
            raise ValueError(f"Missing user message: {task_id}")
        start, end = body_start, body_end
        # Re-encode only the modified body: joining independently encoded suffix
        # tokens may miss a BPE merge across the original user body's end boundary.
        replacement = self._encode(body + entry.suffix_text)
        if self._start in replacement or self._end in replacement:
            raise ValueError(f"Reference would introduce chat message boundaries: {task_id}")
        added = len(replacement) - (end - start)
        if added <= 0 or added > max_added_tokens:
            raise ValueError(f"Critic prompt growth out of budget: task_id={task_id}, added={added}, budget={max_added_tokens}")
        return CriticPromptEdit(task_id, start, end, replacement, added)


def build_privileged_critic_batch(
    batch, *, editor, task_ids, entries, max_added_tokens: int,
    pad_token_id: int, sequence_alignment: int = 1,
):
    """Build from the ORIGINAL actor batch on every value forward/update.

    Prepare and apply edits together, so no stale edit or prompt hash is needed.
    A rollout may pack its initial messages into the response tensor with zero
    action mask. Edit before any action; extra capacity/alignment goes on the left,
    preserving action positions in the existing [:, -response_length-1:-1] slice.
    Do not feed the returned
    critic batch back into this function; training must retain its original batch.
    """
    ids, mask, pos, responses = (batch[k] for k in ("input_ids", "attention_mask", "position_ids", "responses"))
    if ids.ndim != 2 or mask.shape != ids.shape or pos.shape != ids.shape or responses.ndim != 2:
        raise ValueError("Expected matching text-only 2-D input, mask and position tensors")
    size, length = ids.shape
    response_length = responses.shape[1]
    prompt_length = length - response_length
    if not size or response_length <= 0 or prompt_length <= 0 or sequence_alignment <= 0:
        raise ValueError("Expected nonempty prompt/response and positive sequence_alignment")
    if not len(task_ids) == len(entries) == size:
        raise ValueError("Critic batch and task references must have matching lengths")
    if not torch.equal(ids[:, -response_length:], responses):
        raise ValueError("Responses do not match the original input tail")

    edits, starts = [], []
    for row, (task_id, entry) in enumerate(zip(task_ids, entries, strict=True)):
        left = prompt_length - int(mask[row, :prompt_length].sum())
        edit = editor.prepare(ids[row, left:].tolist(), task_id=task_id,
                              entry=entry, max_added_tokens=max_added_tokens)
        # Prefix messages can lie in either tensor. No loss-bearing action may
        # precede the edit, since only positions after it retain tail alignment.
        end_in_response = max(0, left + edit.body_end - prompt_length)
        if batch["response_mask"][row, :end_in_response].any():
            raise ValueError(f"First user message overlaps or follows action tokens: {task_id}")
        edits.append(edit)
        starts.append(left)
    needed = length + max(edit.added_tokens for edit in edits)
    new_length = ((needed + sequence_alignment - 1) // sequence_alignment) * sequence_alignment
    extra = new_length - length
    new_ids = ids.new_full((size, new_length), pad_token_id)
    new_mask = mask.new_zeros((size, new_length))
    for row, (left, edit) in enumerate(zip(starts, edits, strict=True)):
        start, end = left + edit.body_start, left + edit.body_end
        offset = extra - edit.added_tokens
        replacement = ids.new_tensor(edit.replacement_ids)
        new_ids[row, offset:] = torch.cat((ids[row, :start], replacement, ids[row, end:]))
        new_mask[row, offset:] = torch.cat((mask[row, :start], mask.new_ones(len(replacement)), mask[row, end:]))
    replaced = {"input_ids", "attention_mask", "position_ids", "prompts", "responses"}
    result = {key: value.clone() for key, value in batch.items() if key not in replaced}
    result.update(input_ids=new_ids, attention_mask=new_mask,
                  responses=new_ids[:, -response_length:].clone(),
                  position_ids=(new_mask.long().cumsum(-1) - 1).clamp_min(0).to(pos.dtype))
    if "prompts" in batch:
        result["prompts"] = new_ids[:, :-response_length].clone()
    return result


def dump_privileged_critic_inputs(
    actor_batch, critic_batch, *, tokenizer, task_ids, entries, output_dir, sample_count, request_ids=None,
    source="trainer_pre_critic_rpc",
):
    """Save pre-RPC tensors and chats; sample_count=-1 saves every batch row.

    Repeated task IDs are separate rollout samples. No model forward is implied.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    keys = ("input_ids", "attention_mask", "position_ids", "responses", "response_mask", "values", "returns")
    count = len(task_ids) if sample_count == -1 else min(sample_count, len(task_ids))
    for row in range(count):
        directory = output_dir / f"sample_{row:03d}"
        directory.mkdir(exist_ok=True)
        tensors = {
            name: {key: batch[key][row].detach().cpu().clone() for key in keys if key in batch}
            for name, batch in (("actor", actor_batch), ("critic", critic_batch))
        }
        texts = {}
        for name, data in tensors.items():
            ids = data["input_ids"][data["attention_mask"].bool()].tolist()
            texts[name] = tokenizer.decode(ids, skip_special_tokens=False)
            (directory / f"{name}_chat.txt").write_text(texts[name], encoding="utf-8")
        torch.save(tensors, directory / "tensors.pt")
        before, after = tensors["actor"], tensors["critic"]
        active = before["response_mask"].bool()
        width = before["responses"].shape[0]
        added = int(after["attention_mask"].sum() - before["attention_mask"].sum())
        body_start = texts["actor"].index("<|im_start|>user\n") + len("<|im_start|>user\n")
        body_end = texts["actor"].index("<|im_end|>", body_start)
        suffix = entries[row].suffix_text
        expected = texts["actor"][:body_end] + suffix + texts["actor"][body_end:]
        checks = {
            "only_first_user_suffix_changed": texts["critic"] == expected,
            "response_mask_unchanged": torch.equal(before["response_mask"], after["response_mask"]),
            "action_tokens_unchanged": torch.equal(before["responses"][active], after["responses"][active]),
            "value_predecessor_tokens_aligned": torch.equal(
                before["input_ids"][-width-1:-1][active], after["input_ids"][-width-1:-1][active]),
            "value_positions_shifted_by_rubric_length": torch.equal(
                before["position_ids"][-width-1:-1][active] + added,
                after["position_ids"][-width-1:-1][active]),
        }
        for key in ("values", "returns"):
            if key in before:
                checks[f"{key}_unchanged"] = torch.equal(before[key], after[key])
        report = {
            "source": source, "sample_index": row, "task_id": task_ids[row],
            "request_id": None if request_ids is None else request_ids[row],
            "added_tokens": added, "action_tokens": int(active.sum()),
            "actor_first_user_body_chars": [body_start, body_end],
            "critic_rubric_suffix_chars": [body_end, body_end + len(suffix)],
            "checks": checks, "passed": all(checks.values()),
        }
        (directory / "metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reports.append(report)
    if not all(report["passed"] for report in reports):
        raise ValueError(f"Critic input dump alignment check failed: {output_dir}")
    return reports
