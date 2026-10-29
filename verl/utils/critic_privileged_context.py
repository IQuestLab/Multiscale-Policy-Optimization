"""User-message content for a critic with task-specific reference criteria.

Render reference text and cache a verified offline training index. This module
does not change chat templates or actor trajectories. The caller supplies the
original task user message, including any context added by the harness.
"""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from string import Formatter
from types import MappingProxyType
from typing import NamedTuple


CRITIC_USER_TEMPLATE_VERSION = "opsd_user_criteria_v2"
CRITIC_USER_TEMPLATE = (
    "{question}\n\n"
    "Task-specific reference criteria:\n"
    "=== Reference Criteria Begin ===\n"
    "{rubric}\n"
    "=== Reference Criteria End ==="
)


def resolve_critic_user_template(template: str | None = None) -> str:
    """Resolve and validate a user-body template without interpreting task text.

    Require the original question first, with both fields present exactly once.
    Formatting/conversion of fields is disallowed so their contents stay intact.
    """
    if template is None:
        template = CRITIC_USER_TEMPLATE
    if not isinstance(template, str):
        raise TypeError("critic user template must be a string or None")
    try:
        parts = list(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"Invalid critic user template: {exc}") from exc

    fields = [field for _, field, _, _ in parts if field is not None]
    if Counter(fields) != Counter({"question": 1, "rubric": 1}):
        raise ValueError("critic user template must contain {question} and {rubric} exactly once")
    if not template.startswith("{question}"):
        raise ValueError("critic user template must start with {question}")
    if any(spec or conversion for _, _, spec, conversion in parts):
        raise ValueError("critic user template must not format or convert its fields")
    return template


def render_critic_user_content(
    *, question: str, rubric: str, template: str | None = None
) -> str:
    """Render the task user body; leave braces, whitespace and Unicode intact.

    Values are substituted once, so a literal ``{question}`` inside a rubric is
    data, not a second template. No reasoning/generation instruction is added.
    """
    for name, value in (("question", question), ("rubric", rubric)):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value.strip():
            raise ValueError(f"{name} must not be empty")
    return resolve_critic_user_template(template).format(question=question, rubric=rubric)


def critic_user_template_fingerprint(template: str | None = None) -> dict[str, str]:
    """Identity to store with VP/PPO checkpoints, including template overrides."""
    resolved = resolve_critic_user_template(template)
    return {
        "template_version": CRITIC_USER_TEMPLATE_VERSION,
        "template_sha256": hashlib.sha256(resolved.encode("utf-8")).hexdigest(),
    }


class CriticContextEntry(NamedTuple):
    """Immutable reference data; token IDs describe only the appended suffix."""

    task_id: str
    context_text: str
    suffix_text: str
    token_ids: tuple[int, ...]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tokenizer_spec(spec: dict) -> dict:
    # Padding/truncation are invocation state, not vocabulary/encoding identity.
    # Clear them on a private clone so caching cannot truncate or mutate actor state.
    return {**spec, "truncation": None, "padding": None}


class CriticContextIndex:
    """Load once, validate, and cache references for the supplied training IDs.

    Uses the training tokenizer directly; task directories are never read.
    A filtered subset of the exported training tasks may be requested. Missing
    tasks fail; batch lookups may repeat IDs and preserve the supplied order.

    These suffix tokens are not yet a full critic input: the integration layer
    must locate the task user message and verify token boundaries/value alignment.
    Instantiate this class only when the feature is enabled in training.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        index_path: str | Path,
        *,
        training_task_ids,
        tokenizer,
        max_tokens: int,
        template: str | None = None,
    ):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("critic context max_tokens must be a positive integer")
        resolved_template = resolve_critic_user_template(template)
        if isinstance(training_task_ids, (str, bytes)):
            raise ValueError("training_task_ids must be a collection of task IDs")
        requested = list(training_task_ids)
        if not requested or any(not isinstance(k, str) or not k.strip() for k in requested):
            raise ValueError("training_task_ids must contain nonempty strings")
        if len(set(requested)) != len(requested):
            raise ValueError("Duplicate training task_id")

        index_path = Path(index_path)
        report = json.loads((index_path.parent / "extraction_report.json").read_text(encoding="utf-8"))
        if report.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported critic context report schema_version")
        if report.get("ready") is not True or report.get("errors") != []:
            raise ValueError("Critic context export is not ready; resolve extraction errors first")
        if index_path.name != "train_contexts.jsonl" or report.get("index_file") != index_path.name:
            raise ValueError("Expected the published train_contexts.jsonl index")
        raw = index_path.read_bytes()
        index_hash = _sha256(raw)
        if report.get("index_sha256") != index_hash:
            raise ValueError("Critic context index SHA256 mismatch")

        records = {}
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            row = json.loads(line)
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"Invalid task_id at index line {line_number}")
            if task_id in records:
                raise ValueError(f"Duplicate context task_id: {task_id}")
            if row.get("status") != "complete" or row.get("unresolved") != []:
                raise ValueError(f"Unresolved context: {task_id}")
            if not isinstance(row.get("context_text"), str) or not row["context_text"].strip():
                raise ValueError(f"Empty context_text: {task_id}")
            records[task_id] = row
        if len(records) != report.get("input_rows") or len(records) != report.get("exported_rows"):
            raise ValueError("Critic context row count differs from extraction report")
        missing = sorted(set(requested) - records.keys())
        if missing:
            raise ValueError(f"Missing training task context: {', '.join(missing)}")

        # Encode with the actual training tokenizer, without a second tokenizer
        # file or per-task provenance checks. Keep only a backend fingerprint
        # for future checkpoint matching; never change the shared tokenizer.
        backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
        backend_spec = _tokenizer_spec(json.loads(backend.to_str()))
        serialized_backend = json.dumps(backend_spec, ensure_ascii=False, sort_keys=True)
        encoder = type(backend).from_str(serialized_backend)

        entries = {}
        for task_id in requested:
            row = records[task_id]
            suffix = resolved_template.format(question="", rubric=row["context_text"])
            ids = tuple(encoder.encode(suffix, add_special_tokens=False).ids)
            if not ids:
                raise ValueError(f"Tokenizer produced empty context: {task_id}")
            if len(ids) > max_tokens:
                raise ValueError(f"Critic context overflow: task_id={task_id}, tokens={len(ids)}, budget={max_tokens}")
            entries[task_id] = CriticContextEntry(
                task_id, row["context_text"], suffix, ids
            )
        self._entries = MappingProxyType(entries)
        self._metadata = {
            "schema_version": self.SCHEMA_VERSION,
            "index_sha256": index_hash,
            "parquet_sha256": report["parquet_sha256"],
            "tokenizer_backend_sha256": _sha256(serialized_backend.encode("utf-8")),
            "max_tokens": max_tokens,
            "loaded_task_count": len(entries),
            **critic_user_template_fingerprint(resolved_template),
        }
        self._warnings = tuple(copy.deepcopy(report.get("warnings", [])))

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def metadata(self) -> dict:
        """Return an independent snapshot for future checkpoint consistency checks."""
        return dict(self._metadata)

    @property
    def warnings(self) -> tuple:
        return copy.deepcopy(self._warnings)

    def get(self, task_id: str) -> CriticContextEntry:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("A nonempty task_id is required for critic context lookup")
        try:
            return self._entries[task_id]
        except KeyError as exc:
            raise KeyError(f"No cached training critic context for task_id={task_id}") from exc

    def get_many(self, task_ids) -> tuple[CriticContextEntry, ...]:
        if isinstance(task_ids, (str, bytes)):
            raise ValueError("Batch task_ids must be a collection, not a single string")
        return tuple(self.get(task_id) for task_id in task_ids)
