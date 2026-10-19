"""Validate external model outputs before passing them to ICBCBench judges.

This module only reads local files. It never rewrites report text, trusts imported
scores, calls providers, or infers missing responses. IDs refer to the original
public task IDs, not the row numbers in a developer's output file.
"""

import hashlib
import json
import math
from pathlib import Path

from evaluation_toolkit.model_clients import resolve_openrouter_model


SCHEMA_VERSION = 1
_SUCCESS_STATUSES = {"complete", "completed", "success", "succeeded", "ok", "done"}
_COMPLETE_REASONS = {"stop", "end_turn", "stop_sequence", "completed"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key in results file")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("Results file contains a nonfinite numeric value")


def _parse_json(text):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _finite_values(value):
    # parse_constant catches NaN/Infinity, but a legal JSON exponent such as
    # 1e999 also overflows a Python float. Check even fields that will be dropped.
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Results file contains a nonfinite numeric value")
    if isinstance(value, dict):
        for item in value.values():
            _finite_values(item)
    elif isinstance(value, list):
        for item in value:
            _finite_values(item)


def _identifier(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value).strip():
        raise ValueError(f"{label} requires a nonempty string or integer task ID")
    return str(value).strip()


def _read_records(path, raw):
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("Results file must be UTF-8 JSON or JSONL") from None
    if not text.strip():
        raise ValueError("Results file is empty")
    if Path(path).suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = _parse_json(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON on results line {line_number}") from None
            if not isinstance(record, dict):
                raise ValueError(f"Results line {line_number} must be a JSON object")
            records.append((None, record))
        return records, "jsonl"
    try:
        content = _parse_json(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid results JSON at line {exc.lineno}; use .jsonl for line-delimited records") from None
    if isinstance(content, dict):
        return list(content.items()), "json_object"
    if isinstance(content, list):
        return [(None, record) for record in content], "json_array"
    raise ValueError("Results must be an ID-keyed JSON object, a JSON array, or JSONL records")


def _complete_record(record, task_id):
    response = record.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"Task {task_id} requires a nonempty textual response")
    for field in ("error", "errors", "generation_error", "citation_mapping_error", "incomplete_details"):
        if record.get(field):
            raise ValueError(f"Task {task_id} contains an output failure ({field})")
    for field in ("failed", "is_error", "truncated", "is_truncated"):
        if field in record and record[field] is not None:
            if not isinstance(record[field], bool) or record[field]:
                raise ValueError(f"Task {task_id} contains an invalid or failed output flag ({field})")
    for field in ("success", "completed"):
        if field in record and record[field] is not None and record[field] is not True:
            raise ValueError(f"Task {task_id} is not a completed successful output ({field})")
    for field in ("status", "generation_status"):
        value = record.get(field)
        if value is not None and (not isinstance(value, str) or value.strip().lower() not in _SUCCESS_STATUSES):
            raise ValueError(f"Task {task_id} has a non-successful output status ({field})")
    for field in ("finish_reason", "stop_reason"):
        value = record.get(field)
        if value is not None and (not isinstance(value, str) or value.strip().lower() not in _COMPLETE_REASONS):
            raise ValueError(f"Task {task_id} has an unfinished or truncated response ({field})")


def _optional_metadata(record, task_id):
    result = {}
    if "usage" in record:
        if record["usage"] is not None and not isinstance(record["usage"], dict):
            raise ValueError(f"Task {task_id} usage must be a JSON object or null")
        result["usage"] = record["usage"]
    for field in ("provider_citations", "provider_raw_citations"):
        if field in record:
            value = record[field]
            if value is not None and not isinstance(value, (list, dict)):
                raise ValueError(f"Task {task_id} {field} must be an array, object, or null")
            result[field] = value
    return result


def load_results(path, *, model, questions, allow_subset=False):
    """Return ``(normalized_outputs, selected_questions, import_metadata)``.

    ``questions`` must contain the official tasks selected for this evaluation.
    Missing tasks are rejected by default; ``allow_subset=True`` explicitly uses
    only supplied tasks, in the same order as ``questions``. Unknown or duplicate
    IDs are always errors. All response strings are preserved exactly, including
    whitespace and citation syntax. Optional model names may use a recognized
    OpenRouter alias of ``model``; other names must match exactly.

    Only model, response, usage, provider_citations and provider_raw_citations are
    retained. Imported scores, judgments and cache fingerprints are discarded.
    Completion flags are validated when supplied and then discarded. Metadata
    contains digests and coverage only, with no response text or credentials.
    """
    if not isinstance(model, str) or not model.strip() or model != model.strip():
        raise ValueError("model must be a nonempty name without surrounding whitespace")
    if not isinstance(allow_subset, bool):
        raise ValueError("allow_subset must be a boolean")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a nonempty list of official task records")
    question_ids = []
    seen_questions = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each question must be a JSON object")
        task_id = _identifier(question.get("id"), "Question")
        if task_id in seen_questions:
            raise ValueError(f"Duplicate question ID {task_id}")
        seen_questions.add(task_id)
        question_ids.append(task_id)

    raw = Path(path).read_bytes()
    records, source_format = _read_records(path, raw)
    if not records:
        raise ValueError("Results file contains no task outputs")
    normalized = {}
    canonical_model = resolve_openrouter_model(model)
    for key, record in records:
        if not isinstance(record, dict):
            raise ValueError("Every task output must be a JSON object with a response")
        _finite_values(record)
        task_id = _identifier(key if key is not None else record.get("id"), "Result")
        if key is not None and "id" in record and _identifier(record["id"], "Result") != task_id:
            raise ValueError(f"Result key and record id disagree for task {task_id}")
        if task_id in normalized:
            raise ValueError(f"Duplicate result task ID {task_id}")
        if task_id not in seen_questions:
            raise ValueError(f"Unknown task ID {task_id}; result IDs must belong to the selected questions")
        supplied_model = record.get("model")
        if "model" in record and (not isinstance(supplied_model, str) or not supplied_model.strip()
                or supplied_model != supplied_model.strip()
                or resolve_openrouter_model(supplied_model) != canonical_model):
            raise ValueError(f"Task {task_id} model does not match the requested model; use one model per import")
        _complete_record(record, task_id)
        normalized[task_id] = {"model": model, "response": record["response"],
                               **_optional_metadata(record, task_id)}

    omitted_ids = [task_id for task_id in question_ids if task_id not in normalized]
    if omitted_ids and not allow_subset:
        raise ValueError(f"Missing {len(omitted_ids)} task output(s): {', '.join(omitted_ids)}; "
                         "provide every selected task or explicitly allow a subset")
    selected_questions = [question for question, task_id in zip(questions, question_ids) if task_id in normalized]
    normalized = {task_id: normalized[task_id] for task_id in question_ids if task_id in normalized}
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    metadata = {"schema_version": SCHEMA_VERSION, "source_format": source_format,
                "source_sha256": hashlib.sha256(raw).hexdigest(), "source_count": len(records),
                "selected_count": len(selected_questions), "omitted_ids": omitted_ids,
                "normalized_fingerprint": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}
    return normalized, selected_questions, metadata
