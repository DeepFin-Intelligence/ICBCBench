"""Small persistence helpers shared by subjective evaluation entry points."""

import json
import os
import tempfile
import hashlib
from pathlib import Path
from evaluation_toolkit.model_clients import resolve_openrouter_model


def fingerprint(payload):
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_questions(records, *, require_question=True):
    """Validate and normalize question IDs/languages without changing source files."""
    if not isinstance(records, list) or not records:
        raise ValueError("Question data must be a nonempty JSON array")
    languages = {"en": "en", "english": "en", "英文": "en", "英语": "en",
                 "zh": "zh", "chinese": "zh", "中文": "zh"}
    normalized = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each question must be a JSON object")
        raw_id = record.get("id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)) or not str(raw_id).strip():
            raise ValueError("Each question requires a nonempty string or integer ID")
        task_id = str(raw_id).strip()
        if task_id in seen:
            raise ValueError(f"Duplicate question ID {task_id}")
        seen.add(task_id)
        raw_language = record.get("language")
        language = languages.get(raw_language.strip().lower()) if isinstance(raw_language, str) else None
        if language is None:
            raise ValueError(f"Question {task_id} has an invalid language; use en or zh")
        if require_question:
            body = (record.get("question_en") or record.get("question")) if language == "en" else record.get("question")
            if not isinstance(body, str) or not body.strip():
                raise ValueError(f"Question {task_id} requires nonempty question text")
        normalized.append(dict(record, id=task_id, language=language))
    return normalized


def atomic_save(path, data, *, jsonl=False):
    """Replace a result file only after its complete new contents are on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            if jsonl:
                for record in data:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            else:
                json.dump(data, stream, indent=4, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def valid_report(record, model):
    return (
        isinstance(record, dict)
        and isinstance(record.get("model"), str)
        and resolve_openrouter_model(record["model"]) == resolve_openrouter_model(model)
        and isinstance(record.get("response"), str)
        and bool(record["response"].strip())
    )
