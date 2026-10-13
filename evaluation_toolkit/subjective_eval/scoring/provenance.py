"""Match score inputs to the exact questions and generated report texts."""

import re

from evaluation_toolkit.subjective_eval.runtime import fingerprint, validate_questions
from evaluation_toolkit.model_clients import resolve_openrouter_model


def valid_fingerprint(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def question_fingerprint(question):
    return fingerprint(validate_questions([question], require_question=False)[0])


def task_set_fingerprint(questions):
    normalized = validate_questions(questions, require_question=False)
    return fingerprint(sorted(
        [{"id": question["id"], "question_fingerprint": question_fingerprint(question)}
         for question in normalized], key=lambda item: item["id"]))


def report_fingerprint(task_id, report):
    if (not isinstance(report, dict) or not isinstance(report.get("model"), str)
            or not report["model"].strip() or not isinstance(report.get("response"), str)
            or not report["response"].strip()):
        raise ValueError(f"Cannot verify report provenance for task {task_id}")
    return fingerprint({"id": str(task_id), "model": resolve_openrouter_model(report["model"].strip().lower()),
                        "response": report["response"]})


def report_set_fingerprint(task_fingerprints):
    if not task_fingerprints or any(not valid_fingerprint(value) for value in task_fingerprints.values()):
        raise ValueError("Report provenance requires one valid fingerprint per task")
    return fingerprint(sorted(
        [{"id": str(identifier), "report_fingerprint": value}
         for identifier, value in task_fingerprints.items()], key=lambda item: item["id"]))
