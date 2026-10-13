"""Objective metrics computed against the selected dataset, including coverage."""
import argparse
import json
import math
import hashlib
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from evaluation_toolkit.model_clients import resolve_openrouter_model


def judge_fingerprint(question, prediction, judge_model, prompt_templates=None):
    if prompt_templates is None:
        with (Path(__file__).parent / "prompts.json").open(encoding="utf-8") as handle:
            prompt_templates = json.load(handle)
    data = {"question": question, "response": prediction["response"],
            "prediction_model": prediction.get("model"),
            "judge": resolve_openrouter_model(judge_model),
            "judge_prompt": prompt_templates["judge"][question["language"]], "schema_version": 2}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def current_judged_record(record, question, judge_model):
    """A schema-valid judgment is reusable only for its original task and judge."""
    if not (isinstance(record, dict)
            and isinstance(record.get("response"), str) and record["response"].strip()
            and valid_judge_result(record.get("judge_result"))
            and isinstance(record.get("judge_model"), str)
            and resolve_openrouter_model(record["judge_model"]) == resolve_openrouter_model(judge_model)):
        return False
    return record.get("judge_fingerprint") == judge_fingerprint(question, record, judge_model)


def validate_questions(questions, required_fields=("id", "language")):
    if not isinstance(questions, list) or not questions:
        raise ValueError("Dataset must be a nonempty list of questions")
    result, ids = [], set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Every question must be an object")
        for field in required_fields:
            if field not in question or question[field] is None:
                raise ValueError(f"Question is missing required field: {field}")
        raw_id = question["id"]
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            raise ValueError("Question IDs must be strings or integers")
        unique_id = str(raw_id).strip()
        if not unique_id or unique_id in ids:
            raise ValueError(f"Question IDs must be unique and nonempty: {unique_id!r}")
        ids.add(unique_id)
        language = normalize_language(question["language"])
        if "question" in required_fields and (not isinstance(question["question"], str) or not question["question"].strip()):
            raise ValueError(f"Question {unique_id} must contain nonempty question text")
        result.append({**question, "id": unique_id, "language": language})
    return result


def normalize_language(language):
    normalized = str(language).strip().lower()
    if normalized in ("zh", "chinese", "中文"):
        return "zh"
    if normalized in ("en", "english", "英文", "英语"):
        return "en"
    raise ValueError(f"Unsupported question language: {language!r}")


def is_multimodal(question):
    modality = str(question.get("modality") or "").lower().replace("-", "").replace("_", "").replace(" ", "")
    return modality == "multimodal" or bool(question.get("attachment"))


def valid_judge_result(result):
    """Reject malformed model output and invalid cached judgments."""
    return (isinstance(result, dict)
            and result.get("correct") in ("yes", "no")
            and type(result.get("confidence")) is int
            and 0 <= result["confidence"] <= 100)


def calib_err(confidence, correct, p="2", beta=10):
    confidence, correct = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
    if len(confidence) != len(correct):
        raise ValueError("Confidence and correctness must have matching lengths")
    if not len(confidence):
        return float("nan")
    if beta < 1:
        raise ValueError("Calibration bin size must be positive")
    idxs = np.argsort(confidence, kind="stable")
    confidence, correct = confidence[idxs], correct[idxs]
    if len(confidence) < beta:
        bins = [(0, len(confidence))]
    else:
        bins = [(i * beta, (i + 1) * beta) for i in range(len(confidence) // beta)]
        bins[-1] = (bins[-1][0], len(confidence))
    error = 0.0
    for start, end in bins:
        difference = abs(float(np.mean(confidence[start:end])) - float(np.mean(correct[start:end])))
        weight = (end - start) / len(confidence)
        if p == "2":
            error += weight * difference ** 2
        elif p == "1":
            error += weight * difference
        elif p in ("infty", "infinity", "max"):
            error = max(error, difference)
        else:
            raise ValueError("p must be '1', '2', or 'infty'")
    return math.sqrt(error) if p == "2" else error


def compute_metrics(judged_results, total):
    if total < 1:
        raise ValueError("The selected dataset contains no questions")
    # Give equal-confidence answers a stable order across concurrent completion
    # and cache loading, so the calibration bins are reproducible.
    entries = (judged_results[key] for key in sorted(judged_results, key=str))
    judgments = [entry["judge_result"] for entry in entries
                 if isinstance(entry, dict) and valid_judge_result(entry.get("judge_result"))]
    evaluated = len(judgments)
    if evaluated > total:
        raise ValueError("Judged results contain IDs outside the selected dataset")
    correctness = np.array([result["correct"] == "yes" for result in judgments])
    confidence = np.array([result["confidence"] / 100 for result in judgments])
    accuracy = 100 * int(np.sum(correctness)) / total
    return {
        "Accuracy (%)": round(accuracy, 2),
        "CalibrationError (%)": round(100 * calib_err(confidence, correctness), 2) if evaluated else None,
        "Total": total,
        "Evaluated": evaluated,
        "Coverage (%)": round(100 * evaluated / total, 2),
    }


def dump_metrics(prediction_model_name, judge_model, judged_results, n, results_csv_path):
    summary = compute_metrics(judged_results, n)
    print(f"Accuracy: {summary['Accuracy (%)']}% | Total: {n} | Evaluated: {summary['Evaluated']} | Coverage: {summary['Coverage (%)']}%")
    print(f"Calibration Error: {summary['CalibrationError (%)']}")
    if summary["Evaluated"] != n:
        print("Evaluation incomplete: missing or invalid judgments remain; accuracy uses the selected dataset as denominator.")
    row = {"Model": prediction_model_name, **summary, "Judge": judge_model}
    results_csv_path = Path(results_csv_path)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if results_csv_path.exists():
        existing = pd.read_csv(results_csv_path)
        if "Model" not in existing:
            raise ValueError(f"Metrics CSV is missing Model column: {results_csv_path}")
        same_judge = ((existing["Judge"] == judge_model) | existing["Judge"].isna()) if "Judge" in existing else True
        existing = existing[~((existing["Model"] == prediction_model_name) & same_judge)]
    else:
        existing = pd.DataFrame(columns=row.keys())
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(results_csv_path, index=False, encoding="utf-8")
    print(f"Results saved to {results_csv_path}")
    return summary


def get_multimodal_ids_from_json(json_path):
    with open(json_path, encoding="utf-8-sig") as handle:
        questions = validate_questions(json.load(handle))
    return [question["id"] for question in questions if is_multimodal(question)]


def dump_all_metrics(judge_model, language_filter, multimodal_ids=None,
                     judged_dir=None, query_file=None, results_dir=None):
    project_root = Path(__file__).resolve().parents[2]
    judged_dir = Path(judged_dir or project_root / "eval_result/objective_eval/judged_results" / f"judge_{os.path.basename(judge_model)}")
    query_file = Path(query_file or project_root / "data/objective_questions_public_80.json")
    results_dir = Path(results_dir or project_root / "eval_result/objective_eval")
    with query_file.open(encoding="utf-8-sig") as handle:
        questions = validate_questions(json.load(handle))
    exclusions = {str(unique_id) for unique_id in multimodal_ids} if multimodal_ids is not None else set()
    selected_questions = {question["id"]: question for question in questions
                    if (not language_filter or question["language"] == language_filter)
                    and question["id"] not in exclusions}
    selected_ids = set(selected_questions)
    if not selected_ids:
        print(f"No questions selected for language={language_filter}, text_only={multimodal_ids is not None}; skipped.")
        return 0
    files = sorted(judged_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No judged-result JSON files found in {judged_dir}")
    suffix = "_".join(part for part in (language_filter, "text_only" if multimodal_ids is not None else None) if part)
    output = results_dir / (f"{suffix}_evaluation_results.csv" if suffix else "evaluation_results.csv")
    incomplete = False
    for path in files:
        with path.open(encoding="utf-8-sig") as handle:
            results = json.load(handle)
        if not isinstance(results, dict):
            raise ValueError(f"Judged results must be an object: {path}")
        selected = {str(key): entry for key, entry in results.items()
                    if str(key) in selected_ids
                    and current_judged_record(entry, selected_questions[str(key)], judge_model)}
        stale = sum(str(key) in selected_ids and str(key) not in selected for key in results)
        if stale:
            print(f"{path.name}: ignored {stale} invalid or stale judgment(s); rerun judge.py for the current dataset and judge.")
        model_name = path.stem.removeprefix("judged_")
        summary = dump_metrics(model_name, judge_model, selected, len(selected_ids), output)
        incomplete |= summary["Evaluated"] != summary["Total"]
    return int(incomplete)


def main(args):
    try:
        multimodal_ids = get_multimodal_ids_from_json(args.query_file)
        status = 0
        for language, exclusions in [(None, None), ("zh", None), ("en", None), ("zh", multimodal_ids), ("en", multimodal_ids)]:
            status |= dump_all_metrics(args.judge, language, exclusions, args.judged_dir, args.query_file, args.output_dir)
        return status
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Metrics failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute objective metrics and coverage for the selected dataset")
    parser.add_argument("--judge", default="gpt-5.4")
    parser.add_argument("--query_file", default=str(Path(__file__).resolve().parents[2] / "data/objective_questions_public_80.json"))
    parser.add_argument("--judged_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    sys.exit(main(parser.parse_args()))
