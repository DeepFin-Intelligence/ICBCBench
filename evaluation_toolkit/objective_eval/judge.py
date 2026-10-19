"""Judge objective answers, save successful judgments, and report coverage."""
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import copy
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from pydantic import BaseModel, conint
from tqdm import tqdm

from evaluation_toolkit import safe_json_loads
from evaluation_toolkit.model_clients import OPENROUTER, resolve_openrouter_model, require_complete_chat
from evaluation_toolkit.objective_eval.metrics import (
    calib_err, dump_metrics as write_metrics, valid_judge_result, prediction_model_name,
    judge_fingerprint as compute_judge_fingerprint,
)
from evaluation_toolkit.objective_eval.predict import load_questions, positive_int, write_json_atomic
from evaluation_toolkit.dataset_loading import add_hf_dataset_arguments

client = OPENROUTER
with (Path(__file__).parent / "prompts.json").open(encoding="utf-8") as handle:
    prompt_templates = json.load(handle)


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: conint(strict=True, ge=0, le=100)


def extract_answer(question, correct_answer, response, language):
    prompt = prompt_templates["judge"][language].format(
        question=question, correct_answer=correct_answer, response=response)
    try:
        result = client.chat.completions.create(
            model=resolve_openrouter_model(args.judge),
            max_completion_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        require_complete_chat(result)
        payload = safe_json_loads(result.choices[0].message.content)
        if not isinstance(payload, dict):
            raise ValueError("Judge must return a JSON object")
        parsed = ExtractedAnswer(**payload)
        return {
            "correct_answer": correct_answer,
            "model_answer": parsed.extracted_final_answer,
            "reasoning": parsed.reasoning,
            "correct": parsed.correct,
            "confidence": parsed.confidence,
        }
    except Exception as error:
        # Errors stay within their task. A worker must never exit the process and
        # discard successful judgments from other workers.
        print(f"Judgment failed: {error}")
        return None


def judge_fingerprint(question, prediction):
    return compute_judge_fingerprint(question, prediction, args.judge, prompt_templates)


def add_judge_result(question, predictions):
    unique_id = question["id"]
    prediction = copy.deepcopy(predictions[unique_id])
    result = extract_answer(question["question"], question["answer"], prediction["response"], question["language"])
    if result is None:
        return None, None
    prediction["judge_model"] = args.judge
    prediction["judge_result"] = result
    prediction["judge_fingerprint"] = judge_fingerprint(question, prediction)
    return unique_id, prediction


def judge_all_responses(questions, predictions, on_result=None):
    results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(add_judge_result, question, predictions): question for question in questions}
        for future in tqdm(as_completed(futures), total=len(questions)):
            try:
                result = future.result()
            except Exception as error:
                print(f"Judging failed: {error}")
                results.append((None, None))
            else:
                results.append(result)
                if result[0] is not None and on_result is not None:
                    on_result(result)
    return results


def dump_metrics(predictions, n, *, model_name=None):
    if model_name is None:
        model_name = prediction_model_name(predictions, Path(args.predictions).stem)
    output = getattr(args, "metrics_csv", None) or _project_root / "eval_result/objective_eval/evaluation_results.csv"
    return write_metrics(model_name, args.judge, predictions, n, output)


def _run(args):
    if args.num_workers < 1:
        raise ValueError("num_workers must be 1 or greater")
    if not args.local_dataset and not getattr(args, "dataset", None):
        raise ValueError("Provide --local_dataset or --dataset")
    if not args.predictions:
        raise ValueError("Provide --predictions")
    questions = load_questions(args, ("id", "language", "question", "answer"))
    output_dir = Path(getattr(args, "output_dir", None) or _project_root / "eval_result/objective_eval/judged_results" / f"judge_{os.path.basename(args.judge)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"judged_{os.path.basename(args.predictions)}"
    with open(args.predictions, encoding="utf-8-sig") as handle:
        predictions = json.load(handle)
    if not isinstance(predictions, dict):
        raise ValueError("Predictions must be a JSON object")
    model_name = prediction_model_name(predictions, Path(args.predictions).stem)
    if output_path.exists():
        with output_path.open(encoding="utf-8-sig") as handle:
            cache = json.load(handle)
        if not isinstance(cache, dict):
            raise ValueError("Judgment cache must be a JSON object")
        for record in cache.values():
            if (isinstance(record, dict) and isinstance(record.get("judge_model"), str)
                    and resolve_openrouter_model(record["judge_model"]) != resolve_openrouter_model(args.judge)):
                raise ValueError("Judgment output is already used by a different judge; select a separate output_dir")
    else:
        cache = {}
    judged, pending, missing = {}, [], 0
    for question in questions:
        unique_id = question["id"]
        prediction = predictions.get(unique_id)
        if not (isinstance(prediction, dict) and isinstance(prediction.get("response"), str) and prediction["response"].strip()):
            missing += 1
            continue
        previous = cache.get(unique_id)
        if (isinstance(previous, dict) and valid_judge_result(previous.get("judge_result"))
                and all(field in previous["judge_result"] for field in ("correct_answer", "model_answer", "reasoning"))
                and previous.get("judge_fingerprint") == judge_fingerprint(question, prediction)):
            judged[unique_id] = previous
        else:
            pending.append(question)
    def save_result(result):
        unique_id, prediction = result
        judged[unique_id] = prediction
        write_json_atomic(output_path, judged)

    write_json_atomic(output_path, judged)
    try:
        judge_all_responses(pending, predictions, on_result=save_result)
    except BaseException:
        write_json_atomic(output_path, judged)
        raise
    summary = dump_metrics(judged, len(questions), model_name=model_name)
    if missing:
        print(f"Judging incomplete: {missing} question(s) have no valid prediction. Generate the missing predictions and rerun.")
    return int(summary["Evaluated"] != summary["Total"])


def main(options):
    globals()["args"] = options
    try:
        return _run(options)
    except Exception as error:
        print(f"Judging failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--dataset", help="Hugging Face dataset")
    sources.add_argument("--local_dataset", default=None, help="Local JSON dataset file path")
    add_hf_dataset_arguments(parser)
    parser.add_argument("--predictions", required=True, help="Model predictions JSON file")
    parser.add_argument("--num_workers", type=positive_int, default=10, help="Concurrent requests")
    parser.add_argument("--judge", default="gpt-5.4", help="Judge model")
    parser.add_argument("--output_dir", default=None, help="Override judged-result directory")
    parser.add_argument("--metrics_csv", default=None, help="Override metrics CSV path")
    sys.exit(main(parser.parse_args()))
