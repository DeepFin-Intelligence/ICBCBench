import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p.name != "evaluation_toolkit" and _p.parent != _p:
    _p = _p.parent
if _p.name == "evaluation_toolkit":
    _root = _p.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import json
import argparse
import time
import threading


from tqdm import tqdm

from evaluation_toolkit.utils import load_local_dataset
from evaluation_toolkit.dataset_loading import add_hf_dataset_arguments, load_hf_dataset
from evaluation_toolkit.subjective_eval.prompts import *
from evaluation_toolkit.model_clients import *
from evaluation_toolkit.DR_clients import *
from evaluation_toolkit.subjective_eval.runtime import atomic_save, valid_report, fingerprint, validate_questions


client = OPENROUTER

def format_message(topic):
    language = "zh" if topic["language"] in ["zh", "chinese", "中文"] else "en"
    user_topic = (topic.get('question_en') or topic['question']) if language == 'en' else topic['question']
    content = f"用户的研究请求：{user_topic}" if language == "zh" else f"User's Research Query: {user_topic}"
    SYSTEM_PROMPT = WRITING_PROMPT_ZH if language == "zh" else WRITING_PROMPT_EN

    system_role = "model" if "gemini-deep-research" in args.model else "system"  # o1 no sys prompt
    messages = [
        {"role": system_role, "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    return messages


def request_fingerprint(topic):
    return fingerprint({"model": resolve_openrouter_model(args.model), "messages": format_message(topic),
                        "max_tokens": getattr(args, "max_completion_tokens", None), "temperature": getattr(args, "temperature", 0.0),
                        "schema_version": 2})


def attempt_query(topic):
    messages = format_message(topic)
    metadata = None

    try:
        dr_client = create_dr_client(args.model)
        if dr_client is not None:
            dr_client.cancel_event = getattr(args, "research_cancel_event", None)
            response = dr_client.run_research(system_prompt=messages[0]["content"], dr_task=messages[1]["content"])
            content = response.get("content")
            tokens = response.get("usage")
            metadata = {"provider_citations": response.get("citations", []),
                        "provider_raw_citations": response.get("raw_citations", [])}
            if response.get("citation_mapping_error"):
                metadata["citation_mapping_error"] = response["citation_mapping_error"]
        else:
            response = client.chat.completions.create(
                model=resolve_openrouter_model(args.model),
                max_tokens=args.max_completion_tokens,
                temperature=args.temperature,
                messages=messages,
                stream=False,
                extra_body={"thinking": {"type": "enabled"}} if args.model in ["deepseek-v4-pro", "kimi-k2.6"] else {"reasoning": {"enabled": True}}
            )

            require_complete_chat(response)
            content = response.choices[0].message.content
            tokens = response.usage.model_dump() if response.usage else None
    except Exception as e:
        print("\nError:", e)
        return None

    if not isinstance(content, str) or not content.strip():
        return None

    if tokens is None:
        tokens = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    result = (topic["id"], content, tokens)
    return (*result, metadata) if metadata is not None else result

def attempt_all(topics):
    results = []
    for t in tqdm(topics):
        result = attempt_query(t)
        results.append(result)

    return results


from concurrent.futures import ThreadPoolExecutor, as_completed

def attempt_all_concurrent(topics):
    results = []

    # max_workers defines the maximum number of concurrent threads
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # 1. Submit all tasks to the thread pool
        futures = [executor.submit(attempt_query, t) for t in topics]

        # 2. as_completed yields results as soon as a task finishes
        # Use tqdm to update the progress bar in real time
        for future in tqdm(as_completed(futures), total=len(topics)):
            try:
                result = future.result()
            except Exception as exc:
                print(f"Report generation failed: {exc}")
                result = None
            results.append(result)

    return results

def main(args):
    globals()["args"] = args
    if args.num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max_samples must be positive")

    if args.local_dataset:  # Load from local dataset
        dataset = load_local_dataset(args.local_dataset)
    else:
        dataset = load_hf_dataset(args, "subjective")

    # convert to list of json for async parallelism
    topics = validate_questions([dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())])
    request_fingerprints = {str(topic["id"]): request_fingerprint(topic) for topic in topics}

    # If max_samples is set, limit the number of questions
    if args.max_samples:
        topics = topics[:args.max_samples]

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = getattr(args, "output_dir", None) or os.path.normpath(os.path.join(project_root, "eval_result", "subjective_eval", "dr_reports"))
    os.makedirs(output_dir, exist_ok=True)

    output_filepath = os.path.normpath(os.path.join(output_dir, f"reports_{os.path.basename(args.model)}.json"))

    # load only questions without responses
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding='utf-8') as f:
            predictions = json.load(f)
        if not isinstance(predictions, dict):
            raise ValueError("Report cache must be a JSON object")
        for value in predictions.values():
            if (isinstance(value, dict) and value.get("model") is not None
                    and resolve_openrouter_model(value["model"]) != resolve_openrouter_model(args.model)):
                raise ValueError("Report output filename is already used by a different model; select a separate output_dir")
        predictions = {key: value for key, value in predictions.items()
                       if key in request_fingerprints and valid_report(value, args.model)
                       and value.get("request_fingerprint") == request_fingerprints[key]}
        processed_ids = set(predictions)
        topics = [t for t in topics if str(t["id"]) not in processed_ids]
        for topic in topics:
            predictions.pop(str(topic["id"]), None)
        print(f"Loaded previous predictions, {len(predictions)} completed tasks")
        print(f"Remaining {len(topics)} topics to process")
    else:
        predictions = {}
        print(f"Previous output file not found, will create new result file: {output_filepath}")

    # Remove invalidated responses before work starts; a failed replacement must
    # never leave the previous question's report available to downstream scoring.
    atomic_save(output_filepath, predictions)

    succeeded = 0
    args.research_cancel_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=args.num_workers)
    futures = [executor.submit(attempt_query, topic) for topic in topics]
    try:
        for future in tqdm(as_completed(futures), total=len(topics)):
            try:
                result = future.result()
            except Exception as exc:
                print(f"Report generation failed: {exc}")
                result = None
            if result is None:
                continue
            unique_id, response, usage = result[:3]
            metadata = result[3] if len(result) > 3 else {}
            predictions[str(unique_id)] = {"model": args.model, "response": response, "usage": usage,
                                           "request_fingerprint": request_fingerprints[str(unique_id)], **metadata}
            atomic_save(output_filepath, predictions)
            succeeded += 1
    except BaseException:
        args.research_cancel_event.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    atomic_save(output_filepath, predictions)
    failed = len(topics) - succeeded
    if failed:
        print(f"Report generation incomplete: {failed} task(s) failed. Successful reports were saved.")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--dataset", type=str, help="Hugging Face dataset")
    sources.add_argument("--local_dataset", type=str, default=None, help="Local JSON dataset file path")
    add_hf_dataset_arguments(parser)
    parser.add_argument("--model", type=str, required=True, help="Model Endpoint Name")
    parser.add_argument("--max_completion_tokens", type=int, default=None,
                        help="Optional. Limit completion tokens. It can be used to avoid model collapse.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling.")
    parser.add_argument("--num_workers", type=int, default=5,
                        help="Async semaphore size. This depends on your rate limit. Only work in the async version.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional. Limit evaluation to first N samples")
    parser.add_argument("--output_dir", type=str, default=None, help="Override report output directory")
    args = parser.parse_args()
    sys.exit(main(args))
