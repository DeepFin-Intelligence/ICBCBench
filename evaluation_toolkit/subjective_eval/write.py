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


from tqdm import tqdm

from evaluation_toolkit.utils import load_local_dataset
from evaluation_toolkit.subjective_eval.prompts import *
from evaluation_toolkit.model_clients import *
from evaluation_toolkit.DR_clients import *

load_dotenv()

client = OPENROUTER

def format_message(topic):
    language = "zh" if topic["language"] in ["zh", "chinese", "中文"] else "en"
    user_topic = topic['question'] if language == "zh" else topic['question_en']
    content = f"用户的研究请求：{user_topic}" if language == "zh" else f"User's Research Query: {user_topic}"
    SYSTEM_PROMPT = WRITING_PROMPT_ZH if language == "zh" else WRITING_PROMPT_EN

    system_role = "model" if "gemini-deep-research" in args.model else "system"  # o1 no sys prompt
    messages = [
        {"role": system_role, "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    return messages


def attempt_query(topic):
    messages = format_message(topic)

    models_to_check = ["o1", "o3", "gpt-4o-search-preview", "gpt-4o-mini-search-preview", "claude-opus-4-7"]    # Models that do not accept the temperature parameter

    DR_models = {
        "gemini-deep-research": GeminiDeepResearch(),
        "tongyi-deepresearch-30b-a3b": TongyiDeepResearch(),
        "tavily-research": TavilyDeepResearch(),
        "o4-mini-deep-research": OpenaiDeepResearch(model_choose=args.model),
        "o3-deep-research": OpenaiDeepResearch(model_choose=args.model),
        "tongyi-deep-research": TongyiDeepResearch(),
        "Qwen-deep-research": QwenDeepResearch()
    }

    Openrouter_mapping = {
        "perplexity-deep-research": "perplexity/sonar-deep-research",
        "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
        "kimi-k2.5-thinking": "kimi-k2.5"
    }

    try:
        if args.model in DR_models:
            dr_client = DR_models[args.model]
            response = dr_client.run_research(system_prompt=messages[0]["content"], dr_task=messages[1]["content"])
            content = response.get("content")
            tokens = response.get("usage", None)
        else:
            response = client.chat.completions.create(
                model=Openrouter_mapping[args.model] if client == OPENROUTER else args.model,
                # temperature=args.temperature if args.model not in models_to_check else None,
                max_tokens=args.max_completion_tokens,
                messages=messages,
                stream=False,
                extra_body={"thinking": {"type": "enabled"}} if args.model in ["deepseek-v4-pro", "kimi-k2.6"] else {"reasoning": {"enabled": True}}
            )

            content = response.choices[0].message.content
            tokens = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
    except Exception as e:
        print("\nError:", e)
        return None

    if content is None:  # failed
        return None

    if tokens is None:
        tokens = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    return topic["id"], content, tokens

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
            result = future.result()  # Get execution result
            results.append(result)

    return results

def main(args):

    if args.local_dataset:  # Load from local dataset
        dataset = load_local_dataset(args.local_dataset)
    else:
        dataset = None

    # convert to list of json for async parallelism
    topics = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    # If max_samples is set, limit the number of questions
    if args.max_samples:
        topics = topics[:args.max_samples]

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = os.path.normpath(os.path.join(project_root, "eval_result", "subjective_eval", "dr_reports"))
    os.makedirs(output_dir, exist_ok=True)

    output_filepath = os.path.normpath(os.path.join(output_dir, f"reports_{os.path.basename(args.model)}.json"))

    # load only questions without responses
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding='utf-8') as f:
            predictions = json.load(f)
        processed_ids = list(predictions.keys())
        topics = [t for t in topics if str(t["id"]) not in processed_ids]
        print(f"Loaded previous predictions, {len(predictions)} completed tasks")
        print(f"Remaining {len(topics)} topics to process")
    else:
        predictions = {}
        print(f"Previous output file not found, will create new result file: {output_filepath}")

    results = attempt_all_concurrent(topics)

    # It is allowed to rerun this script multiple times if there are failed API calls
    for result in results:
        if result is None:  # API call failed
            continue
        unique_id, response, usage = result
        predictions[unique_id] = {
            "model": args.model,
            "response": response,
            "usage": usage
        }

    # Cache responses
    with open(output_filepath, "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="HLE HF Dataset")
    parser.add_argument("--local_dataset", type=str, default=None, help="Local JSON dataset file path")
    parser.add_argument("--model", type=str, help="Model Endpoint Name")
    parser.add_argument("--max_completion_tokens", type=int, default=None,
                        help="Optional. Limit completion tokens. It can be used to avoid model collapse.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling.")
    parser.add_argument("--num_workers", type=int, default=5,
                        help="Async semaphore size. This depends on your rate limit. Only work in the async version.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional. Limit evaluation to first N samples")
    args = parser.parse_args()
    main(args)