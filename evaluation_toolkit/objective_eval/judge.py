import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p.name != "evaluation_toolkit" and _p.parent != _p:
    _p = _p.parent
if _p.name == "evaluation_toolkit":
    _root = _p.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import os
import json
import copy
import math
import argparse
import asyncio
import numpy as np
from typing import Literal
import sys

import pandas as pd
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from dotenv import load_dotenv
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from evaluation_toolkit import load_local_dataset, safe_json_loads
from evaluation_toolkit.model_clients import *
from evaluation_toolkit.utils import get_eval_data_dir

load_dotenv()

client = OPENROUTER

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json"), "r", encoding='utf-8') as f:
    prompt_templates = json.load(f)


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]  # 100% reliability


def extract_answer(question, correct_answer, response, language):
    prompt_template = prompt_templates["judge"][language]
    SYSTEM_PROMPT = prompt_template.format(question=question, correct_answer=correct_answer, response=response)
    try:
        response = client.chat.completions.parse(
            model=args.judge,
            max_completion_tokens=4096,  # overkill for judge
            messages=[
                {"role": "user", "content": SYSTEM_PROMPT}
            ]
        )
        content = response.choices[0].message.content
        judge_json = safe_json_loads(content)
        return {
            "correct_answer": correct_answer,
            "model_answer": judge_json["extracted_final_answer"],
            "reasoning": judge_json["reasoning"],
            "correct": judge_json["correct"],
            "confidence": judge_json["confidence"]
        }
    except Exception as e:  # very, very rare
        print("\nError:", e)

        # 根据错误类型决定是否终止程序
        error_code = getattr(e, 'status_code', None)

        # 致命错误：认证失败、模型不存在等
        if error_code == 401:
            print("致命错误：API 认证失败，请检查 API Key")
            sys.exit(1)
        elif error_code == 403:
            print("致命错误：无权访问该模型，请检查权限")
            sys.exit(1)
        elif error_code == 404:
            print(f"致命错误：模型 '{args.judge}' 不存在")
            sys.exit(1)
        elif error_code == 503:
            print(f"致命错误：模型 '{args.judge}' 不存在")
            sys.exit(1)
        elif error_code == 500:
            print("服务器内部错误，跳过此问题")
            return None  # 非致命，继续处理其他问题
        elif error_code == 429:
            print("速率限制，建议降低并发数 (--num_workers)")
            return None  # 可以选择继续或退出
        else:
            # 网络错误等其他问题，继续处理
            return None


def add_judge_result(question, predictions):
    unique_id = question["id"]
    prediction = copy.deepcopy(predictions[unique_id])  # not in-place
    question_text = question["question"]
    correct_answer = question["answer"]
    language = "zh" if question["language"] in ["zh", "chinese", "中文"] else "en"

    if "judge_result" in prediction:  # already judged
        return unique_id, prediction

    response = prediction["response"]
    judge_result = extract_answer(question_text, correct_answer, response, language)

    if judge_result is not None:
        prediction["judge_model"] = args.judge
        prediction["judge_result"] = judge_result  # local in-place
        return unique_id, prediction
    else:
        return None, None


def judge_all_responses(questions, predictions):
    results=[]
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(add_judge_result, q, predictions): q for q in questions}
        for future in tqdm(as_completed(futures), total=len(questions)):
            res = future.result()
            if res:
                results.append(res)
    return results


# source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
def calib_err(confidence, correct, p='2', beta=10):
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    # 如果总样本数 < beta，则整个数据作为一个 bin
    if len(confidence) < beta:
        bins = [[0, len(confidence)]]
    else:
        bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
        bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for start, end in bins:  # 遍历每个 bin
        bin_confidence = confidence[start:end]
        bin_correct = correct[start:end]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr


def dump_metrics(predictions, n):
    if n == 0:
        print("No predictions to evaluate")
        return

    correct = []
    confidence = []
    for k, v in predictions.items():
        if "judge_result" in v:
            judge_result = v["judge_result"]
            correct.append("yes" in judge_result["correct"])
            confidence.append(judge_result["confidence"])
        else:
            print(f"Missing judge result for {k}, you should rerun the judge")

    correct = np.array(correct)
    confidence = np.array(confidence) / 100

    # sometimes model collapses on same questions
    if len(correct) != n:
        print(f"Available predictions: {len(correct)} | Total questions: {n}")

    accuracy = round(100 * sum(correct) / n, 2)
    # Wald estimator, 95% confidence interval
    confidence_half_width = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n), 2)
    calibration_error = 100 * round(calib_err(confidence, correct, p='2', beta=100), 2)

    print("*** Metrics ***")
    print(f"Accuracy: {accuracy}% +/- {confidence_half_width}% | n = {n}")
    print(f"Calibration Error: {calibration_error}")

    # 保存结果
    # 提取模型名
    predictions_filename = os.path.basename(args.predictions)
    # 用下划线分割并取最后一部分
    model_name = predictions_filename.rsplit("_", 1)[-1].replace(".json", "")

    # 构造新一行
    columns = ["Model", "Accuracy (%)", "CalibrationError (%)"]
    new_row = pd.DataFrame([{
        "Model": model_name,
        "Accuracy (%)": accuracy,
        "CalibrationError (%)": calibration_error
    }], columns=columns)  # ← 直接指定 columns

    # 结果文件路径
    eval_dir = get_eval_data_dir()
    results_csv = os.path.join(eval_dir, "objective_eval", "evaluation_results.csv")

    # 4. 已有结果 or 创建新 DataFrame
    if os.path.exists(results_csv):
        df_existing = pd.read_csv(results_csv)
    else:
        df_existing = pd.DataFrame(columns=["Model", "Accuracy (%)", "CalibrationError (%)"])

    # 是否已存在该模型的结果
    if model_name in df_existing["Model"].values:
        df_existing = df_existing[df_existing["Model"] != model_name]

    # 加入新结果
    df_new = pd.concat([df_existing, new_row], ignore_index=True)

    # 保存新结果
    df_new.to_csv(results_csv, index=False, encoding="utf-8")
    print(f"Results saved to {results_csv}")


def main(args):
    assert args.num_workers > 1, "num_workers must be 2 or greater"

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = os.path.normpath(os.path.join(project_root, "eval_result", "objective_eval", "judged_results", f"judge_{args.judge}"))
    os.makedirs(output_dir, exist_ok=True)

    output_filepath = f"{output_dir}/judged_{os.path.basename(args.predictions)}"

    if args.local_dataset:  # 从本地文件加载
        dataset = load_local_dataset(args.local_dataset)
    else:  # 从 HuggingFace 加载
        dataset = load_dataset(args.dataset, split="test").to_dict()

    # convert to list of json for async parallelism
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    total_questions = len(questions)

    with open(args.predictions, "r", encoding='utf-8') as f:
        predictions = json.load(f)

    # load only unjudged responses
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding='utf-8') as f:
            judged_predictions = json.load(f)
    else:
        judged_predictions = {}

    questions = [q for q in questions if q["id"] in predictions and q["id"] not in judged_predictions]

    # API will only be called for unjudged responses
    results = judge_all_responses(questions, predictions)

    for unique_id, predictions in results:
        if unique_id is not None:
            judged_predictions[unique_id] = predictions

    # cache judge output
    with open(output_filepath, "w", encoding='utf-8') as f:
        json.dump(judged_predictions, f, indent=4, ensure_ascii=False)

    dump_metrics(judged_predictions, n=total_questions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, help="HLE HF Dataset")
    parser.add_argument("--local_dataset", type=str, default=None, help="Local JSON dataset file path")
    parser.add_argument("--predictions", type=str, help="Model Predictions")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Async semaphore size. This depends on your rate limit.")
    parser.add_argument("--judge", type=str, default="gemini-3-pro-preview",
                        help="Judge model")  # prev: "gpt-4o-2024-08-06"
    args = parser.parse_args()
    main(args)