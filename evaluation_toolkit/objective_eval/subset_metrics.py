import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p.name != "evaluation_toolkit" and _p.parent != _p:
    _p = _p.parent
if _p.name == "evaluation_toolkit":
    _root = _p.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import json
import os
import math

from evaluation_toolkit.utils import get_eval_data_dir


def calib_err(confidence, correct, p='2', beta=100):
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


def dump_metrics(prediction_model_name, judge_model, judged_results, n, results_csv_path):
    if n == 0:
        print("No predictions to evaluate")
        return

    correct = []
    confidence = []
    for k, v in judged_results.items():
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
    calibration_error = round(100 * calib_err(confidence, correct, p='2', beta=10), 2)

    print("*** Metrics ***")
    print(f"Accuracy: {accuracy}% +/- {confidence_half_width}% | n = {n}")
    print(f"Calibration Error: {calibration_error}")

    # 保存结果
    # 构造新一行
    columns = ["Model", "Accuracy (%)", "CalibrationError (%)", "Judge"]
    new_row = pd.DataFrame([{
        "Model": prediction_model_name,
        "Accuracy (%)": accuracy,
        "CalibrationError (%)": calibration_error,
        "Judge": judge_model
    }], columns=columns)  # ← 直接指定 columns

    # 4. 已有结果 or 创建新 DataFrame
    if os.path.exists(results_csv_path):
        df_existing = pd.read_csv(results_csv_path)
    else:
        df_existing = pd.DataFrame(columns=["Model", "Accuracy (%)", "CalibrationError (%)"])

    # 是否已存在该模型的结果
    if prediction_model_name in df_existing["Model"].values:
        df_existing = df_existing[df_existing["Model"] != prediction_model_name]

    # 加入新结果
    df_new = pd.concat([df_existing, new_row], ignore_index=True)

    # 保存新结果
    df_new.to_csv(results_csv_path, index=False, encoding="utf-8")
    print(f"Results saved to {results_csv_path}")


def dump_all_metrics(judge_model, language_filter, subset_filter=None, multimodal_ids=None):

    eval_dir = get_eval_data_dir()
    v2_dir = os.path.join(eval_dir, "objective_eval", "v2_obj_public-private_test")

    judged_dir = os.path.join(eval_dir, "objective_eval", "final_test", "judged_results", f"judge_{judge_model}")

    # Load public query data only
    query_data = []
    public_file = os.path.join(v2_dir, "adjusted_objective_questions_public_80.json")
    if os.path.exists(public_file):
        with open(public_file, "r", encoding="utf-8") as f:
            query_data.extend(json.load(f))

    id_to_lang = {
        str(item["id"]): item.get("language", "").lower()
        for item in query_data
    }

    # Load subset IDs
    subset_ids = None
    if subset_filter:
        subset_file_map = {
            "public": os.path.join(v2_dir, "adjusted_objective_questions_public_80.json"),
        }
        subset_path = subset_file_map.get(subset_filter)
        if subset_path and os.path.exists(subset_path):
            with open(subset_path, "r", encoding="utf-8") as f:
                subset_data = json.load(f)
            subset_ids = {str(item["id"]) for item in subset_data}
            print(f"Loaded {subset_filter} subset: {len(subset_ids)} IDs")
        else:
            print(f"Warning: subset file not found for {subset_filter}")

    # multimodal_ids: user inputs MULTIMODAL ids, we EXCLUDE them to get text-only
    multimodal_set = None
    if multimodal_ids is not None:
        multimodal_set = {str(tid) for tid in multimodal_ids}
        print(f"Loaded multimodal exclusion: {len(multimodal_set)} IDs (text-only = remaining after exclusion)")

    # Result file paths
    suffix_parts = []
    if language_filter:
        suffix_parts.append(language_filter)
    if subset_filter:
        suffix_parts.append(subset_filter)
    if multimodal_set is not None:
        suffix_parts.append("text_only")
    suffix = "_".join(suffix_parts)

    results_dir = v2_dir
    os.makedirs(results_dir, exist_ok=True)
    if suffix:
        results_csv_path = os.path.join(results_dir, f"{suffix}_evaluation_results.csv")
    else:
        results_csv_path = os.path.join(results_dir, "evaluation_results.csv")

    for filename in os.listdir(judged_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(judged_dir, filename)
            # 提取模型名
            predictions_filename = os.path.basename(file_path)
            # 用下划线分割并取最后一部分
            prediction_model_name = predictions_filename.rsplit("_", 1)[-1].replace(".json", "")
            with open(file_path, "r", encoding="utf-8") as f:
                judged_results = json.load(f)

                # Restrict to known public question IDs
                judged_results = {k: v for k, v in judged_results.items() if str(k) in id_to_lang}

                # Apply language filter
                if language_filter:
                    filtered_results = {}
                    for k, v in judged_results.items():
                        raw_lang = id_to_lang.get(str(k), "")
                        query_language = "zh" if raw_lang in ["zh", "chinese", "中文"] else "en"
                        if query_language == language_filter:
                            filtered_results[k] = v
                    judged_results = filtered_results

                # Apply subset filter
                if subset_ids is not None:
                    judged_results = {k: v for k, v in judged_results.items() if str(k) in subset_ids}

                # Apply text-only filter: exclude multimodal ids
                if multimodal_set is not None:
                    judged_results = {k: v for k, v in judged_results.items() if str(k) not in multimodal_set}

                dump_metrics(prediction_model_name, judge_model, judged_results, len(judged_results), results_csv_path)

def get_multimodal_ids_from_json(json_path):
    """Read multimodal question IDs from the modality field of adjusted JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    multimodal_ids = [str(item["id"]) for item in data if item.get("modality", "") == "multimodal"]
    return multimodal_ids


if __name__ == "__main__":

    judge_model = "gpt-5.4"
    # judge_model = "gpt-5.2"

    # Public subset only
    dump_all_metrics(judge_model, None, subset_filter="public")
    dump_all_metrics(judge_model, "zh", subset_filter="public")
    dump_all_metrics(judge_model, "en", subset_filter="public")

    # Text-only subset: derive multimodal IDs from JSON modality field
    eval_dir = get_eval_data_dir()
    public_json = os.path.join(eval_dir, "objective_eval", "v2_obj_public-private_test", "adjusted_objective_questions_public_80.json")
    public_multimodal_ids = get_multimodal_ids_from_json(public_json)
    print(f"Public multimodal IDs from JSON: {public_multimodal_ids}")
    dump_all_metrics(judge_model, "zh", subset_filter="public", multimodal_ids=public_multimodal_ids)
    dump_all_metrics(judge_model, "en", subset_filter="public", multimodal_ids=public_multimodal_ids)
