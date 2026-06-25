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


# source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
def calib_err(confidence, correct, p='2', beta=100):
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    # 濡傛灉鎬绘牱鏈暟 < beta锛屽垯鏁翠釜鏁版嵁浣滀负涓€涓?bin
    if len(confidence) < beta:
        bins = [[0, len(confidence)]]
    else:
        bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
        bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for start, end in bins:  # 閬嶅巻姣忎釜 bin
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

    # 淇濆瓨缁撴灉
    # 鏋勯€犳柊涓€琛?    columns = ["Model", "Accuracy (%)", "CalibrationError (%)", "Judge"]
    new_row = pd.DataFrame([{
        "Model": prediction_model_name,
        "Accuracy (%)": accuracy,
        "CalibrationError (%)": calibration_error,
        "Judge": judge_model
    }], columns=columns)  # 鈫?鐩存帴鎸囧畾 columns

    # 4. 宸叉湁缁撴灉 or 鍒涘缓鏂?DataFrame
    if os.path.exists(results_csv_path):
        df_existing = pd.read_csv(results_csv_path)
    else:
        df_existing = pd.DataFrame(columns=["Model", "Accuracy (%)", "CalibrationError (%)"])

    # 鏄惁宸插瓨鍦ㄨ妯″瀷鐨勭粨鏋?    if prediction_model_name in df_existing["Model"].values:
        df_existing = df_existing[df_existing["Model"] != prediction_model_name]

    # 鍔犲叆鏂扮粨鏋?    df_new = pd.concat([df_existing, new_row], ignore_index=True)

    # 淇濆瓨鏂扮粨鏋?    df_new.to_csv(results_csv_path, index=False, encoding="utf-8")
    print(f"Results saved to {results_csv_path}")


def get_multimodal_ids_from_json(json_path):
    """Read multimodal question IDs from the modality field of JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    multimodal_ids = [str(item["id"]) for item in data if item.get("modality", "") == "multimodal"]
    return multimodal_ids


def dump_all_metrics(judge_model, language_filter, multimodal_ids=None):

    eval_dir = get_eval_data_dir()
    judged_dir = os.path.join(eval_dir, "objective_eval", "final_test", "judged_results", f"judge_{judge_model}")

    query_file = os.path.join(eval_dir, "objective_eval", "final_test", "qa_140.json")
    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    id_to_lang = {
        str(item["id"]): item.get("language", "").lower()
        for item in query_data
    }

    # multimodal_ids: user inputs MULTIMODAL ids, we EXCLUDE them to get text-only
    multimodal_set = None
    if multimodal_ids is not None:
        multimodal_set = {str(tid) for tid in multimodal_ids}
        print(f"Loaded multimodal exclusion: {len(multimodal_set)} IDs (text-only = remaining after exclusion)")

    # 缁撴灉鏂囦欢璺緞
    results_dir = os.path.join(eval_dir, "objective_eval", "final_test", "judged_results", f"judge_{judge_model}")
    suffix_parts = []
    if language_filter:
        suffix_parts.append(language_filter)
    if multimodal_set is not None:
        suffix_parts.append("text_only")
    suffix = "_".join(suffix_parts)

    if suffix:
        results_csv_path = os.path.join(results_dir, f"{suffix}_evaluation_results.csv")
    else:
        results_csv_path = os.path.join(results_dir, "evaluation_results.csv")

    for filename in os.listdir(judged_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(judged_dir, filename)
            # 鎻愬彇妯″瀷鍚?            predictions_filename = os.path.basename(file_path)
            # 鐢ㄤ笅鍒掔嚎鍒嗗壊骞跺彇鏈€鍚庝竴閮ㄥ垎
            prediction_model_name = predictions_filename.rsplit("_", 1)[-1].replace(".json", "")
            with open(file_path, "r", encoding="utf-8") as f:
                judged_results = json.load(f)

                # Apply text-only filter: exclude multimodal ids
                if multimodal_set is not None:
                    judged_results = {k: v for k, v in judged_results.items() if str(k) not in multimodal_set}

                if language_filter:
                    filtered_results = {}
                    for k, v in judged_results.items():
                        # 鑾峰彇璇?ID 瀵瑰簲鐨勮瑷€锛岄粯璁よ涓?"en"
                        raw_lang = id_to_lang.get(str(k), "")
                        query_language = "zh" if raw_lang in ["zh", "chinese", "涓枃"] else "en"

                        # 浠呬繚鐣欏尮閰嶇殑璇█
                        if query_language == language_filter:
                            filtered_results[k] = v

                    judged_results = filtered_results
                dump_metrics(prediction_model_name, judge_model, judged_results, len(judged_results), results_csv_path)

if __name__ == "__main__":

    judge_model = "gpt-5.4"
    # judge_model = "gpt-5.2"

    eval_dir = get_eval_data_dir()
    query_file = os.path.join(eval_dir, "objective_eval", "final_test", "qa_140.json")
    multimodal_ids = get_multimodal_ids_from_json(query_file)
    print(f"Multimodal IDs from JSON: {multimodal_ids}")

    dump_all_metrics(judge_model, None)
    dump_all_metrics(judge_model, "zh")
    dump_all_metrics(judge_model, "en")
    dump_all_metrics(judge_model, "zh", multimodal_ids=multimodal_ids)
    dump_all_metrics(judge_model, "en", multimodal_ids=multimodal_ids)
