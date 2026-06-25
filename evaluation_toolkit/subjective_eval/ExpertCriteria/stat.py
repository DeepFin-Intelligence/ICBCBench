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
import pandas as pd
import glob
import os
import argparse

from evaluation_toolkit.utils import get_eval_data_dir


def merge_multiple_jsonl_to_csv(file_list, output_csv_path, language_filter, id_to_lang_map):
    """
    读取多个 jsonl 文件，并将所有模型的分数合并到一个 CSV 中。
    """
    all_model_data = {}  # 存储模型分数 {model_name: {task_id: score}}
    # ref_scores_map = {}  # 存储参考分数 {task_id: score}
    all_task_ids = set()  # 收集所有出现过的任务ID

    # 1. 遍历所有文件
    for file_path in file_list:
        print(f"正在处理: {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    entry = json.loads(line)
                    if language_filter:
                        # 获取该 ID 对应的语言，默认设为 "en"
                        raw_lang = id_to_lang_map.get(str(entry['task_id']), "")
                        query_language = "zh" if raw_lang in ["zh", "chinese", "中文"] else "en"

                        # 仅保留匹配的语言
                        if query_language != language_filter:
                            continue

                    # 提取关键信息
                    model = entry.get('model', 'Unknown')
                    tid = str(entry.get('task_id'))  # 统一转字符串防止类型不一致
                    eval_score = entry.get('total_score', 0)
                    # ref_score = entry.get('ref_score', 0)

                    # 记录任务ID
                    all_task_ids.add(tid)

                    # 存储模型分数
                    if model not in all_model_data:
                        all_model_data[model] = {}
                    all_model_data[model][tid] = eval_score

                    # 存储/更新参考分数 (假设所有文件的参考标准一致，覆盖即可)
                    # ref_scores_map[tid] = ref_score

        except Exception as e:
            print(f"警告: 读取文件 {file_path} 时出错: {e}")

    # 2. 准备 CSV 数据行
    # 对任务ID进行排序 (1, 2, 3...)
    sorted_task_ids = sorted(list(all_task_ids), key=lambda x: int(x) if x.isdigit() else x)

    rows = []

    # --- 添加模型行 ---
    for model, scores in all_model_data.items():
        row = {'Model': model}
        score_sum = 0
        count = 0

        for tid in sorted_task_ids:
            s = scores.get(tid, 0)  # 缺省补0
            row[tid] = round(s, 2)
            score_sum += s
            if s > 0:
                count += 1

        # 计算百分制总分
        avg_score = score_sum / count if count > 0 else 0
        # row['Avg Score'] = round(avg_score * 10, 2)  # 保留两位小数
        row['Avg Score'] = round(avg_score, 2)
        rows.append(row)

    # # --- 添加参考分数行 (最后一行) ---
    # if ref_scores_map:
    #     ref_row = {'Model': 'ref_report'}
    #     ref_sum = 0
    #     ref_count = 0
    #     for tid in sorted_task_ids:
    #         s = ref_scores_map.get(tid, 0)
    #         ref_row[tid] = round(s, 2)
    #         ref_sum += s
    #         ref_count += 1
    #
    #     ref_avg = ref_sum / ref_count if ref_count > 0 else 0
    #     ref_row['Total Score'] = round(ref_avg * 10, 2)
    #     rows.append(ref_row)

    # 3. 生成 DataFrame 并保存
    if rows:
        df = pd.DataFrame(rows)
        # 调整列顺序: Model, 1, 2, 3..., Total Score
        column_order = ['Model'] + sorted_task_ids + ['Avg Score']
        df = df[column_order]

        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig 防止Excel中文乱码
        print(f"\n成功! 已合并 {len(rows)} 个模型的数据到: {output_csv_path}")
        return df
    else:
        print("未提取到任何数据。")
        return None


# ==========================================
# 使用方法示例
# ==========================================

def stat(input_dir, query_file, output_csv_path, language_filter=None):
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    id_to_lang = {
        str(item["id"]): item.get("language", "").lower()
        for item in query_data
    }

    if language_filter:
        base, ext = os.path.splitext(output_csv_path)
        output_csv_path = f"{base}_{language_filter}{ext}"

    if jsonl_files:
        print(f"发现文件: {jsonl_files}")
        merge_multiple_jsonl_to_csv(jsonl_files, output_csv_path, language_filter, id_to_lang)
    else:
        print("当前目录下没有找到 .jsonl 文件")


if __name__ == '__main__':
    eval_dir = get_eval_data_dir()
    default_input_dir = os.path.join(eval_dir, "subjective_eval", "final_test", "3rd_judge_gemini-3.1-pro-preview")
    default_query_file = os.path.join(eval_dir, "subjective_eval", "final_test", "subjective_60.jsonl")
    default_output_csv = os.path.join(eval_dir, "subjective_eval", "final_test", "3rd_judge_gemini-3.1-pro-preview", "3rd_leaderboard.csv")

    parser = argparse.ArgumentParser(description="合并多轮 ExpertCriteria jsonl 评分为 leaderboard CSV")
    parser.add_argument("--input_dir", type=str, default=default_input_dir, help="jsonl 评分文件所在目录")
    parser.add_argument("--query_file", type=str, default=default_query_file, help="问题 JSONL 文件路径")
    parser.add_argument("--output_csv", type=str, default=default_output_csv, help="输出 CSV 路径")
    parser.add_argument("--language_filter", type=str, default=None, choices=["zh", "en"], help="语言过滤")
    args = parser.parse_args()

    stat(args.input_dir, args.query_file, args.output_csv, args.language_filter)