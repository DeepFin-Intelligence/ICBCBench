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

def merge_multiple_jsonl_to_csv(file_list, output_csv_path, language_filter, id_to_lang_map):
    """
    Read multiple jsonl files and merge all model scores into a single CSV.
    """
    all_model_data = {}  # Store model scores {model_name: {task_id: score}}
    # ref_scores_map = {}  # Store reference scores {task_id: score}
    all_task_ids = set()  # Collect all task IDs that appear

    # 1. Iterate over all files
    for file_path in file_list:
        print(f"Processing: {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    entry = json.loads(line)
                    if language_filter:
                        # Get the language for this ID, default to "en"
                        raw_lang = id_to_lang_map.get(str(entry['task_id']), "")
                        query_language = "zh" if raw_lang in ["zh", "chinese", "中文"] else "en"

                        # Keep only matching language
                        if query_language != language_filter:
                            continue

                    # Extract key information
                    model = entry.get('model', 'Unknown')
                    tid = str(entry.get('task_id'))  # Convert to string to prevent type inconsistency
                    eval_score = entry.get('total_score', 0)
                    # ref_score = entry.get('ref_score', 0)

                    # Record task ID
                    all_task_ids.add(tid)

                    # Store model score
                    if model not in all_model_data:
                        all_model_data[model] = {}
                    all_model_data[model][tid] = eval_score

                    # Store/update reference score (assuming all files share the same reference standard, overwrite is fine)
                    # ref_scores_map[tid] = ref_score

        except Exception as e:
            print(f"Warning: Error reading file {file_path}: {e}")

    # 2. Prepare CSV data rows
    # Sort task IDs (1, 2, 3...)
    sorted_task_ids = sorted(list(all_task_ids), key=lambda x: int(x) if x.isdigit() else x)

    rows = []

    # --- Add model rows ---
    for model, scores in all_model_data.items():
        row = {'Model': model}
        score_sum = 0
        count = 0

        for tid in sorted_task_ids:
            s = scores.get(tid, 0)  # Default to 0 if missing
            row[tid] = round(s, 2)
            score_sum += s
            if s > 0:
                count += 1

        # Calculate average score
        avg_score = score_sum / count if count > 0 else 0
        # row['Avg Score'] = round(avg_score * 10, 2)  # Keep two decimal places
        row['Avg Score'] = round(avg_score, 2)
        rows.append(row)

    # # --- Add reference score row (last row) ---
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

    # 3. Generate DataFrame and save
    if rows:
        df = pd.DataFrame(rows)
        # Adjust column order: Model, 1, 2, 3..., Avg Score
        column_order = ['Model'] + sorted_task_ids + ['Avg Score']
        df = df[column_order]

        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig prevents Excel garbled Chinese characters
        print(f"\nSuccess! Merged data from {len(rows)} models into: {output_csv_path}")
        return df
    else:
        print("No data extracted.")
        return None


# ==========================================
# Usage example
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
        print(f"Found files: {jsonl_files}")
        merge_multiple_jsonl_to_csv(jsonl_files, output_csv_path, language_filter, id_to_lang)
    else:
        print("No .jsonl files found in the current directory")


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    default_input_dir = os.path.normpath(os.path.join(project_root, "eval_result", "subjective_eval", "scores", "judge_gemini-3.1-pro-preview"))
    default_query_file = os.path.normpath(os.path.join(project_root, "data", "subjective_questions_public_40.json"))
    default_output_csv = os.path.normpath(os.path.join(project_root, "eval_result", "subjective_eval", "scores", "judge_gemini-3.1-pro-preview", "leaderboard.csv"))

    parser = argparse.ArgumentParser(description="Merge multiple ExpertCriteria jsonl scores into a leaderboard CSV")
    parser.add_argument("--input_dir", type=str, default=default_input_dir, help="Directory containing jsonl scoring files")
    parser.add_argument("--query_file", type=str, default=default_query_file, help="Path to the question JSONL file")
    parser.add_argument("--output_csv", type=str, default=default_output_csv, help="Output CSV path")
    parser.add_argument("--language_filter", type=str, default=None, choices=["zh", "en"], help="Language filter")
    args = parser.parse_args()

    stat(args.input_dir, args.query_file, args.output_csv, args.language_filter)
