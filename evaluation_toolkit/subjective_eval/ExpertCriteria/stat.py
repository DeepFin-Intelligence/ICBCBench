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
import math
from evaluation_toolkit.subjective_eval.runtime import validate_questions
from evaluation_toolkit.model_clients import resolve_openrouter_model
from evaluation_toolkit.subjective_eval.scoring.provenance import (
    question_fingerprint, task_set_fingerprint, report_set_fingerprint, valid_fingerprint)
from evaluation_toolkit.subjective_eval.ExpertCriteria.score_by_criteria import validate_dimension_scores

def merge_multiple_jsonl_to_csv(file_list, output_csv_path, language_filter, id_to_lang_map, expected_ids=None, question_records=None):
    """
    Read multiple jsonl files and merge all model scores into a single CSV.
    """
    all_model_data = {}  # Store model scores {model_name: {task_id: score}}
    all_model_records = {}
    model_names = {}
    question_by_id = ({question['id']: question for question in validate_questions(question_records, require_question=False)}
                      if question_records is not None else {})
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
                    if not isinstance(entry, dict):
                        raise ValueError('Each expert score must be a JSON object')
                    raw_id = entry.get('task_id')
                    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)) or not str(raw_id).strip():
                        raise ValueError('Each expert score requires a nonempty task_id')
                    entry['task_id'] = str(raw_id).strip()
                    if expected_ids is not None and str(entry['task_id']) not in id_to_lang_map:
                        raise ValueError(f"Unknown task ID {entry['task_id']}")
                    if language_filter:
                        # Get the language for this ID, default to "en"
                        if str(entry['task_id']) not in id_to_lang_map:
                            raise ValueError(f"Unknown task ID {entry['task_id']}")
                        raw_lang = id_to_lang_map[str(entry['task_id'])]
                        query_language = "zh" if raw_lang in ["zh", "chinese", "中文"] else "en"

                        # Keep only matching language
                        if query_language != language_filter:
                            continue

                    # Extract key information
                    model = entry.get('model')
                    if not isinstance(model, str) or not model.strip():
                        raise ValueError('Each expert score requires a nonempty model name')
                    normalized_model = resolve_openrouter_model(model.strip().lower())
                    model_names.setdefault(normalized_model, model.strip())
                    tid = str(entry.get('task_id'))  # Convert to string to prevent type inconsistency
                    eval_score = entry.get('total_score')
                    if isinstance(eval_score, bool) or not isinstance(eval_score, (int, float)) or not math.isfinite(eval_score) or not 0 <= eval_score <= 100:
                        raise ValueError(f"Invalid expert total_score for task {tid}: {eval_score}")
                    if 'dimensions_breakdown' in entry or valid_fingerprint(entry.get('report_fingerprint')):
                        question = question_by_id.get(tid, {})
                        criteria = ((question.get('expert_evaluation_criteria_en') or question.get('expert_evaluation_criteria'))
                                    if question.get('language') == 'en' else question.get('expert_evaluation_criteria'))
                        total = validate_dimension_scores(entry.get('dimensions_breakdown'), criteria or '')
                        if not math.isclose(total, eval_score):
                            raise ValueError(f'Expert total_score does not match dimension scores for task {tid}')
                    # ref_score = entry.get('ref_score', 0)

                    # Record task ID
                    all_task_ids.add(tid)

                    # Store model score
                    if normalized_model not in all_model_data:
                        all_model_data[normalized_model] = {}
                        all_model_records[normalized_model] = {}
                    if tid in all_model_data[normalized_model]:
                        raise ValueError(f'Duplicate expert score for model {model} task {tid}')
                    all_model_data[normalized_model][tid] = eval_score
                    all_model_records[normalized_model][tid] = entry

                    # Store/update reference score (assuming all files share the same reference standard, overwrite is fine)
                    # ref_scores_map[tid] = ref_score

        except Exception as e:
            raise ValueError(f"Error reading score file {file_path}: {e}") from e

    # 2. Prepare CSV data rows
    # Sort task IDs (1, 2, 3...)
    if expected_ids is not None:
        all_task_ids.update(expected_ids)
    sorted_task_ids = sorted(list(all_task_ids), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

    rows = []

    # --- Add model rows ---
    for model, scores in all_model_data.items():
        row = {'Model': model_names[model]}
        score_sum = 0
        count = 0

        for tid in sorted_task_ids:
            s = scores.get(tid, 0)  # Default to 0 if missing
            # Intermediate CSVs feed multi-round aggregation: keep full precision,
            # including fractions introduced by the registered rubric correction.
            row[tid] = s
            score_sum += s
            if tid in scores:
                count += 1

        # Calculate average score
        avg_score = score_sum / count if count > 0 else 0
        row['Avg Score'] = avg_score
        if expected_ids is not None:
            row['Evaluated Tasks'] = len(set(scores) & set(expected_ids))
            row['Total Tasks'] = len(expected_ids)
            row['Coverage (%)'] = round(row['Evaluated Tasks'] / len(expected_ids) * 100, 2) if expected_ids else 0
        scoped_ids = set(expected_ids if expected_ids is not None else scores)
        selected_questions = [question_by_id[tid] for tid in sorted(scoped_ids) if tid in question_by_id]
        row['Task Set Fingerprint'] = (task_set_fingerprint(selected_questions)
                                       if selected_questions and len(selected_questions) == len(scoped_ids) else '')
        report_hashes = {}
        verified = bool(selected_questions) and set(scores) == scoped_ids
        for tid, record in all_model_records[model].items():
            report_hash = record.get('report_fingerprint')
            if (not valid_fingerprint(report_hash) or tid not in question_by_id
                    or record.get('question_fingerprint') != question_fingerprint(question_by_id[tid])):
                verified = False
            else:
                report_hashes[tid] = report_hash
        row['Report Set Fingerprint'] = report_set_fingerprint(report_hashes) if verified else ''
        row['Provenance Verified'] = verified
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
        if expected_ids is not None:
            column_order += ['Evaluated Tasks', 'Total Tasks', 'Coverage (%)']
        column_order += ['Task Set Fingerprint', 'Report Set Fingerprint', 'Provenance Verified']
        df = df[column_order]

        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig prevents Excel garbled Chinese characters
        print(f"\nSuccess! Merged data from {len(rows)} models into: {output_csv_path}")
        return df
    else:
        raise ValueError("No valid scores match the requested language")


# ==========================================
# Usage example
# ==========================================

def stat(input_dir, query_file, output_csv_path, language_filter=None):
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)
    query_data = validate_questions(query_data, require_question=False)

    id_to_lang = {
        str(item["id"]): item.get("language", "").lower()
        for item in query_data
    }
    expected_ids = {task_id for task_id, language in id_to_lang.items()
                    if language_filter is None or ("zh" if language in ["zh", "chinese", "中文"] else "en") == language_filter}

    if language_filter:
        base, ext = os.path.splitext(output_csv_path)
        output_csv_path = f"{base}_{language_filter}{ext}"

    if jsonl_files:
        print(f"Found files: {jsonl_files}")
        frame = merge_multiple_jsonl_to_csv(jsonl_files, output_csv_path, language_filter, id_to_lang, expected_ids, query_data)
        if frame['Coverage (%)'].lt(100).any():
            print("Expert leaderboard is incomplete; see Evaluated Tasks, Total Tasks, and Coverage (%).")
            return 1
        if not frame['Provenance Verified'].all():
            print('Expert leaderboard has unverified provenance; regenerate scores with the current toolkit before final aggregation.')
            return 1
        return 0
    else:
        raise FileNotFoundError(f"No .jsonl score files found in {input_dir}")


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

    sys.exit(stat(args.input_dir, args.query_file, args.output_csv, args.language_filter))
