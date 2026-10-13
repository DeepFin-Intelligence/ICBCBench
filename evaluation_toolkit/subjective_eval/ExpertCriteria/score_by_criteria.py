import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p.name != "evaluation_toolkit" and _p.parent != _p:
    _p = _p.parent
if _p.name == "evaluation_toolkit":
    _root = _p.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import concurrent.futures
import json
import re
import time
import math
import unicodedata
from evaluation_toolkit.model_clients import *
import argparse
from tqdm import tqdm

from evaluation_toolkit import safe_json_loads
from evaluation_toolkit.utils import get_project_root
from evaluation_toolkit.subjective_eval.runtime import atomic_save, valid_report, fingerprint, validate_questions
from evaluation_toolkit.subjective_eval.scoring.provenance import question_fingerprint, report_fingerprint

client = OPENROUTER


def parse_criteria_dimensions(criteria_text):
    """Read the numbered Markdown dimensions and point caps in public rubrics.

    Unstructured custom rubric text returns an empty mapping. A rubric containing
    numbered dimension headings must be completely parseable and total 100.
    """
    if not isinstance(criteria_text, str):
        raise ValueError("Scoring criteria must be text")
    dimensions = {}
    primary = None
    primary_caps = {}
    for raw_line in criteria_text.splitlines():
        line = unicodedata.normalize("NFKC", raw_line).strip().strip("*").strip()
        if not (line.startswith("#") or raw_line.strip().startswith("**")):
            continue
        heading = re.sub(r"^#+\s*", "", line)
        top = re.match(r"^(\d+)\.\s+", heading)
        sub = re.match(r"^\((\d+)\)\s*", heading)
        if not top and not sub:
            continue
        points = re.search(r"\((?:共\s*)?(\d+(?:\.\d+)?)\s*(?:points?|分)\)\s*$", heading, re.I)
        if points is None:
            raise ValueError(f"Cannot parse dimension point limit: {raw_line}")
        cap = float(points.group(1))
        if top:
            primary = top.group(1)
            if primary in primary_caps:
                raise ValueError(f"Duplicate primary dimension {primary}")
            primary_caps[primary] = cap
        else:
            if primary is None:
                raise ValueError("Secondary dimension has no primary heading")
            dimension_id = f"{primary}.{sub.group(1)}"
            if dimension_id in dimensions:
                raise ValueError(f"Duplicate dimension {dimension_id} in scoring criteria")
            dimensions[dimension_id] = cap
    if primary_caps:
        for primary, cap in primary_caps.items():
            subtotal = sum(v for k, v in dimensions.items() if k.split(".")[0] == primary)
            if not math.isclose(subtotal, cap):
                raise ValueError(f"Incomplete scoring criteria for primary dimension {primary}: {subtotal} != {cap}")
        if not math.isclose(sum(dimensions.values()), 100):
            raise ValueError("Scoring criteria dimension limits must total 100")
    return dimensions


def validate_dimension_scores(dimension_scores, eval_criteria):
    if not isinstance(dimension_scores, list) or not dimension_scores:
        raise ValueError("Judge must return a nonempty JSON array of dimension scores")
    limits = parse_criteria_dimensions(eval_criteria)
    seen = set()
    total = 0
    for dimension in dimension_scores:
        if not isinstance(dimension, dict):
            raise ValueError("Each dimension score must be a JSON object")
        dimension_id = dimension.get("dimension_id")
        if not isinstance(dimension_id, str) or not re.fullmatch(r"\d+\.\d+", dimension_id):
            raise ValueError("Each score requires a dimension_id such as 1.1")
        if dimension_id in seen:
            raise ValueError(f"Duplicate score dimension {dimension_id}")
        seen.add(dimension_id)
        score = dimension.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Dimension {dimension_id} score must be a finite number")
        if limits and dimension_id not in limits:
            raise ValueError(f"Unexpected score dimension {dimension_id}")
        cap = limits.get(dimension_id, 100)
        if score < 0 or score > cap:
            raise ValueError(f"Dimension {dimension_id} score must be between 0 and {cap}")
        total += score
    if limits and seen != set(limits):
        raise ValueError(f"Missing score dimensions: {sorted(set(limits) - seen)}")
    if not math.isfinite(total) or total > 100:
        raise ValueError("Total expert score must be between 0 and 100")
    return total


def score_fingerprint(question, report, judge):
    language = question.get("language", "zh")
    criteria = (question.get("expert_evaluation_criteria_en") or question["expert_evaluation_criteria"]) if language == "en" else question["expert_evaluation_criteria"]
    return fingerprint({"judge": judge, "question": question.get("question_en") or question.get("question"),
                        "report": report["response"], "report_model": report["model"],
                        "prompt": format_whole_prompt(criteria, report["response"], language),
                        "dimension_limits": parse_criteria_dimensions(criteria), "schema_version": 2})

def format_whole_prompt(criteria_text, report_text, q_language):
    prompt_template_zh = f"""你是一位极其严苛的金融研究报告资深评审专家。你的任务是仔细阅读【研究报告】，对照【评分标准】进行极其严格的量化评估。

    ### 评估准则
    1. 证据驱动：每一个得分点必须能在报告中找到具体的数据支撑、事实引用或完整的逻辑链条作为证据，不允许“提及概念即给分”。
    2. 高分门槛：顶档分数要求该部分报告内容具备极高的信息密度，与评分标准完全契合；缺乏深度的内容只能停留在中间档或最低档。
    3. 颗粒度审查：若报告仅停留在宽泛的现象描述，缺乏结构化的拆解和实质性论证，应视作论点缺乏支撑，根据评分标准区分“表层陈述”与“深度解析”。
    
    ### 评估流程
    为了保证评分的客观性和准确性，请你在内部执行以下评估步骤，但最终仅输出 JSON 结果：
    1. 提取文本：针对每一个二级维度，首先在报告全文中寻找对应该维度的文本内容，例如具体事实、数据或逻辑链条。请确保提取的内容与该维度紧密相关，并能支持你的评分。
    2. 识别缺漏：客观分析该维度下的论证是否闭环。指出报告中存在的证据缺失、逻辑断层、或是颗粒度不足的具体地方。
    3. 对齐等级：将提取到的文本内容与该维度下的分档描述（如 7-8分、4-6分等）进行严格比对，确定报告符合哪一档。
    3. 精准赋分：在确定的分档区间内，根据证据的详实程度给出具体分数，并撰写简短的扣分/给分理由。

    ### 评分标准
    {criteria_text}

    ### 研究报告
    {report_text}

    ### 输出要求
    请直接输出一个合法的 JSON 数组，包含每个维度的分析与得分。不要包含任何 Markdown 格式符号。
      [
        {{
          "dimension_id": "1.1",
          "dimension_name": "二级维度的名字",
          "evidence": "报告中客观存在的支撑信息",
          "shortcomings": "以严苛的视角，指出该维度下报告的具体缺点",
          "reasoning": "结合上述证据与不足，给出扣分理由和分档说明",
          "score": 0
        }},
        {{
          "dimension_id": "1.2",
          "dimension_name": "二级维度的名字",
          "evidence": "...",
          "shortcomings": "...",
          "reasoning": "...",
          "score": 0
        }}
        // ... 请务必完整输出所有二级维度的评价，保持结构一致 ...
      ]
    """

    prompt_template_en = f"""You are an extremely strict and fastidious senior review expert for financial research reports. Your task is to carefully read the [Research Report] and conduct an **extremely rigorous** quantitative evaluation against the [Grading Criteria].

    ### Assessment Principles
    1. Evidence-Driven: Every point awarded must be supported by specific data, factual citations, or complete logical chains found in the report. Merely "mentioning a concept" earns no points.
    2. High-Score Threshold: Top-tier scores require the report content in that section to possess an extremely high information density and align perfectly with the grading criteria. Content lacking depth must be restricted to the middle or lowest scoring tiers.
    3. Granularity Check: If the report merely provides broad descriptions of phenomena without structured breakdowns and substantive argumentation, it should be deemed as lacking support. You must differentiate between "surface-level statements" and "in-depth analysis" based on the grading criteria.

    ### Evaluation Process
    To ensure the objectivity and accuracy of your scoring, please execute the following evaluation steps internally, but ultimately output ONLY the JSON result:
    1. Extract Text: For each secondary dimension, first locate the corresponding text content in the full report, such as specific facts, data, or logical chains. Ensure the extracted content is closely related to the dimension and can objectively support your score.
    2. Identify Shortcomings: Objectively analyze whether the argumentation under that dimension forms a closed loop. Point out specific areas in the report where there is a lack of evidence, logical gaps, or insufficient granularity.
    3. Align with Tier: Strictly compare the extracted text content with the tier descriptions (e.g., 7-8 points, 4-6 points) under that dimension to determine which tier the report falls into.
    4. Precise Scoring: Within the determined tier range, assign a specific score based on the comprehensiveness of the evidence, and write a brief rationale for the points awarded or deducted.

    ### Grading Criteria
    {criteria_text}

    ### Research Report
    {report_text}

    ### Output Requirements
    Please directly output a valid JSON array containing the analysis and score for each dimension. Do not include any Markdown formatting symbols (such as ```json or ```).
      [
        {{
          "dimension_id": "1.1",
          "dimension_name": "Name of the secondary dimension",
          "evidence": "Objective supporting information found in the report",
          "shortcomings": "From a strict perspective, point out specific flaws where the report is not fully articulated, missing information, or speaking in generalities",
          "reasoning": "Combining the shortcomings above, provide the rationale for point deductions and tier placement",
          "score": 0
        }},
        {{
          "dimension_id": "1.2",
          "dimension_name": "Name of the secondary dimension",
          "evidence": "...",
          "shortcomings": "...",
          "reasoning": "...",
          "score": 0
        }}
        // ... Please ensure you output the evaluation for ALL secondary dimensions completely, keeping the structure consistent ...
      ]
    """

    if q_language == "en":
        return prompt_template_en
    else:
        return prompt_template_zh

def call_llm(messages):
    # print(f"Calling LLM...")
    response = client.chat.completions.create(
        model=resolve_openrouter_model(args.judge),
        messages=messages,
        temperature=0.0,
        # response_format={
        #     'type': 'json_object'
        # }
    )

    require_complete_chat(response)
    eval_result = response.choices[0].message.content

    return eval_result

# Evaluate by passing sub-dimensions into prompts separately; currently deprecated
def evaluate_report_with_sub_criteria(report_text, eval_criteria):
    print(f"Starting report evaluation (length: {len(report_text)} characters)")

    final_results = {
        "total_score": 0,
        "dimensions_breakdown": []
    }

    # Generate prompts for each dimension
    eval_prompts = format_whole_prompt(eval_criteria, report_text)

    eval_result = {
        "total_score": 0,
        "dimensions_breakdown": []
    }

    # Iterate over each sub-dimension
    for dim in eval_criteria['dimensions']:
        dim_id = dim['id']
        dim_name = dim['name']

        dim_result = {
            "dimension_id": dim_id,
            "dimension_name": dim_name,
            "dimension_score": 0,
            "sub_dimensions_breakdown": []
        }

        for sub in dim['sub_criteria']:
            sub_id = sub['id']
            sub_name = sub['name']
            eval_prompt = eval_prompts[sub_id.replace('.', '_')]

            print(f"Evaluating dimension {sub_id}: {dim_name} - {sub_name}")

            messages = [
                {"role": "system", "content": eval_prompt},
                {"role": "user", "content": f"Report to be evaluated:\n{report_text}"}
            ]

            try:
                eval_result_str = call_llm(messages)

                # Parse result
                clean_str = eval_result_str.replace("```json", "").replace("```", "").strip()
                eval_result = json.loads(clean_str)

                # Accumulate score
                sub_score = eval_result["score"]
                dim_result["dimension_score"] += sub_score
                dim_result["sub_dimensions_breakdown"].append({
                        "sub_id": sub_id.replace('.', '_'),
                        "sub_name": sub_name,
                        **eval_result
                    }
                )

            except Exception as e:
                print(f"  [Error] Error occurred while evaluating dimension {sub_id}: {e}")

        print(f"  -> {dim_name} score: {dim_result['dimension_score']}")
        final_results["total_score"] += dim_result["dimension_score"]
        final_results["dimensions_breakdown"].append(dim_result)

    return final_results

# Evaluate all dimensions at once
def evaluate_report_with_whole_criteria(report_text, eval_criteria, q_language):
    # print(f"Starting report evaluation (length: {len(report_text)} characters)")

    # Generate prompt
    eval_prompts = format_whole_prompt(eval_criteria, report_text, q_language)
    limits = parse_criteria_dimensions(eval_criteria)
    if limits:
        eval_prompts += "\nRequired dimension_id values and maximum scores (return every ID exactly once):\n" + json.dumps(limits, ensure_ascii=False)

    if eval_prompts:
        # print("Evaluation prompt constructed")
        messages = [
            {"role": "user", "content": eval_prompts}
        ]
    else:
        print("Evaluation prompt not constructed")
        return None

    eval_result = {
        "total_score": 0,
        "dimensions_breakdown": []
    }

    dimension_scores_str = call_llm(messages)

    # Parse result
    dimension_scores = safe_json_loads(dimension_scores_str)

    eval_result['total_score'] = validate_dimension_scores(dimension_scores, eval_criteria)
    eval_result["dimensions_breakdown"] = dimension_scores

    return eval_result

def process_single_report(task_id, report_content, args, question_dict):
    """
    Worker function for processing a single evaluation task.
    Returns: (result_dict, log_message)
    """
    q_item = question_dict.get(task_id)
    if q_item is None:
        return None, f"Topic with task_id {task_id} not found"

    # Load question
    q_language = q_item.get('language', 'zh')
    task_prompt = (q_item.get('question_en') or q_item.get('question')) if q_language == 'en' else q_item.get('question')

    if not task_prompt:
        return None, f"Question content for task_id {task_id} is empty"

    # Load criteria
    eval_criteria = (q_item.get('expert_evaluation_criteria_en') or q_item['expert_evaluation_criteria']) if q_language == 'en' else q_item['expert_evaluation_criteria']

    # Load article
    eval_report = report_content["response"]
    model_string = report_content["model"]

    # OpenAI deep research citation links are embedded in the text and need to be removed for evaluation
    DRs_to_check = ["o3-deep-research-ssvip", "o4-mini-deep-research"]
    if any(dr_model in model_string for dr_model in DRs_to_check):
        eval_report = re.sub(r'\(\[.*?\]\(.*?\)\)', '', eval_report)
        eval_report = re.sub(r' +', ' ', eval_report).strip()

    # Execute evaluation (catch exceptions to prevent a single task failure from crashing the entire thread pool)
    try:
        new_score_record = evaluate_report_with_whole_criteria(eval_report, eval_criteria, q_language)

        result = {
            "task_id": task_id,
            "model": args.report_model,
            "judge": args.judge,
            "score_fingerprint": score_fingerprint(q_item, report_content, args.judge),
            "question_fingerprint": question_fingerprint(q_item),
            "report_fingerprint": report_fingerprint(task_id, report_content),
            "question": task_prompt,
            **new_score_record
        }
        log_msg = f"Task {task_id} completed | Final total score: {new_score_record.get('total_score')}"
        return result, log_msg

    except Exception as e:
        return None, f"Exception occurred during evaluation of Task {task_id}: {str(e)}"

def main(args):
    globals()["args"] = args
    if args.max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max_samples must be positive")
    # Load question data
    with open(args.question_path, 'r', encoding='utf-8') as f:
        question_data = json.load(f)
    question_data = validate_questions(question_data)

    question_dict = {str(item["id"]): item for item in question_data}
    if len(question_dict) != len(question_data):
        raise ValueError("Question IDs must be unique")

    # Load reports to be scored
    model_filename = os.path.basename(args.report_model)
    report_path = os.path.normpath(os.path.join(args.report_dir, f"reports_{model_filename}.json"))

    # Read generated report content
    with open(report_path, "r", encoding="utf-8") as f:
        reports_data = json.load(f)
    if not isinstance(reports_data, dict):
        raise ValueError("Reports must be a JSON object")
    selected_ids = list(question_dict)[:args.max_samples] if args.max_samples is not None else list(question_dict)
    missing_reports = set(selected_ids) - set(reports_data)
    unknown_reports = set(reports_data) - set(question_dict)
    if missing_reports:
        print(f"Reports missing for selected tasks: {sorted(missing_reports)}")
    if unknown_reports:
        print(f"Ignoring reports with unknown task IDs: {sorted(unknown_reports)}")


    output_dir = os.path.join(args.output_dir, f"judge_{os.path.basename(args.judge)}")
    output_path = os.path.join(output_dir, f"scores_{model_filename}.jsonl")

    print(f"Expected output file: {output_path}")

    # A cached row is complete only when every rubric dimension is valid.
    cached_records = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record.get("model") not in (None, args.report_model):
                        raise RuntimeError("Score filename is already used by a different model; choose a separate output_dir")
                    if record.get("judge") and resolve_openrouter_model(record["judge"]) != resolve_openrouter_model(args.judge):
                        raise RuntimeError("Score output directory is already used by a different judge; choose a separate output_dir")
                    task_id = str(record["task_id"])
                    question = question_dict[task_id]
                    criteria = (question.get("expert_evaluation_criteria_en") or question["expert_evaluation_criteria"]) if question.get("language") == "en" else question["expert_evaluation_criteria"]
                    total = validate_dimension_scores(record["dimensions_breakdown"], criteria)
                    if record.get("model") != args.report_model or record.get("judge") != args.judge or record.get("total_score") != total or record.get("score_fingerprint") != score_fingerprint(question, reports_data[task_id], args.judge):
                        raise ValueError("Incomplete or inconsistent cached score")
                    if (record.get("question_fingerprint") != question_fingerprint(question)
                            or record.get("report_fingerprint") != report_fingerprint(task_id, reports_data[task_id])):
                        raise ValueError("Cached score provenance does not match current questions and reports")
                    cached_records[task_id] = record
                except RuntimeError:
                    raise
                except Exception as exc:
                    print(f"Ignoring invalid cached score on line {line_number}: {exc}")

    items_to_process = [(task_id, reports_data[task_id]) for task_id in selected_ids
                        if task_id in reports_data and task_id not in cached_records]
    print(f"Total {len(reports_data)} reports, {len(items_to_process)} tasks pending scoring")
    total_processed = 0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers)
    futures = {}
    try:
        for task_id, content in items_to_process:
            if not valid_report(content, args.report_model):
                print(f"[SKIP] Task {task_id} has an empty report or different model")
                continue
            futures[executor.submit(process_single_report, task_id, content, args, question_dict)] = task_id
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scoring progress"):
            task_id = futures[future]
            try:
                result, log_message = future.result()
            except Exception as exc:
                print(f"Task {task_id} failed: {exc}")
                continue
            if result is None:
                print(f"[SKIP] {log_message}")
                continue
            cached_records[task_id] = result
            atomic_save(output_path, list(cached_records.values()), jsonl=True)
            total_processed += 1
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    atomic_save(output_path, list(cached_records.values()), jsonl=True)
    failed = len(items_to_process) - total_processed + len(missing_reports) + len(unknown_reports)
    print(f"Processed {total_processed} tasks; scores saved to {output_path}")
    if failed:
        print(f"Scoring incomplete: {failed} task(s) failed; rerun to retry.")
    return 1 if failed else 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Automatic Scoring System for Securities Research Reports")
    parser.add_argument("--judge", type=str, required=True, help="Scoring model")
    parser.add_argument("--report_model", type=str, required=True, help="Model to be evaluated")
    project_root = get_project_root()
    default_question_path = os.path.normpath(os.path.join(project_root, "data", "subjective_questions_public_40.json"))
    default_report_dir = os.path.normpath(os.path.join(project_root, "eval_result", "subjective_eval", "dr_reports"))
    default_output_dir = os.path.normpath(os.path.join(project_root, "eval_result", "subjective_eval", "scores"))
    parser.add_argument("--question_path", type=str, default=default_question_path, help="Path to the question and scoring criteria JSON file")
    parser.add_argument("--report_dir", type=str, default=default_report_dir, help="Directory containing generated reports")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Path to save scoring results")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate, None means evaluate all unscored tasks")
    parser.add_argument("--max_workers", type=int, default=10, help="Number of concurrent workers")

    args = parser.parse_args()
    sys.exit(main(args))
