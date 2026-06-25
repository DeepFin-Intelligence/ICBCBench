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
from evaluation_toolkit.model_clients import *
import argparse
from tqdm import tqdm

from evaluation_toolkit import safe_json_loads
from evaluation_toolkit.utils import get_eval_data_dir, get_project_root

client = OPENROUTER

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
        model=args.judge,
        messages=messages,
        temperature=0.0,
        # response_format={
        #     'type': 'json_object'
        # }
    )

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

    eval_result['total_score'] = sum(int(dimension_score["score"]) for dimension_score in dimension_scores)
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
    task_prompt = q_item['question'] if q_language == 'zh' else q_item['question_en']

    if not task_prompt:
        return None, f"Question content for task_id {task_id} is empty"

    # Load criteria
    eval_criteria = q_item['expert_evaluation_criteria'] if q_language == 'zh' else q_item[
        'expert_evaluation_criteria_en']

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
            "question": task_prompt,
            **new_score_record
        }
        log_msg = f"Task {task_id} completed | Final total score: {new_score_record.get('total_score')}"
        return result, log_msg

    except Exception as e:
        return None, f"Exception occurred during evaluation of Task {task_id}: {str(e)}"

def main(args):
    # Load data from topics.json
    with open(args.question_path, 'r', encoding='utf-8') as f:
        question_data = json.load(f)

    question_dict = {item["id"]: item for item in question_data}

    # Load reports to be scored
    report_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(args.question_path)), f"dr_reports/reports_{args.report_model}.json"))

    # Read generated report content
    with open(report_path, "r", encoding="utf-8") as f:
        reports_data = json.load(f)


    output_dir = os.path.join(args.output_dir, f"judge_{args.judge}")
    output_path = os.path.join(output_dir, f"scores_{args.report_model}.jsonl")

    print(f"Expected output file: {output_path}")

    # Check for existing saved scoring results
    already_scored_tasks = set()

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # ensure not an empty line
                        score_record = json.loads(line)
                        already_scored_tasks.add(score_record["task_id"])
        except Exception as e:
            print(f"Error reading existing scoring results: {e}")
            already_scored_tasks = set()
    else:
        print(f"{output_path} does not exist, will create it")
        os.makedirs(output_dir, exist_ok=True)

    # Filter out unscored reports
    unprocessed_reports = {}
    for task_id, report_content in reports_data.items():
        if task_id not in already_scored_tasks:
            unprocessed_reports[task_id] = report_content
        else:
            print(f"Task {task_id} already scored, skipping")

    print(f"Total {len(reports_data)} tasks, {len(unprocessed_reports)} tasks pending scoring")

    if not unprocessed_reports:
        print("No new tasks to score, exiting")
        exit(0)

    save_interval = 1  # save every N tasks, adjust as needed

    # Truncate task list in advance based on max_samples to avoid submitting extra tasks
    items_to_process = list(unprocessed_reports.items())
    if args.max_samples:
        items_to_process = items_to_process[:args.max_samples]
        print(f"Set max_samples = {args.max_samples}, will process {len(items_to_process)} tasks")

    buffer = []
    total_processed = 0
    # Concurrency can be configured in args, default is 10 if not set
    max_workers = getattr(args, 'max_workers', 10)

    print(f"\nStarting concurrent evaluation, concurrency: {max_workers}")

    # Use thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the thread pool
        future_to_task = {
            executor.submit(process_single_report, task_id, report_content, args, question_dict): task_id
            for task_id, report_content in items_to_process
        }

        # Use as_completed with tqdm to get completed results as they finish
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(items_to_process),
                           desc="Scoring progress"):
            task_id = future_to_task[future]

            try:
                result, log_msg = future.result()

                if result is not None:
                    buffer.append(result)
                    total_processed += 1
                else:
                    tqdm.write(f"[SKIP] {log_msg}")

                    # Handle file saving in the main thread to ensure thread safety
                    if len(buffer) >= save_interval:
                        # tqdm.write(f"Processed {total_processed} tasks, saving intermediate results...")
                        with open(output_path, "a", encoding="utf-8") as f:
                            for item in buffer:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        buffer.clear()
                        # tqdm.write("Intermediate results saved")

            except Exception as exc:
                tqdm.write(f"Task {task_id} raised a fatal exception: {exc}")

    # Save remaining results to file
    if len(buffer) > 0:
        print(f"\nSaving results of the last {len(buffer)} tasks...")
        with open(output_path, "a", encoding="utf-8") as f:
            for item in buffer:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("All results saved")

    print(f"Processed {total_processed} tasks in total, scoring results saved to: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Automatic Scoring System for Securities Research Reports")
    parser.add_argument("--judge", type=str, required=True, help="Scoring model")
    parser.add_argument("--report_model", type=str, required=True, help="Model to be evaluated")
    project_root = get_project_root()
    eval_dir = get_eval_data_dir()
    default_question_path = os.path.join(project_root, "report_collect_and_eval", "subjective_report_questions.jsonl")
    default_output_dir = os.path.join(eval_dir, "subjective_eval", "test_ICBC_DR_Tasks", "scores")
    parser.add_argument("--question_path", type=str, default=default_question_path, help="Path to the question and scoring criteria JSON file")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Path to save scoring results")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate, None means evaluate all unscored tasks")
    parser.add_argument("--max_workers", type=int, default=10, help="Number of concurrent workers")

    args = parser.parse_args()
    main(args)

