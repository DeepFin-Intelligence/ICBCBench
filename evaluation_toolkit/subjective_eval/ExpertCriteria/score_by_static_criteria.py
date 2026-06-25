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

client = DMX

def format_prompts(criteria_json):
    prompts = {}

    # 循环遍历每一个一级维度
    for dim in criteria_json['dimensions']:
        dim_id = dim['id']
        dim_name = dim['name']

        for sub in dim['sub_criteria']:
            # 1. 构建该维度下的评分标准文本
            criteria_text = ""

            # 格式化子维度的标题
            criteria_text += f"### 子维度 {sub['id']}: {sub['name']} (满分 {sub['max_points']} 分)\n"
            criteria_text += "评分阶梯 (Rubrics):\n"

            # 格式化具体的评分细则
            for rubric in sub['rubrics']:
                score_str = f"{rubric['score_range'][0]}-{rubric['score_range'][1]}分"
                criteria_text += f"- [{score_str}]: {rubric['description']}\n"

            sub_id = sub['id'].replace('.', '_')
            sub_name = sub['name']

            # 2. 将构建好的标准填入 Prompt 模板
            prompt_template = f"""你是一位资深的研究报告评审专家。
请阅读用户上传的研究报告，仅针对以下特定维度进行评分。

---
## 当前评估维度：{dim_name} - {sub_name} (满分 {sub['max_points']} 分)

请严格依据以下评分标准进行打分：
{criteria_text}
---

## 输出要求
请直接输出 JSON 格式的结果，不要包含任何 Markdown 格式符号（如 ```json）。
输出结构如下：
{{
    "score": "int (0-{sub['max_points']})",
    "reasoning": "string (引用原文并解释评分依据)"
}}"""

            prompts[sub_id] = prompt_template

    return prompts


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
    # print(f"正在调用 LLM...")
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

# 子维度分别传入prompt进行评估，目前已废弃
def evaluate_report_with_sub_criteria(report_text, eval_criteria):
    print(f"开始评估报告 (长度: {len(report_text)} 字符)")

    final_results = {
        "total_score": 0,
        "dimensions_breakdown": []
    }

    # 生成各维度 Prompt
    eval_prompts = format_whole_prompt(eval_criteria, report_text)

    eval_result = {
        "total_score": 0,
        "dimensions_breakdown": []
    }

    # 遍历每一个子维度
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

            print(f"正在评估维度{sub_id}: {dim_name} - {sub_name}")

            messages = [
                {"role": "system", "content": eval_prompt},
                {"role": "user", "content": f"待评测报告：\n{report_text}"}
            ]

            try:
                eval_result_str = call_llm(messages)

                # 解析结果
                clean_str = eval_result_str.replace("```json", "").replace("```", "").strip()
                eval_result = json.loads(clean_str)

                # 累加分数
                sub_score = eval_result["score"]
                dim_result["dimension_score"] += sub_score
                dim_result["sub_dimensions_breakdown"].append({
                        "sub_id": sub_id.replace('.', '_'),
                        "sub_name": sub_name,
                        **eval_result
                    }
                )

            except Exception as e:
                print(f"  [Error] 评估维度 {sub_id} 时发生错误: {e}")

        print(f"  -> {dim_name} 得分: {dim_result['dimension_score']}")
        final_results["total_score"] += dim_result["dimension_score"]
        final_results["dimensions_breakdown"].append(dim_result)

    return final_results

# 整体维度一次评估
def evaluate_report_with_whole_criteria(report_text, eval_criteria, q_language):
    # print(f"开始评估报告 (长度: {len(report_text)} 字符)")

    # 生成 Prompt
    eval_prompts = format_whole_prompt(eval_criteria, report_text, q_language)

    if eval_prompts:
        # print("已构造评估 prompt")
        messages = [
            {"role": "user", "content": eval_prompts}
        ]
    else:
        print("未构造评估 prompt")
        return None

    eval_result = {
        "total_score": 0,
        "dimensions_breakdown": []
    }

    dimension_scores_str = call_llm(messages)

    # 解析结果
    dimension_scores = safe_json_loads(dimension_scores_str)

    eval_result['total_score'] = sum(int(dimension_score["score"]) for dimension_score in dimension_scores)
    eval_result["dimensions_breakdown"] = dimension_scores

    return eval_result

def process_single_report(task_id, report_content, args, question_dict):
    """
    处理单个评估任务的工作线程函数
    返回: (result_dict, log_message)
    """
    q_item = question_dict.get(task_id)
    if q_item is None:
        return None, f"未找到task_id为 {task_id} 的话题"

    # 加载 question
    q_language = q_item.get('language', 'zh')
    task_prompt = q_item['question'] if q_language == 'zh' else q_item['question_en']

    if not task_prompt:
        return None, f"task_id为 {task_id} 的题目内容为空"

    # 加载 criteria
    eval_criteria = q_item['expert_evaluation_criteria'] if q_language == 'zh' else q_item[
        'expert_evaluation_criteria_en']

    # 加载文章
    eval_report = report_content["response"]
    model_string = report_content["model"]

    # openai deep research 的引用链接写在文中，评估时需去除
    DRs_to_check = ["o3-deep-research-ssvip", "o4-mini-deep-research"]
    if any(dr_model in model_string for dr_model in DRs_to_check):
        eval_report = re.sub(r'\(\[.*?\]\(.*?\)\)', '', eval_report)
        eval_report = re.sub(r' +', ' ', eval_report).strip()

    # 执行评估 (捕获可能发生的异常，防止单个任务崩溃导致整个线程池死掉)
    try:
        new_score_record = evaluate_report_with_whole_criteria(eval_report, eval_criteria, q_language)

        result = {
            "task_id": task_id,
            "model": args.report_model,
            "question": task_prompt,
            **new_score_record
        }
        log_msg = f"Task {task_id} 完成 | 最终总分: {new_score_record.get('total_score')}"
        return result, log_msg

    except Exception as e:
        return None, f"Task {task_id} 评估过程中发生异常: {str(e)}"

def main(args):
    # 从 topics.json 加载数据
    with open(args.question_path, 'r', encoding='utf-8') as f:
        question_data = json.load(f)

    question_dict = {item["id"]: item for item in question_data}

    # 加载待评分报告
    report_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(args.question_path)), f"dr_reports/reports_{args.report_model}.json"))

    # 读取已生成报告内容
    with open(report_path, "r", encoding="utf-8") as f:
        reports_data = json.load(f)


    output_dir = os.path.join(args.output_dir, f"judge_{args.judge}")
    output_path = os.path.join(output_dir, f"scores_{args.report_model}.jsonl")

    print(f"预期输出文件: {output_path}")

    # 检查是否存在已保存的评分结果
    already_scored_tasks = set()

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # 确保不是空行
                        score_record = json.loads(line)
                        already_scored_tasks.add(score_record["task_id"])
        except Exception as e:
            print(f"读取已有评分结果时出错: {e}")
            already_scored_tasks = set()
    else:
        print(f"{output_path} 不存在，将创建该文件")
        os.makedirs(output_dir, exist_ok=True)

    # 过滤出未评分的报告
    unprocessed_reports = {}
    for task_id, report_content in reports_data.items():
        if task_id not in already_scored_tasks:
            unprocessed_reports[task_id] = report_content
        else:
            print(f"任务 {task_id} 已经评分，跳过")

    print(f"总共 {len(reports_data)} 个任务，{len(unprocessed_reports)} 个任务待评分")

    if not unprocessed_reports:
        print("没有需要评分的新任务，程序退出")
        exit(0)

    save_interval = 1  # 每10个任务保存一次，可根据需要调整

    # 提前根据 max_samples 截断任务列表，避免提交多余任务
    items_to_process = list(unprocessed_reports.items())
    if args.max_samples:
        items_to_process = items_to_process[:args.max_samples]
        print(f"设定了 max_samples = {args.max_samples}，即将处理 {len(items_to_process)} 个任务")

    buffer = []
    total_processed = 0
    # 可以在 args 中配置并发数，如果没有配置，默认为 10
    max_workers = getattr(args, 'max_workers', 10)

    print(f"\n开启并发评估，并发数: {max_workers}")

    # 使用线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到线程池
        future_to_task = {
            executor.submit(process_single_report, task_id, report_content, args, question_dict): task_id
            for task_id, report_content in items_to_process
        }

        # 使用 as_completed 结合 tqdm 获取完成的结果（谁先完成就先处理谁）
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(items_to_process),
                           desc="评分进度"):
            task_id = future_to_task[future]

            try:
                result, log_msg = future.result()

                if result is not None:
                    buffer.append(result)
                    total_processed += 1
                else:
                    tqdm.write(f"[SKIP] {log_msg}")

                    # 在主线程中集中处理文件保存，绝对线程安全
                    if len(buffer) >= save_interval:
                        # tqdm.write(f"已处理 {total_processed} 个任务，正在保存中间结果...")
                        with open(output_path, "a", encoding="utf-8") as f:
                            for item in buffer:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        buffer.clear()
                        # tqdm.write("中间结果已保存")

            except Exception as exc:
                tqdm.write(f"任务 {task_id} 引发了致命异常: {exc}")

    # 保存剩余结果到文件
    if len(buffer) > 0:
        print(f"\n正在保存最后 {len(buffer)} 个任务的结果...")
        with open(output_path, "a", encoding="utf-8") as f:
            for item in buffer:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("所有结果已保存")

    print(f"共处理 {total_processed} 个任务，评分结果已保存至: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="证券研究报告自动评分系统")
    parser.add_argument("--judge", type=str, required=True, help="打分模型")
    parser.add_argument("--report_model", type=str, required=True, help="待评估的模型")
    project_root = get_project_root()
    eval_dir = get_eval_data_dir()
    default_question_path = os.path.join(project_root, "report_collect_and_eval", "subjective_report_questions.jsonl")
    default_output_dir = os.path.join(eval_dir, "subjective_eval", "test_ICBC_DR_Tasks", "scores")
    parser.add_argument("--question_path", type=str, default=default_question_path, help="问题和评分标准JSON文件路径")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="评分结果输出文件路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最大评估样本数，None表示评估所有未评分任务")
    parser.add_argument("--max_workers", type=int, default=10, help="并发数")

    args = parser.parse_args()
    main(args)

