import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p.name != "evaluation_toolkit" and _p.parent != _p:
    _p = _p.parent
if _p.name == "evaluation_toolkit":
    _root = _p.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import csv
import json
import os
import argparse
from tqdm import tqdm
from evaluation_toolkit.subjective_eval.FACT.time_decay import *
from evaluation_toolkit.subjective_eval.FACT import authority
from evaluation_toolkit.utils import get_eval_data_dir

def fact_evaluate(data, args, id_to_lang):
    # ======= 引用链接有效性 (严格按数学定义: S_citation = (1/T) * sum(N_s,t / |U_t|) ========
    print(" ======= 引用链接有效性 ========")
    total_tasks = 0
    doc_support_rates_list = []

    data = filter_data_by_language(data, args.language_filter, id_to_lang)

    for d in tqdm(data):
        total_tasks += 1
        doc_citations = 0        # |U_t|
        doc_support_citations = 0  # N_s,t

        citations_deduped = d.get('citations_deduped', {})
        if not citations_deduped:
            doc_support_rates_list.append(0)
            continue

        for url, c in citations_deduped.items():
            facts = c.get('facts', [])
            n_facts = len(facts)
            doc_citations += n_facts

            if n_facts == 0:
                continue

            # validate_error or scrape failed: count towards |U_t| but not N_s,t
            if c.get('validate_error') is not None:
                continue
            if c.get('url_content', '').startswith('scrape failed'):
                continue

            validate_res = c.get('validate_res', [])
            for i in range(min(n_facts, len(validate_res))):
                if validate_res[i]['result'] == 'supported':
                    doc_support_citations += 1

        doc_score = (doc_support_citations / doc_citations) if doc_citations > 0 else 0
        doc_support_rates_list.append(doc_score)

    # S_citation = (1/T) * sum(S_citation^(t))
    avg_citation_score = sum(doc_support_rates_list) / total_tasks if total_tasks > 0 else 0
    avg_citation_score *= 100  # 调整为 0-100 范围

    print("-" * 30)
    print(f"Total Tasks (T): {total_tasks}")
    print("-" * 30)
    print("【Citation Score - 严格按定义】")
    print(f"Macro Avg (per-task): {avg_citation_score:.4f}")
    print("-" * 30)

    return {
        "Total Tasks": total_tasks,
        "Avg Citation Score (Macro)": avg_citation_score
    }
    

def authority_evaluate(data, args, id_to_lang):
    # ======= 权威加权引用链接有效性 (分母严格为 |U_t|) ========
    print(" ======= 权威加权引用链接有效性 ========")
    total_tasks = 0
    doc_weighted_support_rates_list = []

    data = filter_data_by_language(data, args.language_filter, id_to_lang)

    for d in tqdm(data):
        total_tasks += 1
        doc_citations = 0
        doc_weighted_support_sum = 0.0

        citations_deduped = d.get('citations_deduped', {})
        if not citations_deduped:
            doc_weighted_support_rates_list.append(0)
            continue

        for url, c in citations_deduped.items():
            facts = c.get('facts', [])
            n_facts = len(facts)
            doc_citations += n_facts

            if n_facts == 0:
                continue

            # validate_error or scrape failed: count towards |U_t| but not weighted sum
            if c.get('validate_error') is not None:
                continue
            if c.get('url_content', '').startswith('scrape failed'):
                continue

            true_url = c.get('true_url', url)
            auth_weight = authority.get_authority_weight(true_url)

            validate_res = c.get('validate_res', [])
            for i in range(min(n_facts, len(validate_res))):
                if validate_res[i]['result'] == 'supported':
                    doc_weighted_support_sum += auth_weight

        doc_score = (doc_weighted_support_sum / doc_citations) if doc_citations > 0 else 0
        doc_weighted_support_rates_list.append(doc_score)

    avg_weighted_score = sum(doc_weighted_support_rates_list) / total_tasks if total_tasks > 0 else 0

    print("-" * 30)
    print(f"Total Tasks (T): {total_tasks}")
    print("-" * 30)
    print("【Authority-Weighted - 严格按定义】")
    print(f"Macro Avg (per-task): {avg_weighted_score:.4f}")
    print("-" * 30)

    return {
        "Total Tasks": total_tasks,
        "Avg Authority-Weighted Score (Macro)": avg_weighted_score
    }



def source_evaluate(data, args, id_to_lang):
    # ========= Source Quality (S_auth, S_time, S_source) ==========
    print("\n ========= Source Quality ==========")
    data = filter_data_by_language(data, args.language_filter, id_to_lang)

    total_tasks = 0
    doc_auth_scores = []
    doc_time_scores = []
    doc_source_scores = []

    for d in tqdm(data, desc="Source Quality Evaluate"):
        total_tasks += 1
        citations_deduped = d.get('citations_deduped', {})

        if not citations_deduped:
            doc_auth_scores.append(0)
            doc_time_scores.append(0)
            doc_source_scores.append(0)
            continue

        auth_scores = []
        time_scores = []
        source_scores = []

        for url, c in citations_deduped.items():
            # 跳过验证错误的引用
            if c.get('validate_error') is not None:
                continue

            true_url = c.get('true_url', url)
            auth_score = authority.get_authority_weight(true_url)

            publish_time = c.get('publish_time')
            time_score = calculate_time_score(publish_time, alpha=0.002)
            # 跳过无法解析发布时间的引用
            if time_score < 0:
                continue

            auth_scores.append(auth_score)
            time_scores.append(time_score)
            source_scores.append(auth_score * time_score)

        m = len(auth_scores)
        if m == 0:
            doc_auth_scores.append(0)
            doc_time_scores.append(0)
            doc_source_scores.append(0)
        else:
            doc_auth_scores.append(sum(auth_scores) / m)
            doc_time_scores.append(sum(time_scores) / m)
            doc_source_scores.append(sum(source_scores) / m)

    avg_auth = sum(doc_auth_scores) / total_tasks if total_tasks > 0 else 0
    avg_time = sum(doc_time_scores) / total_tasks if total_tasks > 0 else 0
    avg_source = sum(doc_source_scores) / total_tasks if total_tasks > 0 else 0
    avg_source *= 100  # 调整为 0-100 范围

    print(f"Total Tasks (T): {total_tasks}")
    print(f"Avg Source Authority Score (S_auth): {avg_auth:.4f}")
    print(f"Avg Source Timeliness Score (S_time): {avg_time:.4f}")
    print(f"Avg Source Quality Score (S_source): {avg_source:.4f}")

    return {
        "Total Tasks": total_tasks,
        "Avg Source Authority Score (S_auth)": avg_auth,
        "Avg Source Timeliness Score (S_time)": avg_time,
        "Avg Source Quality Score (S_source)": avg_source
    }


def filter_data_by_language(data, language_filter, id_to_lang):
    if not language_filter or not id_to_lang:
        return data

    filtered_data = []
    for d in data:
        # 获取 jsonl 中的 id，默认为空字符串防止报错
        doc_id = str(d.get('id', ''))
        raw_lang = id_to_lang.get(doc_id, '').lower()

        # 归一化判断逻辑
        query_language = "zh" if raw_lang in ["zh", "chinese", "中文", "zh-cn"] else "en"

        if query_language == language_filter:
            filtered_data.append(d)

    print(f"[{language_filter} 过滤] 原始数据 {len(data)} 条 -> 过滤后保留 {len(filtered_data)} 条")
    return filtered_data

def run_stat(args, id_to_lang, output_csv, language_filter, subset_ids=None):
    """执行一次统计并写入 CSV
    subset_ids: 可选，id 集合，仅统计该子集内的数据
    """
    # 临时设置语言过滤
    original_filter = getattr(args, 'language_filter', None)
    args.language_filter = language_filter

    # 1. 读取数据（每次重新读取，避免被 time_decay 修改）
    raw_data = load_jsonl(args.input_path)

    # 1.5 根据子集 id 过滤
    if subset_ids:
        raw_data = [d for d in raw_data if str(d.get('id', '')) in subset_ids]
        print(f"[子集过滤] 过滤后保留 {len(raw_data)} 条记录")

    # 2. 执行评估并获取字典格式的结果
    fact_res = fact_evaluate(raw_data, args, id_to_lang)
    source_res = source_evaluate(raw_data, args, id_to_lang)

    # 3. 组装宽表格式（每行一个模型）
    row = {
        "Model Name": args.model_name,
        "Total Tasks": fact_res.get("Total Tasks", 0),
        "Avg Citation Score (Macro)": fact_res.get("Avg Citation Score (Macro)", 0),
        "Avg Source Authority Score (S_auth)": source_res.get("Avg Source Authority Score (S_auth)", 0),
        "Avg Source Timeliness Score (S_time)": source_res.get("Avg Source Timeliness Score (S_time)", 0),
        "Avg Source Quality Score (S_source)": source_res.get("Avg Source Quality Score (S_source)", 0),
    }

    # 5. 写入 CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    file_exists = os.path.isfile(output_csv)

    with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
        fieldnames = ["Model Name", "Total Tasks", "Avg Citation Score (Macro)",
                      "Avg Source Authority Score (S_auth)", "Avg Source Timeliness Score (S_time)",
                      "Avg Source Quality Score (S_source)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"\n评估结果已成功追加至: {output_csv}")

    # 恢复原始语言过滤
    args.language_filter = original_filter


if __name__ == "__main__":

    eval_dir = get_eval_data_dir()
    default_query_file = os.path.join(eval_dir, "subjective_eval", "final_test", "subjective_60.jsonl")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="name of the model to record")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--query_file", type=str, default=default_query_file, help="问题 JSONL 文件路径")
    parser.add_argument("--language_filter", type=str, required=False)
    parser.add_argument("--public_ids", type=str, default=None,
                        help="Public subset id JSONL file path (e.g., subjective_report_questions_public_40.jsonl)")
    args = parser.parse_args()

    with open(args.query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    id_to_lang = {
        str(item["id"]): item.get("language", "").lower()
        for item in query_data
    }

    # 加载子集 id
    public_ids = None
    if args.public_ids and os.path.exists(args.public_ids):
        with open(args.public_ids, "r", encoding="utf-8") as f:
            public_data = json.load(f)
        public_ids = {str(item["id"]) for item in public_data}
        print(f"[Public subset] Loaded {len(public_ids)} ids from {args.public_ids}")

    def run_with_subset(subset_ids, suffix):
        base, ext = os.path.splitext(args.output_csv)
        if args.language_filter:
            output_csv = f"{base}_{suffix}_{args.language_filter}{ext}"
            run_stat(args, id_to_lang, output_csv, args.language_filter, subset_ids)
        else:
            for lang in [None, "zh", "en"]:
                if lang:
                    output_csv = f"{base}_{suffix}_{lang}{ext}"
                else:
                    output_csv = f"{base}_{suffix}{ext}"
                print(f"\n{'='*50}")
                print(f"Processing: {suffix.upper()} | {lang.upper() if lang else 'ALL'}")
                print(f"{'='*50}")
                run_stat(args, id_to_lang, output_csv, lang, subset_ids)

    # 如果有子集配置，运行 public
    if public_ids:
        run_with_subset(public_ids, "public")

    # 如果没有指定子集，保持原有行为
    if not public_ids:
        if args.language_filter:
            run_stat(args, id_to_lang, args.output_csv, args.language_filter)
        else:
            base, ext = os.path.splitext(args.output_csv)
            for lang in [None, "zh", "en"]:
                if lang:
                    output_csv = f"{base}_{lang}{ext}"
                else:
                    output_csv = args.output_csv
                print(f"\n{'='*50}")
                print(f"Processing: {lang.upper() if lang else 'ALL'}")
                print(f"{'='*50}")
                run_stat(args, id_to_lang, output_csv, lang)





