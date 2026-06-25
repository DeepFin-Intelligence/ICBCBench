import json
import argparse
import math
import os
from datetime import datetime
from tqdm import tqdm
from dateutil import parser as date_parser  # 强大的时间解析库
from evaluation_toolkit.subjective_eval.FACT.utils import *


# ==========================================
# 核心计算函数
# ==========================================

def calculate_time_score(publish_time_str, alpha=0.002):
    """
    计算基于发布时间的时间衰减分数。
    公式: Score = e^(-alpha * (T_curr - T_pub))
    单位: 天
    """
    # 1. 如果没有发布时间，返回 None 或 默认分
    if not publish_time_str:
        return -1

    try:
        # 2. 解析时间字符串
        # fuzzy=True 可以处理类似 "Published: 2023-10-01" 这样的非标准字符串
        pub_time = date_parser.parse(str(publish_time_str), fuzzy=True)

        # 移除时区信息以进行比较（或者统一转为 UTC，这里简化为 naive time）
        pub_time = pub_time.replace(tzinfo=None)
        curr_time = datetime.now().replace(tzinfo=None)

        # 3. 计算天数差
        delta = curr_time - pub_time
        days_diff = delta.days

        # 如果发布时间在未来（脏数据），视为 0 天
        if days_diff < 0:
            days_diff = 0

        # 4. 指数衰减公式
        score = math.exp(-alpha * days_diff)

        return round(score, 4)

    except Exception as e:
        # 解析失败，返回默认分，并在日志中记录（可选）
        # print(f"Time parse error: {e} | Raw string: {publish_time_str}")
        return -1


def process_citation_time(citation_item, alpha):
    """
    处理单个引用条目，兼容 validate.py 的数据结构
    citation_item: (url, data_dict)
    """
    url = citation_item[0]
    data_dict = citation_item[1]

    publish_time = data_dict.get('publish_time')

    score = calculate_time_score(publish_time, alpha)

    return {
        "url": url,
        "time_decay_score": score,
        "publish_time_raw": publish_time  # 可选：保留原始时间方便debug
    }


# ==========================================
# 主流程
# ==========================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to raw data or validated data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save data with time scores_dynamic")
    parser.add_argument("--alpha", type=float, default=0.002,
                        help="Decay rate alpha. Default 0.002 (approx 1 year half-life)")

    args = parser.parse_args()


def save_time_score(data_list, save_path, alpha):

    if os.path.exists(save_path):
        os.remove(save_path)

    print(f"Processing {len(data_list)} documents for time decay evaluation...")

    for d in tqdm(data_list):
        # 获取所有引用
        citations = d.get('citations_deduped', {})

        # 遍历每一个引用链接
        for url, content in citations.items():
            # 计算分数
            score = calculate_time_score(content.get('publish_time'), alpha)

            # 将分数直接写入原数据结构中
            # 结果保存在 citations_deduped[url]['time_decay_score']
            content['time_decay_score'] = score

        with open(save_path, 'a+', encoding='utf-8') as f_out:
            # 写入文件
            f_out.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Done! Results saved to {save_path}")