import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
import argparse
import math
import os
from datetime import datetime
from tqdm import tqdm
from dateutil import parser as date_parser  # Powerful time parsing library
from evaluation_toolkit.subjective_eval.FACT.utils import *


# ==========================================
# Core calculation functions
# ==========================================

def calculate_time_score(publish_time_str, alpha=0.002):
    """
    Calculate time decay score based on publication time.
    Formula: Score = e^(-alpha * (T_curr - T_pub))
    Unit: days
    """
    # 1. If no publication time, return None or default score
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be a finite nonnegative number')
    if not publish_time_str:
        return -1

    try:
        # 2. Parse time string
        # fuzzy=True handles non-standard strings like "Published: 2023-10-01"
        pub_time = date_parser.parse(str(publish_time_str), fuzzy=True)

        # Remove timezone info for comparison (or convert to UTC; here simplified to naive time)
        pub_time = pub_time.replace(tzinfo=None)
        curr_time = datetime.now().replace(tzinfo=None)

        # 3. Calculate day difference
        delta = curr_time - pub_time
        days_diff = delta.days

        # If publication time is in the future (dirty data), treat as 0 days
        if days_diff < 0:
            days_diff = 0

        # 4. Exponential decay formula
        score = math.exp(-alpha * days_diff)

        return round(score, 4)

    except Exception as e:
        # Parse failed, return default score, log optionally
        # print(f"Time parse error: {e} | Raw string: {publish_time_str}")
        return -1


def process_citation_time(citation_item, alpha):
    """
    Process a single citation item, compatible with validate.py data structure
    citation_item: (url, data_dict)
    """
    url = citation_item[0]
    data_dict = citation_item[1]

    publish_time = data_dict.get('publish_time')

    score = calculate_time_score(publish_time, alpha)

    return {
        "url": url,
        "time_decay_score": score,
        "publish_time_raw": publish_time  # Optional: keep raw time for debugging
    }


# ==========================================
# Main pipeline
# ==========================================


def save_time_score(data_list, save_path, alpha):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be a finite nonnegative number')
    print(f"Processing {len(data_list)} documents for time decay evaluation...")
    for record in tqdm(data_list):
        for citation in record.get('citations_deduped', {}).values():
            citation['time_decay_score'] = calculate_time_score(citation.get('publish_time'), alpha)
    write_jsonl(save_path, data_list)
    print(f"Done! Results saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Raw or validated JSONL file")
    parser.add_argument("--output_path", required=True, help="Output JSONL with time scores")
    parser.add_argument("--alpha", type=float, default=0.002, help="Decay rate per day")
    args = parser.parse_args()
    save_time_score(load_jsonl(args.input_path), args.output_path, args.alpha)
