import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
import argparse
import math
import os
from datetime import datetime, timezone
from tqdm import tqdm
from dateutil import parser as date_parser  # Powerful time parsing library
from evaluation_toolkit.subjective_eval.FACT.utils import *


# ==========================================
# Core calculation functions
# ==========================================

def reference_time(as_of=None):
    """Resolve a frozen UTC reference; ISO dates denote midnight UTC."""
    if as_of is None:
        return datetime.now(timezone.utc)
    try:
        value = as_of if isinstance(as_of, datetime) else date_parser.isoparse(as_of)
    except (ValueError, TypeError) as exc:
        raise ValueError("as_of must be an ISO date or datetime, e.g. 2026-10-16") from exc
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def calculate_time_score(publish_time_str, alpha=0.002, as_of=None):
    """exp(-alpha * age in whole UTC days); unknown dates return -1.

    Partial publication dates use January 1 / day 1 for missing components.
    Future dates have age zero, consistent with the original scoring convention.
    """
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be a finite nonnegative number')
    current = reference_time(as_of)
    if not publish_time_str:
        return -1
    try:
        published = date_parser.parse(str(publish_time_str), fuzzy=True,
                                      default=datetime(1970, 1, 1, tzinfo=timezone.utc))
        published = reference_time(published)
        days = max(0, (current - published).days)
        return round(math.exp(-alpha * days), 4)
    except (ValueError, TypeError, OverflowError):
        return -1


def process_citation_time(citation_item, alpha, as_of=None):
    """
    Process a single citation item, compatible with validate.py data structure
    citation_item: (url, data_dict)
    """
    url = citation_item[0]
    data_dict = citation_item[1]

    publish_time = data_dict.get('publish_time')

    score = calculate_time_score(publish_time, alpha, as_of)

    return {
        "url": url,
        "time_decay_score": score,
        "publish_time_raw": publish_time  # Optional: keep raw time for debugging
    }


# ==========================================
# Main pipeline
# ==========================================


def save_time_score(data_list, save_path, alpha, as_of=None):
    as_of = reference_time(as_of).isoformat()
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be a finite nonnegative number')
    print(f"Processing {len(data_list)} documents for time decay evaluation...")
    for record in tqdm(data_list):
        record['evaluation_as_of_utc'] = as_of
        for citation in record.get('citations_deduped', {}).values():
            citation['time_decay_score'] = calculate_time_score(citation.get('publish_time'), alpha, as_of)
    write_jsonl(save_path, data_list)
    print(f"Done! Results saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Raw or validated JSONL file")
    parser.add_argument("--output_path", required=True, help="Output JSONL with time scores")
    parser.add_argument("--alpha", type=float, default=0.002, help="Decay rate per day")
    parser.add_argument("--as_of", help="Frozen evaluation date/datetime in ISO format (UTC by default)")
    args = parser.parse_args()
    save_time_score(load_jsonl(args.input_path), args.output_path, args.alpha, args.as_of)
