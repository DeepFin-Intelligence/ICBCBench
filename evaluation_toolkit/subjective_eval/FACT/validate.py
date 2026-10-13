import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p.name != "evaluation_toolkit" and _p.parent != _p:
    _p = _p.parent
if _p.name == "evaluation_toolkit":
    _root = _p.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import multiprocessing
import json
import os
import time
import argparse
from tqdm import tqdm
from functools import partial
from evaluation_toolkit.subjective_eval.FACT.utils import load_jsonl, write_jsonl, records_by_id, normalize_validation, validation_cache_matches
from evaluation_toolkit.utils import load_local_dataset
from evaluation_toolkit.model_clients import *

import platform

client = OPENROUTER

prompt_template = """你会看到一个参考资料和一些statement，请你判断对于参考资料来说statement是supported、unsupported、或者unknown，注意：
首先判断参考资料是否存在有效内容，如果参考资料中没有任何有效信息，如"page not found"页面，则认为所有statement的状态都是unknown。
除此之外，参考资料有效的情况下，对于一个statement来说，如果它包含的事实或数据在参考资料中可以全部或部分找到，就认为它是supported的（数据接受四舍五入）；如果statement中所有的事实和数据在参考资料中都找不到，认为它是unsupported的。

你应该返回json列表格式，列表中的每一项包含statement的序号和判断结果，例如：
[
    {{
        "idx": 1,
        "result": "supported"
    }},
    {{
        "idx": 2,
        "result": "unsupported"
    }}
]

下面是参考资料和statements：
<reference>
{reference}
</reference>

<statements>
{statements}
</statements>

下面开始判断，直接输出json列表，不要输出任何闲聊或解释。"""

prompt_template_en = """You will be provided with a reference and some statements. Please determine whether each statement is 'supported', 'unsupported', or 'unknown' with respect to the reference. Please note:
First, assess whether the reference contains any valid content. If the reference contains no valid information, such as a 'page not found' message, then all statements should be considered 'unknown'.
If the reference is valid, for a given statement: if the facts or data it contains can be found entirely or partially within the reference, it is considered 'supported' (data accepts rounding); if all facts and data in the statement cannot be found in the reference, it is considered 'unsupported'.

You should return the result in a JSON list format, where each item in the list contains the statement's index and the judgment result, for example:
[
    {{
        "idx": 1,
        "result": "supported"
    }},
    {{
        "idx": 2,
        "result": "unsupported"
    }}
]

Below are the reference and statements:
<reference>
{reference}
</reference>

<statements>
{statements}
</statements>

Begin the assessment now. Output only the JSON list, without any conversational text or explanations."""

def validate(data, id_to_lang_map, judge_model="google/gemini-3-flash-preview"):
    url = data[0]
    ref = data[1].get('url_content')
    facts = data[1].get('facts')
    article_id = data[1].get('article_id')

    if not isinstance(facts, list) or any(not isinstance(fact, str) or not fact.strip() for fact in facts):
        return {'url': url, 'validate_res': [], 'error': 'Facts must be a list of nonempty text'}
    if not facts:
        return {'url': url, 'validate_res': [], 'error': None}
    if not isinstance(ref, str) or not ref.strip():
        return {
            "url": url,
            "validate_res": [],
            "error": "no reference"
        }
    
    # Skip LLM validation if scraping failed
    if isinstance(ref, str) and ref.startswith('scrape failed'):
        return {
            "url": url,
            "validate_res": [{"idx": i, "result": "unknown"} for i in range(len(facts))],
            "error": None
        }
    
    # Determine language based on article ID
    if not article_id:
        error_msg = "Article ID not provided for language determination"
        print(error_msg)
        return {
            "url": url,
            "validate_res": [],
            "error": error_msg
        }
    
    if article_id not in id_to_lang_map:
        error_msg = f"Language not found for article ID: {article_id}"
        print(error_msg)
        return {
            "url": url,
            "validate_res": [],
            "error": error_msg
        }
    
    lang = id_to_lang_map[article_id]
    
    facts_str = '\n'.join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])

    if lang == "zh":
        user_prompt = prompt_template.format(reference=ref, statements=facts_str)
    elif lang == "en":
        user_prompt = prompt_template_en.format(reference=ref, statements=facts_str)
    else:
        error_msg = f"Unsupported language: {lang}"
        print(error_msg)
        return {
            "url": url,
            "validate_res": [],
            "error": error_msg
        }
    
    retries = 0
    error = None
    while retries < 3:
        try:
            # response = call_model(user_prompt)
            response = client.chat.completions.create(
                model=resolve_openrouter_model(judge_model),
                messages=[{"role": "user", "content": user_prompt}],
            )
            if getattr(response.choices[0], 'finish_reason', None) in ('length', 'content_filter'):
                raise ValueError('Validation response did not finish normally')

            response = response.choices[0].message.content

            validate_res = normalize_validation(
                json.loads(response.replace("```json", "").replace("```", "")),
                len(facts), one_based=True)

            return {
                "url": url,
                "validate_res": validate_res,
                "error": None
            }
        except Exception as e:
            error = str(e)
            retries += 1
            if retries < 3:
                time.sleep(3)
    
    return {
        "url": url,
        "validate_res": [],
        "error": error
    } 


def merge_validation_cache(raw_data, existing_docs, judge_model=None):
    for record in raw_data:
        previous = existing_docs.get(str(record['id']), {})
        for url, citation in record.get('citations_deduped', {}).items():
            cached = previous.get('citations_deduped', {}).get(url, {})
            if validation_cache_matches(citation, cached, judge_model):
                citation['validate_res'] = normalize_validation(cached['validate_res'], len(citation['facts']))
                citation['validate_error'] = None
                if cached.get('validation_model'):
                    citation['validation_model'] = cached['validation_model']
            elif not validation_cache_matches(citation, citation, judge_model):
                citation.pop('validate_res', None)
                citation.pop('validate_error', None)
                citation.pop('validation_model', None)
    return raw_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--raw_data_path', required=True)
    parser.add_argument('--query_data_path', required=True)
    parser.add_argument('--n_total_process', type=int, default=1)
    parser.add_argument('--judge', default='google/gemini-3-flash-preview')
    args = parser.parse_args()
    if args.n_total_process < 1:
        parser.error('--n_total_process must be at least 1')
    raw_data = list(records_by_id(load_jsonl(args.raw_data_path)).values())
    for record in raw_data:
        if not isinstance(record.get('citations_deduped'), dict):
            record['citations_deduped'] = {}
            record['extraction_error'] = 'Missing or invalid citations_deduped input'
    queries = load_local_dataset(args.query_data_path)
    query_data = [dict(zip(queries.keys(), row)) for row in zip(*queries.values())]
    id_to_lang_map = {str(item['id']): item['language'] for item in query_data}
    if not id_to_lang_map:
        raise ValueError('No valid language information found in query data')
    missing = set(id_to_lang_map) - {record['id'] for record in raw_data}
    print(f'FACT input coverage: {len(id_to_lang_map) - len(missing)}/{len(id_to_lang_map)} reports; {len(missing)} missing.')
    raw_data.extend({'id': identifier, 'citations_deduped': {},
                     'extraction_error': 'Missing report in input'} for identifier in sorted(missing))
    existing_docs = records_by_id(load_jsonl(args.output_path)) if os.path.exists(args.output_path) else {}
    canonical_judge = resolve_openrouter_model(args.judge)
    merge_validation_cache(raw_data, existing_docs, canonical_judge)
    print(f'Processing {len(raw_data)} reports...')
    worker = partial(validate, id_to_lang_map=id_to_lang_map, judge_model=args.judge)
    pool = multiprocessing.Pool(args.n_total_process) if args.n_total_process > 1 else None
    try:
        write_jsonl(args.output_path, raw_data)
        for record in tqdm(raw_data):
            citations = []
            for url, citation in record.get('citations_deduped', {}).items():
                if not validation_cache_matches(citation, citation, canonical_judge):
                    citation['article_id'] = record['id']
                    citations.append((url, citation))
            results = pool.imap(worker, citations) if pool else map(worker, citations)
            for result in results:
                citation = record['citations_deduped'][result['url']]
                citation['validate_res'] = result['validate_res']
                citation['validate_error'] = result['error']
                citation['validation_model'] = canonical_judge
                write_jsonl(args.output_path, raw_data)
    finally:
        if pool:
            pool.close()
            pool.join()
    failed = sum(bool(citation.get('validate_error') is not None or citation.get('scrape_error')
                 or (citation.get('url_content') or '').startswith('scrape failed')) for record in raw_data
                 for citation in record.get('citations_deduped', {}).values())
    failed += sum(bool(record.get('dedup_error') or record.get('extraction_error')) for record in raw_data)
    if failed:
        print(f'Citation validation incomplete: {failed} failure(s); rerun to retry.')
    sys.exit(1 if failed else 0)
