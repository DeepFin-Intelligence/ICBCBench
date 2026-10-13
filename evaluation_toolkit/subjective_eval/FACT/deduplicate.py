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
import argparse
from functools import partial
from evaluation_toolkit.utils import load_local_dataset
from evaluation_toolkit.model_clients import *
from evaluation_toolkit.subjective_eval.FACT.utils import load_jsonl, write_jsonl, records_by_id, validate_citations, valid_deduplication_cache

client = OPENROUTER

prompt_template = """你会看到一个statement列表，你需要对其去重，并返回去重后的statement序号列表，注意：只有表达完全一致的事情时，两个statement才被认为是重复的，如果列表中没有重复的statement，则返回完整的列表。

你应该返回一个List(int)，列表中的每一项是去重后留下的，不重复的statement的序号，例如：
[1, 3, 5]

下面是你需要去重的statement列表
{statements}

下面开始提取，直接输出整数列表，不要输出任何闲聊或解释。"""


prompt_template_en = """You will be given a list of statements. You need to de-duplicate them and return a list of indices of the unique statements. Note: Two statements are considered duplicates only if they express *exactly the same thing*. If there are no duplicate statements in the list, return the complete list of indices.

You should return a List(int), where each item in the list is the index of a unique, non-duplicated statement that has been retained. For example:
[1, 3, 5]

Below is the list of statements you need to de-duplicate:
{statements}

Please begin the extraction now. Output only the integer list, without any conversational text or explanations."""


def deduplicate_report(d, id_to_lang_map, judge_model="google/gemini-3-flash-preview"):
    d = dict(d)
    d['id'] = str(d.get('id', ''))
    d['dedup_model'] = resolve_openrouter_model(judge_model)
    d.pop('dedup_error', None)
    try:
        raw_citations = d.get('citations')
        if not isinstance(raw_citations, list):
            raise ValueError('Citation extraction failed; rerun extraction first.')
        # The reference index is extraction metadata, not needed for URL grouping.
        citations = validate_citations([dict(item, ref_idx=item.get('ref_idx', 0)) for item in raw_citations])
        language = id_to_lang_map[d['id']]
        if language not in ('zh', 'en'):
            raise ValueError(f'Unsupported language: {language}')
    except Exception as exc:
        d['citations_deduped'] = {}
        d['dedup_error'] = str(exc)
        return d, 1
    groups = {}
    for citation in citations:
        groups.setdefault(citation['url'], []).append(citation)
    deduped = {}
    for url, group in groups.items():
        indices = [1] if len(group) == 1 else None
        if indices is None:
            template = prompt_template if language == 'zh' else prompt_template_en
            statements = '\n'.join(f'{index + 1}. {item["fact"]}' for index, item in enumerate(group))
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=resolve_openrouter_model(judge_model),
                        messages=[{'role': 'user', 'content': template.format(statements=statements)}],
                        max_tokens=65536)
                    if getattr(response.choices[0], 'finish_reason', None) in ('length', 'content_filter'):
                        raise ValueError('Deduplication response did not finish normally')
                    text = response.choices[0].message.content
                    candidate = json.loads(text.replace('```json', '').replace('```', ''))
                    if (not isinstance(candidate, list) or not candidate
                            or any(type(index) is not int or not 1 <= index <= len(group) for index in candidate)
                            or len(set(candidate)) != len(candidate)):
                        raise ValueError('Invalid deduplication indices')
                    indices = sorted(candidate)
                    break
                except Exception as exc:
                    print(f'Deduplication error for {d["id"]}, attempt {attempt + 1}/3: {exc}')
        if indices is None:
            indices = list(range(1, len(group) + 1))
            d['dedup_error'] = 'Semantic deduplication failed; all facts retained for retry.'
        deduped[url] = {'facts': [group[index - 1]['fact'] for index in indices], 'url_content': None}
    d['citations_deduped'] = deduped
    return d, int(bool(d.get('dedup_error')))


def run(data, output_path, id_to_lang_map, judge_model="google/gemini-3-flash-preview"):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    failed = 0
    with open(output_path, 'a', encoding='utf-8') as handle:
        for item in data:
            record, error = deduplicate_report(item, id_to_lang_map, judge_model)
            failed += error
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
    return failed


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
    query_data = load_local_dataset(args.query_data_path)
    queries = [dict(zip(query_data.keys(), row)) for row in zip(*query_data.values())]
    id_to_lang_map = {str(item['id']): item['language'] for item in queries}
    if not id_to_lang_map:
        raise ValueError('No valid language information found in query data')
    raw_data = records_by_id(load_jsonl(args.raw_data_path))
    missing = set(id_to_lang_map) - set(raw_data)
    print(f'FACT input coverage: {len(set(raw_data) & set(id_to_lang_map))}/{len(id_to_lang_map)} reports; {len(missing)} missing.')
    for identifier in missing:
        raw_data[identifier] = {'id': identifier, 'citations': 'extraction failed',
                                'extraction_error': 'Missing report in input'}
    existing = records_by_id(load_jsonl(args.output_path)) if os.path.exists(args.output_path) else {}
    successful = {identifier: existing[identifier] for identifier, item in raw_data.items()
                  if identifier in existing and not existing[identifier].get('dedup_error')
                  and existing[identifier].get('dedup_model') == resolve_openrouter_model(args.judge)
                  and existing[identifier].get('citations') == item.get('citations')
                  and valid_deduplication_cache(item, existing[identifier])}
    write_jsonl(args.output_path, successful.values())
    pending = [item for identifier, item in raw_data.items() if identifier not in successful]
    print(f'Processing {len(pending)} reports...')
    worker = partial(deduplicate_report, id_to_lang_map=id_to_lang_map, judge_model=args.judge)
    failed = 0
    pool = multiprocessing.Pool(args.n_total_process) if args.n_total_process > 1 else None
    try:
        results = pool.imap(worker, pending) if pool else map(worker, pending)
        for record, error in results:
            failed += error
            successful[record['id']] = record
            write_jsonl(args.output_path, successful.values())
    finally:
        if pool:
            pool.close()
            pool.join()
    if failed:
        print(f'Citation deduplication incomplete: {failed} report(s) failed; rerun to retry.')
    sys.exit(1 if failed else 0)
