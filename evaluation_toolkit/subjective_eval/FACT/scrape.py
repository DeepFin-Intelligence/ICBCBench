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
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from evaluation_toolkit.subjective_eval.FACT.utils import load_jsonl, scrape_url, write_jsonl, records_by_id, validation_cache_matches
from evaluation_toolkit.utils import load_local_dataset
from evaluation_toolkit.model_clients import *


def scrape(citation_url):
    result = {'error': 'No scrape attempt completed'}
    for attempt in range(3):
        try:
            result = scrape_url(citation_url)
        except Exception as exc:
            result = {'error': str(exc)}
        if 'error' not in result and isinstance(result.get('content'), str) and result['content'].strip():
            break
        result.setdefault('error', 'Reference content is empty')
        if attempt < 2:
            time.sleep(2 ** (attempt + 1))
    if 'error' in result:
        return {'url': citation_url, 'true_url': citation_url,
                'url_content': f'scrape failed: {result["error"]}',
                'publish_time': 'unknown', 'scrape_error': result['error']}
    return {'url': citation_url, 'true_url': result.get('url', citation_url),
            'url_content': '\n\n'.join(result.get(key, '') or '' for key in ('title', 'description', 'content')),
            'publish_time': result.get('publish_time') or 'unknown', 'scrape_error': None}


def needs_scrape(citation):
    content = citation.get('url_content')
    return not isinstance(content, str) or not content.strip() or content.startswith('scrape failed') or bool(citation.get('scrape_error'))


def merge_scrape_cache(raw_data, existing_docs):
    for record in raw_data:
        previous = existing_docs.get(str(record['id']), {})
        for url, citation in record.get('citations_deduped', {}).items():
            cached = previous.get('citations_deduped', {}).get(url, {})
            if not needs_scrape(cached) and needs_scrape(citation):
                # Keep current facts and validation; reuse only fetched source fields.
                for field in ('url_content', 'true_url', 'publish_time', 'scrape_error'):
                    if field in cached:
                        citation[field] = cached[field]
                if citation.get('facts') == cached.get('facts') and validation_cache_matches(citation, cached):
                    citation['validate_res'] = cached['validate_res']
                    citation['validate_error'] = None
                    if cached.get('validation_model'):
                        citation['validation_model'] = cached['validation_model']
                else:
                    citation.pop('validate_res', None)
                    citation.pop('validate_error', None)
                    citation.pop('validation_model', None)
    return raw_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--raw_data_path', required=True)
    parser.add_argument('--n_total_process', type=int, default=1)
    args = parser.parse_args()
    if args.n_total_process < 1:
        parser.error('--n_total_process must be at least 1')
    raw_data = list(records_by_id(load_jsonl(args.raw_data_path)).values())
    for record in raw_data:
        if not isinstance(record.get('citations_deduped'), dict):
            record['citations_deduped'] = {}
            record['extraction_error'] = 'Missing or invalid citations_deduped input'
    existing = records_by_id(load_jsonl(args.output_path)) if os.path.exists(args.output_path) else {}
    merge_scrape_cache(raw_data, existing)
    write_jsonl(args.output_path, raw_data)
    print(f'Processing {len(raw_data)} reports...')
    with ThreadPoolExecutor(max_workers=args.n_total_process) as executor:
        for record in tqdm(raw_data):
            citations = [url for url, value in record.get('citations_deduped', {}).items() if needs_scrape(value)]
            for result in executor.map(scrape, citations):
                citation = record['citations_deduped'][result['url']]
                old_content = citation.get('url_content')
                for field in ('url_content', 'true_url', 'publish_time', 'scrape_error'):
                    citation[field] = result[field]
                if old_content != citation['url_content']:
                    citation.pop('validate_res', None)
                    citation.pop('validate_error', None)
                    citation.pop('validation_model', None)
                write_jsonl(args.output_path, raw_data)
    failed = sum(needs_scrape(citation) for record in raw_data
                 for citation in record.get('citations_deduped', {}).values())
    failed += sum(bool(record.get('dedup_error') or record.get('extraction_error')) for record in raw_data)
    if failed:
        print(f'Citation scraping incomplete: {failed} failure(s); rerun to retry.')
    sys.exit(1 if failed else 0)
