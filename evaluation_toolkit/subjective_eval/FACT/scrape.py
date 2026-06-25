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
from evaluation_toolkit.subjective_eval.FACT.utils import load_jsonl, scrape_url
from evaluation_toolkit.utils import load_local_dataset
from evaluation_toolkit.model_clients import *

load_dotenv()

def scrape(citation_url):
    max_retries = 3
    retries = 0
    while retries < max_retries:
        result = scrape_url(citation_url)
        retries += 1
        if 'error' in result:
            wait_time = 2 ** retries  # 指数退避: 2s, 4s, 8s
            print(f"scrape failed for {citation_url}, retrying in {wait_time}s... (attempt {retries}/{max_retries})")
            time.sleep(wait_time)
        else:
            break

    url_content = None
    if 'error' not in result:
        true_url = result.get('url', citation_url)
        title = result.get('title', '')
        content = result.get('content', '')
        description = result.get('description', '')
        publish_time = result.get('publish_time', 'unknown')

        url_content = f"{title}\n\n{description}\n\n{content}"
    else:
        url_content = f"scrape failed: {result.get('error', 'unknown error')}"
        publish_time = 'unknown'
        true_url = citation_url
    # print(f"url_content: {url_content}")
    return {
        'url': citation_url,
        'true_url': true_url,
        'url_content': url_content,
        'publish_time': publish_time
    }


if __name__ == '__main__':
    # if platform.system() == 'Darwin':
    #     try:
    #         multiprocessing.set_start_method('spawn')
    #     except RuntimeError:
    #         pass
    # else:
    #     try:
    #         multiprocessing.set_start_method('fork')
    #     except RuntimeError:
    #         pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument("--n_total_process", type=int, default=1)
    args = parser.parse_args()
    
    output_path = args.output_path
    
    # initialize variables
    raw_data = []
    data_to_process = []
    processed = []
    
    try:
        raw_data = load_jsonl(args.raw_data_path)
        
        # URL-level resume: merge existing scraped data
        existing_docs = {}
        if os.path.exists(output_path):
            for d in load_jsonl(output_path):
                existing_docs[d['id']] = d
            # merge existing citations_deduped into raw_data
            for d in raw_data:
                if d['id'] in existing_docs:
                    existing_citations = existing_docs[d['id']].get('citations_deduped', {})
                    for url, content in existing_citations.items():
                        if url in d['citations_deduped'] and content.get('url_content'):
                            d['citations_deduped'][url] = content
        
        data_to_process = raw_data
    except Exception as e:
        import sys
        print(f"cannot process file {args.raw_data_path}: {e}")
        sys.exit(f'{args.raw_data_path} has not been processed yet...')
    
    print(f"processing {len(data_to_process)} instances...")
    
    all_results = []

    def needs_scrape(v):
            if 'url_content' not in v or not v['url_content']:
                return True
            if isinstance(v['url_content'], str) and v['url_content'].startswith('scrape failed'):
                return True
            return False
    
    for d in tqdm(data_to_process):
        # get the citations that need to be scraped (URL-level filtering)
        # scrape failed URLs will be retried (they might succeed next time)
        citations = list([k for k, v in d['citations_deduped'].items() if needs_scrape(v)])
        results = []

        if citations:
            n_total_process = min(args.n_total_process, len(citations))

            if n_total_process == 1:
                results = [scrape(citation) for citation in citations]
            elif n_total_process > 1:
                try:
                    with ThreadPoolExecutor(max_workers=n_total_process) as executor:
                        results = list(executor.map(scrape, citations))
                except KeyboardInterrupt:
                    print("scrape process interrupted")
                    continue
                except Exception as e:
                    print(f"scrape process error: {e}")
                    # if error, fallback to single thread
                    results = [scrape(citation) for citation in citations]

            # update the url_content
            for res in results:
                d['citations_deduped'][res['url']]['true_url'] = res['true_url']
                d['citations_deduped'][res['url']]['url_content'] = res['url_content']
                d['citations_deduped'][res['url']]['publish_time'] = res['publish_time']
        
        all_results.append(d)

    # rewrite the entire output file to avoid duplicate IDs
    with open(output_path, 'w', encoding='utf-8') as f:
        for d in all_results:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
