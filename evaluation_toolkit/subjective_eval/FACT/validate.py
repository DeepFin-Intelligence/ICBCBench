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
from evaluation_toolkit.subjective_eval.FACT.utils import load_jsonl
from evaluation_toolkit.utils import load_local_dataset
from evaluation_toolkit.model_clients import *

import platform

client = MYDMX

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

def validate(data, id_to_lang_map):
    url = data[0]
    ref = data[1]['url_content']
    facts = data[1]['facts']
    article_id = data[1].get('article_id')

    if ref is None:
        return {
            "url": url,
            "validate_res": [],
            "error": "no reference"
        }
    
    # Skip LLM validation if scraping failed
    if isinstance(ref, str) and ref.startswith('scrape failed'):
        return {
            "url": url,
            "validate_res": [{"idx": i + 1, "result": "unknown"} for i in range(len(facts))],
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
                model="gemini-3-flash-preview",
                messages=[{"role": "user", "content": user_prompt}],
                response_format={'type': 'json_object'},
            )

            response = response.choices[0].message.content

            validate_res = json.loads(response.replace("```json", "").replace("```", ""))
            for _v in validate_res:
                _v['idx'] -= 1
            assert len(validate_res) == len(facts)

            return {
                "url": url,
                "validate_res": validate_res,
                "error": None
            }
        except Exception as e:
            error = str(e)
            time.sleep(3)
            retries += 1
    
    return {
        "url": url,
        "validate_res": [],
        "error": error
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
    parser.add_argument("--query_data_path", type=str, required=True, help="Path to query data with language information")
    parser.add_argument("--n_total_process", type=int, default=1)
    args = parser.parse_args()
    
    output_path = args.output_path
    raw_data = load_jsonl(args.raw_data_path)
    
    # Load the query data to get language information
    query_data = load_local_dataset(args.query_data_path)
    query_data = [dict(zip(query_data.keys(), values)) for values in zip(*query_data.values())]
    
    # Create a mapping from ID to language
    id_to_lang_map = {item['id']: item['language'] for item in query_data if 'id' in item}
    
    if not id_to_lang_map:
        raise ValueError("No valid language information found in query data")
    
    n_total_process = args.n_total_process

    # URL-level resume: merge existing validated data
    existing_docs = {}
    if os.path.exists(output_path):
        for d in load_jsonl(output_path):
            existing_docs[d['id']] = d
        # merge existing citations_deduped into raw_data
        for d in raw_data:
            if d['id'] in existing_docs:
                existing_citations = existing_docs[d['id']].get('citations_deduped', {})
                for url, content in existing_citations.items():
                    if url in d['citations_deduped'] and content.get('validate_res') is not None:
                        d['citations_deduped'][url] = content

    data_to_process = raw_data

    print(f"Processing {len(data_to_process)} instances...")
    
    all_results = []

    for d in tqdm(data_to_process):
        # get the citations that need to be validated (URL-level filtering)
        # retry URLs with validate_error on rerun
        citations = [(k, v) for k, v in d['citations_deduped'].items() 
                     if 'validate_res' not in v or v.get('validate_error') is not None]
        
        # Add article_id to each citation's value for language determination
        article_id = d.get('id')
        if not article_id:
            print(f"Warning: Article has no ID field, skipping validation")
            all_results.append(d)
            continue
            
        for citation in citations:
            citation[1]['article_id'] = article_id

        if citations:
            if n_total_process == 1:
                results = [validate(citation, id_to_lang_map) for citation in citations]
            elif n_total_process > 1:
                run_partial = partial(validate, id_to_lang_map=id_to_lang_map)
                with multiprocessing.Pool(processes=n_total_process) as pool:
                    results = pool.map(run_partial, citations)
            
            for res in results:
                d['citations_deduped'][res['url']]['validate_res'] = res['validate_res']
                d['citations_deduped'][res['url']]['validate_error'] = res['error']
        
        all_results.append(d)

    # rewrite the entire output file to avoid duplicate IDs
    with open(output_path, 'w', encoding='utf-8') as f:
        for d in all_results:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

