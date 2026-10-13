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
import re
from functools import partial
from evaluation_toolkit.utils import load_local_dataset
from evaluation_toolkit.model_clients import *
from evaluation_toolkit.subjective_eval.FACT.utils import load_jsonl, write_jsonl, records_by_id, validate_citations
from evaluation_toolkit.subjective_eval.runtime import validate_questions
from evaluation_toolkit.subjective_eval.scoring.provenance import question_fingerprint, report_fingerprint


client = OPENROUTER


prompt_template = """你会看到一篇研究报告，研究报告正文中会有一些对参考文献的引用。
正文中的引用可能以如下形式出现：
1. 一段文字+空格+数字，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级 15"
2. 一段文字+[（一个或多个)数字]，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级[15]"
3. 一段文字+[（一个或多个)数字†(一些行号等内容)]，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级[15†L10][5L23][7†summary][9summary]"
4. [引用来源](引用链接)，例如："根据[ChinaFile: A Guide to Social Class in Modern China](https://www.chinafile.com/reporting-opinion/media/guide-social-class-modern-china)'s分类，中国社会可分为九个阶层"

请从正文中找出**所有**引用了参考文献的地方，提取出(fact, ref_idx, url)三元组，提取的时候，注意以下事项：
1. 由于后续需要检验这些facts是否正确，你可能需要在引用的前后寻找一些上下文，以确保fact是完整可理解的，而不是简单的词组或短语
2. 如果一个fact引用了多个文献，那么它应该对应多个三元组，例如如果引用了2个文献，则应该是(fact, ref_idx_1, url_1)和(fact, ref_idx_2, url_2)
3. 对于第三种形式的引用，ref_idx仅考虑第一个数字部分，不考虑其他指示具体位置的内容；对于第四种形式的引用（即引用来源和链接直接出现在正文中）的情况，ref_idx统一设置为0
4. 如果正文中没有标出引用的具体位置（比如仅在文章结尾列出了参考文献列表，而没有在正文中标出），请返回空列表

你应该返回json列表格式，列表中的每一项是一个三元组，例如：
[
    {{
        "fact": "原文中的文本片段，注意中文引号要用全角, 英文引号前加单个反斜杠转义",
        "ref_idx": "该段文字引用的参考文献在参考文献列表中的索引",
        "url": "该段文字引用的参考文献链接（从研究报告结尾的参考文献列表或引用处的括号中提取）"
    }}
]

下面是研究报告的正文：
{report_text}

下面开始提取，直接输出json列表，不要输出任何额外解释。"""


prompt_template_en = """You will be provided with a research report. The body of the report will contain some citations to references.

Citations in the main text may appear in the following forms:
1. A segment of text + space + number, for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels 15"
2. A segment of text + [number], for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels[15]"
3. A segment of text + [number†(some line numbers, etc.)], for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels[15†L10][5L23][7†summary]"
4. [Citation Source](Citation Link), for example: "According to [ChinaFile: A Guide to Social Class in Modern China](https://www.chinafile.com/reporting-opinion/media/guide-social-class-modern-china)'s classification, Chinese society can be divided into nine strata"

Please identify **all** instances where references are cited in the main text, and extract (fact, ref_idx, url) triplets. When extracting, pay attention to the following:
1. Since these facts will need to be verified later, you may need to look for some context before and after the citation to ensure that the fact is complete and understandable, rather than just a simple phrase or short expression.
2. If a fact cites multiple references, then it should correspond to two triplets: (fact, ref_idx_1, url_1) and (fact, ref_idx_2, url_2).
3. For the fourth form of citation (i.e., where the citation source and link appear directly in the text), the ref_idx should be uniformly set to 0.
4. If the main text does not specify the exact location of the citation (for example, only the reference list is listed at the end of the article, without specifying the citation point in the text), please return an empty list.

You should return a JSON list format, where each item in the list is a triplet, for example:
[
    {{
        "fact": "Text segment from the original document. Note that Chinese quotation marks should use full-width marks. And add a single backslash before the English quotation mark to make it a readable for python json module.",
        "ref_idx": "The index of the cited reference in the reference list for this text segment.",
        "url": "The URL of the cited reference for this text segment (extracted from the reference list at the end of the research report or from the parentheses at the citation point)."
    }}
]

Here is the main text of the research report:
{report_text}

Please begin the extraction now. Output only the JSON list directly, without any chitchat or explanations."""

def clean_urls(input_text):
    # match [title](url) format
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    def repl(match):
        title = match.group(1)
        url = match.group(2)
        # truncate #:~:text= and its content
        cut_idx = url.find('#:~:text=')
        if cut_idx != -1:
            url = url[:cut_idx]
        return f'[{title}]({url})'
    
    return pattern.sub(repl, input_text)


def remove_urls(input_text):
    # match [title](url) format, only remove the content in the parentheses, keep [title]
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    # replace [title](url) with [title]
    return pattern.sub(r'[\1]', input_text)


def clean_escape(input_text):
    # replace illegal escape characters
    input_text = input_text.replace("\\>", ">")
    input_text = input_text.replace("\\<", "<")
    input_text = input_text.replace("\\+", "+")
    input_text = input_text.replace("\\~", "~")
    return input_text

def safe_json_parse(response_text):
    """Reject incomplete JSON rather than silently dropping citations."""
    return json.loads(response_text)


def extract_report(d, id_to_lang_map, judge_model="google/gemini-3-flash-preview"):
    d = dict(d)
    article_id = str(d.get('id', ''))
    d['id'] = article_id
    d['extraction_model'] = resolve_openrouter_model(judge_model)
    error = None
    try:
        language = id_to_lang_map[article_id]
        if language not in ('zh', 'en'):
            raise ValueError(f"Unsupported language: {language}")
        if not isinstance(d.get('response'), str) or not d['response'].strip():
            raise ValueError('Report response is empty')
        if d.get('citation_mapping_error'):
            raise ValueError(d['citation_mapping_error'])
        template = prompt_template if language == 'zh' else prompt_template_en
        user_prompt = template.format(report_text=d['response'])
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=resolve_openrouter_model(judge_model),
                    messages=[{"role": "user", "content": user_prompt}], max_tokens=65536)
                if getattr(response.choices[0], 'finish_reason', None) in ('length', 'content_filter'):
                    raise ValueError('Extraction response did not finish normally')
                text = response.choices[0].message.content
                if not isinstance(text, str) or not text.strip():
                    raise ValueError('Empty extraction response')
                # Partial/truncated JSON must never silently become complete output.
                citations = validate_citations(json.loads(clean_escape(text.replace('```json', '').replace('```', ''))))
                for citation in citations:
                    citation['fact'] = remove_urls(citation['fact'])
                d['citations'] = citations
                d.pop('extraction_error', None)
                return d, 0
            except Exception as exc:
                error = str(exc)
                print(f"Extraction error for {article_id}, attempt {attempt + 1}/3: {exc}")
    except Exception as exc:
        error = str(exc)
    d['citations'] = 'extraction failed'
    d['extraction_error'] = error
    return d, 1


def run(data, output_path, id_to_lang_map, judge_model="google/gemini-3-flash-preview"):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    failed = 0
    with open(output_path, 'a', encoding='utf-8') as handle:
        for item in data:
            record, error = extract_report(item, id_to_lang_map, judge_model)
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
    queries = validate_questions([dict(zip(query_data.keys(), row)) for row in zip(*query_data.values())], require_question=False)
    id_to_lang_map = {str(item['id']): item['language'] for item in queries}
    question_map = {item['id']: item for item in queries}
    if not id_to_lang_map:
        raise ValueError('No valid language information found in query data')
    raw_data = []
    for key, value in load_local_dataset(args.raw_data_path).items():
        raw_data.append(dict(value, id=str(key)))
    present = {item['id'] for item in raw_data}
    missing = set(id_to_lang_map) - present
    print(f'FACT input coverage: {len(present & set(id_to_lang_map))}/{len(id_to_lang_map)} reports; {len(missing)} missing.')
    raw_data.extend({'id': identifier, 'response': None} for identifier in sorted(missing))
    for item in raw_data:
        if item['id'] in question_map:
            item['question_fingerprint'] = question_fingerprint(question_map[item['id']])
        try:
            item['report_fingerprint'] = report_fingerprint(item['id'], item)
        except ValueError:
            item.pop('report_fingerprint', None)
    if 'openai' in args.raw_data_path:
        for item in raw_data:
            if isinstance(item.get('response'), str):
                item['response'] = clean_urls(item['response'])
    existing = records_by_id(load_jsonl(args.output_path)) if os.path.exists(args.output_path) else {}
    successful = {}
    for item in raw_data:
        previous = existing.get(item['id'])
        if (previous and previous.get('response') == item.get('response')
                and previous.get('question_fingerprint') == item.get('question_fingerprint')
                and previous.get('report_fingerprint') == item.get('report_fingerprint')
                and all(previous.get(key) == item.get(key) for key in
                        ('provider_citations', 'provider_raw_citations', 'citation_mapping_error'))
                and previous.get('extraction_model') == resolve_openrouter_model(args.judge)):
            try:
                validate_citations(previous.get('citations'))
                successful[item['id']] = previous
            except ValueError:
                pass
    write_jsonl(args.output_path, successful.values())
    pending = [item for item in raw_data if item['id'] not in successful]
    print(f'Processing {len(pending)} reports...')
    worker = partial(extract_report, id_to_lang_map=id_to_lang_map, judge_model=args.judge)
    failed = 0
    # Workers return data; only the parent writes the shared JSONL checkpoint.
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
        print(f'Citation extraction incomplete: {failed} report(s) failed; rerun to retry.')
    sys.exit(1 if failed else 0)
