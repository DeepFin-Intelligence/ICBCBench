"""FACT audit regressions: real SDK/HTTP parsing, no live network or services."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openai import OpenAI
from importlib import import_module

# The SDK selects its supported HTTP client (OpenAI 3.x currently uses httpx2).
http_module = OpenAI.__init__.__globals__.get('httpx') or OpenAI.__init__.__globals__['httpx2']
httpx = import_module(http_module.__name__)

os.environ['PYTHON_DOTENV_DISABLED'] = '1'
from evaluation_toolkit import model_clients
from evaluation_toolkit.subjective_eval.FACT import authority, extract, deduplicate, scrape, validate, stat, utils

ROOT = Path(__file__).resolve().parents[1]


def sdk_client(content):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            'id': 'chatcmpl-audit', 'object': 'chat.completion', 'created': 0,
            'model': 'audit', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': content}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})
    client = OpenAI(api_key='local-test', base_url='https://mock.invalid/v1',
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return client, calls


class FactReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.stack.enter_context(patch.object(model_clients, 'load_dotenv'))
        self.stack.enter_context(patch.object(utils.requests, 'get', side_effect=AssertionError('Live HTTP forbidden')))
        self.stack.enter_context(patch.object(validate.time, 'sleep'))

    def cli(self, script, arguments):
        with patch.object(sys, 'argv', [script, *arguments]):
            try:
                runpy.run_path(str(ROOT/'evaluation_toolkit/subjective_eval/FACT'/script), run_name='__main__')
            except SystemExit as exc:
                return exc.code or 0
        return 0

    def save_jsonl(self, name, records):
        path = self.base/name
        utils.write_jsonl(str(path), records)
        return path

    def query(self, count=1):
        path = self.base/'queries.json'
        path.write_text(json.dumps([{'id': str(index), 'language': 'en' if index % 2 else 'zh'} for index in range(1, count + 1)]))
        return path

    def test_no_jina_key_runs_real_html_parser_and_metadata(self):
        html = '<html><title>Example &amp; title</title><meta content="2026-10-07T10:00:00Z" property="article:published_time"><meta content="summary" name="description"><script>secret()</script><body>Supported &amp; verified</body></html>'
        response = SimpleNamespace(headers={'content-type': 'text/html'}, content=html.encode(), text=html,
            url='https://example.test/final', raise_for_status=lambda: None)
        with patch.dict(os.environ, {}, clear=True), patch.object(utils, 'get_redirected_url', return_value='https://example.test'), patch.object(utils.requests, 'get', return_value=response):
            result = utils.scrape_url('https://example.test')
        self.assertEqual(result['publish_time'], '2026-10-07T10:00:00Z')
        self.assertIn('Supported & verified', result['content'])
        self.assertNotIn('secret', result['content'])
        self.assertEqual(result['description'], 'summary')

    def test_jina_failure_falls_back_to_real_local_parser(self):
        html = '<body>Direct content</body>'
        response = SimpleNamespace(headers={'content-type': 'text/html'}, content=html.encode(), text=html,
            url='https://example.test', raise_for_status=lambda: None)
        with patch.dict(os.environ, {'JINA_API_KEY': 'local-test'}, clear=True), patch.object(utils, 'get_redirected_url', return_value='https://example.test'), patch.object(utils.WebScrapingJinaTool, '__call__', return_value={'error': 'offline'}), patch.object(utils.requests, 'get', return_value=response):
            self.assertEqual(utils.scrape_url('https://example.test')['content'], 'Direct content')

    def test_local_pdf_fallback_extracts_actual_pdf_text(self):
        import pymupdf
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((72, 72), 'PDF reference fact')
            pdf = document.tobytes()
        response = SimpleNamespace(headers={'content-type': 'application/pdf'}, content=pdf,
            url='https://example.test/reference.pdf', raise_for_status=lambda: None)
        with patch.object(utils.requests, 'get', return_value=response):
            result = utils._local_scrape(response.url)
        self.assertIn('PDF reference fact', result['content'])
        self.assertNotIn('%PDF-', result['content'])

    def test_extraction_rejects_missing_fields_with_real_sdk(self):
        client, calls = sdk_client('[{"fact":"claim"}]')
        self.addCleanup(client.close)
        with patch.object(extract, 'client', client):
            record, failed = extract.extract_report({'id':'1', 'response':'report'}, {'1':'en'})
        self.assertEqual(failed, 1)
        self.assertEqual(record['citations'], 'extraction failed')
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]['model'], 'google/gemini-3-flash-preview')

    def test_extraction_rejects_partial_json(self):
        client, _ = sdk_client('[{"fact":"claim","ref_idx":1,"url":"https://example.test"}, {')
        self.addCleanup(client.close)
        with patch.object(extract, 'client', client):
            self.assertEqual(extract.extract_report({'id':'1', 'response':'report'}, {'1':'en'})[1], 1)

    def test_validation_sorts_complete_indices_with_real_sdk(self):
        client, _ = sdk_client('[{"idx":2,"result":"unsupported"},{"idx":1,"result":"supported"}]')
        self.addCleanup(client.close)
        with patch.object(validate, 'client', client):
            result = validate.validate(('https://example.test', {'facts':['first', 'second'], 'url_content':'ref', 'article_id':'1'}), {'1':'en'})
        self.assertIsNone(result['error'])
        self.assertEqual(result['validate_res'], [{'idx':0, 'result':'supported'}, {'idx':1, 'result':'unsupported'}])

    def test_validation_rejects_invalid_indices_and_result_enum(self):
        for content in ('[{"idx":999,"result":"supported"},{"idx":999,"result":"supported"}]',
                        '[{"idx":1,"result":"supported"},{"idx":1,"result":"supported"}]',
                        '[{"idx":1,"result":"maybe"},{"idx":2,"result":"supported"}]'):
            with self.subTest(content=content):
                client, calls = sdk_client(content)
                self.addCleanup(client.close)
                with patch.object(validate, 'client', client):
                    result = validate.validate(('https://example.test', {'facts':['a','b'], 'url_content':'ref', 'article_id':'1'}), {'1':'en'})
                self.assertIsNotNone(result['error'])
                self.assertEqual(result['validate_res'], [])
                self.assertEqual(len(calls), 3)

    def test_repaired_scrape_invalidates_old_unknown_validation(self):
        query = self.query()
        raw = self.save_jsonl('scraped.jsonl', [{'id':'1','citations_deduped':{'https://example.test':{'facts':['claim'], 'url_content':'good ref', 'publish_time':'2026-10-07'}}}])
        output = self.save_jsonl('validated.jsonl', [{'id':'1','citations_deduped':{'https://example.test':{'facts':['claim'], 'url_content':'scrape failed: offline','validate_res':[{'idx':0,'result':'unknown'}], 'validate_error':None}}}])
        client, calls = sdk_client('[{"idx":1,"result":"supported"}]')
        self.addCleanup(client.close)
        with patch.object(model_clients, 'get_openrouter_client', return_value=client):
            self.assertEqual(self.cli('validate.py', ['--query_data_path',str(query),'--raw_data_path',str(raw),'--output_path',str(output)]), 0)
        citation = utils.load_jsonl(output)[0]['citations_deduped']['https://example.test']
        self.assertEqual(citation['url_content'], 'good ref')
        self.assertEqual(citation['validate_res'][0]['result'], 'supported')
        self.assertEqual(len(calls), 1)

    def test_scrape_cache_preserves_new_facts_and_drops_stale_judgment(self):
        current = [{'id':'1','citations_deduped':{'https://example.test':{'facts':['new claim'],'url_content':None}}}]
        previous = {'1':{'id':'1','citations_deduped':{'https://example.test':{'facts':['old claim'],'url_content':'ref','validate_res':[{'idx':0,'result':'supported'}],'validate_error':None}}}}
        scrape.merge_scrape_cache(current, previous)
        citation = current[0]['citations_deduped']['https://example.test']
        self.assertEqual(citation['facts'], ['new claim'])
        self.assertEqual(citation['url_content'], 'ref')
        self.assertNotIn('validate_res', citation)

    def test_scrape_failure_exits_nonzero_and_creates_parent(self):
        raw = self.save_jsonl('deduped.jsonl', [{'id':'1','citations_deduped':{'https://example.test':{'facts':['claim'],'url_content':None}}}])
        output = self.base/'nested/scraped.jsonl'
        with patch.object(utils, 'scrape_url', return_value={'error':'offline'}):
            self.assertEqual(self.cli('scrape.py', ['--raw_data_path',str(raw),'--output_path',str(output)]), 1)
        self.assertIn('scrape failed', utils.load_jsonl(output)[0]['citations_deduped']['https://example.test']['url_content'])
        validated = self.base/'nested/validated.jsonl'
        self.assertEqual(self.cli('validate.py', ['--raw_data_path',str(output),'--output_path',str(validated),
            '--query_data_path',str(self.query())]),1)

    def test_jsonl_blanklines_and_atomic_rewrite(self):
        path = self.base/'blank.jsonl'
        path.write_text('\n{"id":"1"}\n\n')
        self.assertEqual(utils.load_jsonl(path), [{'id':'1'}])
        with self.assertRaises(TypeError):
            utils.write_jsonl(path, [{'id':'2'}, {'bad':object()}])
        self.assertEqual(utils.load_jsonl(path), [{'id':'1'}])

    def test_stat_missing_tasks_zero_and_repeated_csv_has_one_model(self):
        record = {'id':'1','citations_deduped':{'https://www.sec.gov':{'facts':['claim'], 'url_content':'ref', 'validate_res':[{'idx':0,'result':'supported'}], 'validate_error':None, 'publish_time':'2099-01-01'}}}
        source = self.save_jsonl('validated.jsonl', [record])
        args = SimpleNamespace(input_path=str(source),model_name='test',language_filter=None)
        csv_path = self.base/'fact.csv'
        for _ in range(2):
            self.assertEqual(stat.run_stat(args, {'1':'en','2':'zh'}, str(csv_path), None), 2)
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]['Avg Citation Score (Macro)']), 50)
        self.assertEqual(int(rows[0]['Total Tasks']), 2)
        self.assertEqual(int(rows[0]['Evaluated Tasks']), 1)
        self.assertEqual(float(rows[0]['Coverage (%)']), 50)

    def test_stat_rejects_invalid_validation_as_zero_and_incomplete(self):
        record = {'id':'1','citations_deduped':{'https://www.sec.gov':{'facts':['a','b'], 'url_content':'ref', 'validate_res':[{'idx':998,'result':'supported'},{'idx':998,'result':'supported'}]}}}
        args = SimpleNamespace(language_filter=None)
        self.assertEqual(stat.fact_evaluate([record],args,{'1':'en'})['Avg Citation Score (Macro)'],0)

    def test_source_missing_date_stays_in_denominator(self):
        records = [{'id':'1','citations_deduped':{
            'https://sec.gov/dated':{'publish_time':'2099-01-01'},
            'https://sec.gov/undated':{'publish_time':'unknown'}}}]
        results = stat.source_evaluate(records,SimpleNamespace(language_filter=None),{'1':'en'})
        self.assertEqual(results['Avg Source Authority Score (S_auth)'],1)
        self.assertEqual(results['Avg Source Timeliness Score (S_time)'],.5)
        self.assertEqual(results['Avg Source Quality Score (S_source)'],50)

    def test_authority_respects_exact_whitelist_subdomains_and_ports(self):
        for url in ('https://www.ecb.europa.eu','https://www.boj.or.jp','https://www.reuters.com:443/test'):
            self.assertEqual(authority.get_authority_weight(url), 1)
        self.assertEqual(authority.get_authority_weight('https://reuters.com.attacker.test'), .2)

    def test_missing_40_reports_exit_nonzero_with_complete_retry_records(self):
        query = self.query(40)
        reports = self.base/'reports.json'
        reports.write_text('{}')
        output = self.base/'extracted.jsonl'
        env = {key:value for key,value in os.environ.items() if 'KEY' not in key and 'TOKEN' not in key}
        env['PYTHON_DOTENV_DISABLED']='1'
        process = subprocess.run([sys.executable,str(ROOT/'evaluation_toolkit/subjective_eval/FACT/extract.py'),
            '--query_data_path',str(query),'--raw_data_path',str(reports),'--output_path',str(output),
            '--n_total_process','2'],env=env,cwd=ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(process.returncode,1,process.stdout+process.stderr)
        records = utils.load_jsonl(output)
        self.assertEqual(len(records),40)
        self.assertTrue(all(record['citations']=='extraction failed' for record in records))
        self.assertIn('0/40 reports; 40 missing', process.stdout)

    def test_changed_judge_invalidates_otherwise_matching_validation(self):
        citation = {'facts':['claim'],'url_content':'ref','validate_res':[{'idx':0,'result':'supported'}],
                    'validate_error':None,'validation_model':'old/model'}
        current = [{'id':'1','citations_deduped':{'https://example.test':dict(citation)}}]
        validate.merge_validation_cache(current,{'1':{'citations_deduped':{'https://example.test':citation}}},'new/model')
        self.assertNotIn('validate_res',current[0]['citations_deduped']['https://example.test'])

    def test_incomplete_stage_schema_is_not_reported_as_success(self):
        raw = self.save_jsonl('malformed.jsonl',[{'id':'1'}])
        self.assertEqual(self.cli('scrape.py',['--raw_data_path',str(raw),
            '--output_path',str(self.base/'scraped.jsonl')]),1)
        self.assertEqual(self.cli('validate.py',['--raw_data_path',str(raw),
            '--output_path',str(self.base/'validated.jsonl'),'--query_data_path',str(self.query())]),1)

    def test_fact_csv_coverage_distinguishes_unknown_from_failed_processing(self):
        base = {'facts':['claim'],'url_content':'reference','validate_res':[{'idx':0,'result':'unknown'}],
                'validate_error':None,'publish_time':'unknown'}
        records = [{'id':'1','citations_deduped':{'https://example.test':dict(base)}},
                   {'id':'2','citations_deduped':{'https://example.test':dict(base,validate_error='offline')}},
                   {'id':'3','citations_deduped':{},'citations':'extraction failed'},
                   {'id':'4','citations_deduped':{}},
                   {'id':'5','citations_deduped':{},'dedup_error':'offline'},
                   {'id':'6','citations_deduped':{'https://example.test':dict(base,url_content='scrape failed: offline')}},
                   {'id':'7','citations_deduped':{'https://example.test':dict(base,validate_res=[{'idx':3,'result':'supported'}])}}]
        source = self.save_jsonl('coverage.jsonl',records)
        args = SimpleNamespace(input_path=str(source),model_name='new',language_filter=None)
        csv_path = self.base/'coverage.csv'
        # Preserve an old row whose historical CSV had no coverage fields.
        csv_path.write_text('Model Name,Total Tasks,Avg Citation Score (Macro),Avg Source Authority Score (S_auth),Avg Source Timeliness Score (S_time),Avg Source Quality Score (S_source)\nold,8,0,0,0,0\n')
        self.assertEqual(stat.run_stat(args,{str(index):'en' for index in range(1,9)},str(csv_path),None),7)
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['Model Name'],'old')
        self.assertEqual(rows[0]['Coverage (%)'],'')
        self.assertEqual(rows[1]['Model Name'],'new')
        self.assertEqual(int(rows[1]['Evaluated Tasks']),2)
        self.assertEqual(float(rows[1]['Coverage (%)']),25)

    def test_real_multiprocessing_dedup_and_validate_preserve_40_records(self):
        query = self.query(40)
        citations = [{'id':str(index),'citations':[{'fact':'claim ' + 'x'*20000,'ref_idx':1,'url':'https://example.test'}]} for index in range(1,41)]
        raw = self.save_jsonl('raw.jsonl', citations)
        deduped = self.base/'deduped.jsonl'
        env = {key:value for key,value in os.environ.items() if 'KEY' not in key and 'TOKEN' not in key}
        env['PYTHON_DOTENV_DISABLED']='1'
        command = [sys.executable,str(ROOT/'evaluation_toolkit/subjective_eval/FACT/deduplicate.py'),
                   '--query_data_path',str(query),'--raw_data_path',str(raw),'--output_path',str(deduped),'--n_total_process','2']
        process = subprocess.run(command,env=env,cwd=ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        records = utils.load_jsonl(deduped)
        self.assertEqual(len(records),40)
        for record in records:
            record['citations_deduped']['https://example.test']['url_content']='scrape failed: offline'
        scraped = self.save_jsonl('scraped.jsonl',records)
        validated = self.base/'validated.jsonl'
        command = [sys.executable,str(ROOT/'evaluation_toolkit/subjective_eval/FACT/validate.py'),
                   '--query_data_path',str(query),'--raw_data_path',str(scraped),'--output_path',str(validated),'--n_total_process','2']
        process = subprocess.run(command,env=env,cwd=ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(process.returncode,1,process.stdout+process.stderr)
        self.assertEqual(len(utils.load_jsonl(validated)),40)
        self.assertEqual(len({record['id'] for record in utils.load_jsonl(validated)}),40)


if __name__ == '__main__':
    unittest.main()
