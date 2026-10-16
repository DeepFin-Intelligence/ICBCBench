"""Regression tests for public data and CLI pipelines; no live services required."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pymupdf

os.environ['PYTHON_DOTENV_DISABLED'] = '1'

from evaluation_toolkit import model_clients
from evaluation_toolkit import DR_clients
from evaluation_toolkit.objective_eval import predict, judge
from evaluation_toolkit.subjective_eval import write
from evaluation_toolkit.subjective_eval.ExpertCriteria import score_by_criteria, stat as expert_stat
from evaluation_toolkit.subjective_eval.FACT import extract, deduplicate, validate, stat as fact_stat
from evaluation_toolkit.subjective_eval.scoring.compute_final_score import run_all_versions

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE = json.loads((ROOT/'data/objective_questions_public_80.json').read_text())
SUBJECTIVE = json.loads((ROOT/'data/subjective_questions_public_40.json').read_text())


def fake_client(content=None, side_effect=None):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(model_dump=lambda: {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2})
    )
    create = Mock(return_value=response, side_effect=side_effect)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def expert_scores(question):
    criteria = question.get('expert_evaluation_criteria_en') or question['expert_evaluation_criteria']
    limits = score_by_criteria.parse_criteria_dimensions(criteria) or {'1.1': 100}
    return json.dumps([{'dimension_id': dimension, 'score': round(cap * .8, 2)}
                       for dimension, cap in limits.items()])


def fake_expert_client(questions):
    def respond(**kwargs):
        prompt = kwargs['messages'][0]['content']
        for question in questions:
            criteria = question.get('expert_evaluation_criteria_en') or question['expert_evaluation_criteria']
            if criteria in prompt:
                return fake_client(expert_scores(question)).chat.completions.create.return_value
        raise AssertionError('Unknown scoring criteria')
    return fake_client(side_effect=respond)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.stdout = io.StringIO()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(self.stdout))
        self.stack.enter_context(contextlib.redirect_stderr(self.stdout))
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Live network forbidden in tests')))
        self.stack.enter_context(patch.object(model_clients, 'load_dotenv'))

    def save(self, name, value):
        path = self.base/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False))
        return str(path)

    def options(self, dataset, out, **kwargs):
        return SimpleNamespace(local_dataset=dataset, dataset=None, model='gpt-4o',
                               num_workers=1, max_samples=None, temperature=0,
                               max_completion_tokens=128, output_dir=str(out), **kwargs)

    def pdf_download(self):
        with pymupdf.open() as document:
            for index in range(8):
                page = document.new_page(width=64, height=64)
                page.insert_text((8, 25), f'Page {index + 1}', fontsize=8)
            response = Mock(content=document.tobytes())
        return patch.object(predict.requests, 'get', return_value=response)

    def cli(self, script, arguments):
        with patch.object(sys, 'argv', [script, *arguments]):
            try:
                runpy.run_path(str(ROOT/script), run_name='__main__')
            except SystemExit as e:
                return e.code or 0
        return 0

    def test_all_documented_entrypoints_help_without_credentials(self):
        env = {k:v for k,v in os.environ.items() if 'KEY' not in k and 'TOKEN' not in k}
        env['PYTHON_DOTENV_DISABLED']='1'
        scripts = ['objective_eval/predict.py','objective_eval/judge.py','objective_eval/metrics.py','subjective_eval/write.py',
                   'subjective_eval/ExpertCriteria/score_by_criteria.py','subjective_eval/ExpertCriteria/stat.py',
                   'subjective_eval/FACT/extract.py','subjective_eval/FACT/deduplicate.py',
                   'subjective_eval/FACT/scrape.py','subjective_eval/FACT/validate.py',
                   'subjective_eval/FACT/stat.py','subjective_eval/FACT/time_decay.py',
                   'subjective_eval/scoring/compute_final_score.py']
        for script in scripts:
            with self.subTest(script=script):
                p = subprocess.run([sys.executable, str(ROOT/'evaluation_toolkit'/script), '--help'],
                                   cwd=self.base, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
                self.assertIn('usage:', p.stdout)

    def test_openrouter_aliases_and_custom_ids(self):
        cases = {'gpt-4o':'openai/gpt-4o', 'gpt-5.4':'openai/gpt-5.4',
                 'gemini-3.1-pro':'google/gemini-3.1-pro-preview',
                 'gemini-3-flash-preview':'google/gemini-3-flash-preview',
                 'perplexity-deep-research':'perplexity/sonar-deep-research',
                 'custom/model':'custom/model', 'custom-endpoint':'custom-endpoint'}
        for name, expected in cases.items():
            self.assertEqual(model_clients.resolve_openrouter_model(name), expected)

    def test_missing_key_is_detected_at_request_time(self):
        model_clients.get_openrouter_client.cache_clear()
        self.addCleanup(model_clients.get_openrouter_client.cache_clear)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'OPENROUTER_API_KEY'):
                _ = model_clients.OPENROUTER.chat

    def test_only_selected_dr_provider_is_initialized(self):
        names=['GeminiDeepResearch','TongyiDeepResearch','QwenDeepResearch','TavilyDeepResearch','OpenaiDeepResearch']
        with contextlib.ExitStack() as stack:
            mocks={name:stack.enter_context(patch.object(DR_clients,name)) for name in names}
            self.assertIsNone(DR_clients.create_dr_client('gpt-4o'))
            DR_clients.create_dr_client('tavily-research')
            mocks['TavilyDeepResearch'].assert_called_once_with()
            for name in names:
                if name!='TavilyDeepResearch': mocks[name].assert_not_called()

    def test_all_public_english_topics_use_available_fields(self):
        write.args=SimpleNamespace(model='gpt-4o')
        for q in SUBJECTIVE:
            messages=write.format_message(q)
            self.assertIn(q['question'],messages[1]['content'])
        for q in SUBJECTIVE:
            with self.subTest(task_id=q['id']), patch.object(score_by_criteria,'call_llm') as call:
                call.return_value = expert_scores(q)
                result, _=score_by_criteria.process_single_report(q['id'],{'response':'report','model':'gpt-4o'},
                            SimpleNamespace(report_model='gpt-4o',judge='gpt-5.4'),{q['id']:q})
                self.assertAlmostEqual(result['total_score'],80)

    def test_existing_bilingual_schema_is_supported(self):
        q={'id':'bilingual','language':'en','question':'中文','question_en':'English',
           'expert_evaluation_criteria':'中文标准','expert_evaluation_criteria_en':'English criteria'}
        write.args=SimpleNamespace(model='gpt-4o')
        self.assertIn('English',write.format_message(q)[1]['content'])
        with patch.object(score_by_criteria,'call_llm',return_value=expert_scores(q)) as call:
            result,_=score_by_criteria.process_single_report(q['id'],{'response':'report','model':'gpt-4o'},
                        SimpleNamespace(report_model='gpt-4o',judge='gpt-5.4'),{q['id']:q})
            self.assertEqual(result['total_score'],80)
            self.assertIn('English criteria',call.call_args.args[0][0]['content'])

    def test_public_attachments_are_present_and_encoded(self):
        with self.pdf_download() as download:
            for q in OBJECTIVE:
                blocks,_=predict.build_attachment_content(q.get('attachment') or [])
                self.assertEqual(len(blocks),len(q.get('attachment') or []))
            q=next(q for q in OBJECTIVE if q['id']=='25')
            blocks,annotations=predict.build_attachment_content(q['attachment'])
        self.assertEqual(len(blocks),2)
        self.assertTrue(all(x['image_url']['url'].startswith('data:image/') for x in blocks))
        self.assertTrue(all('file_path' not in item for item in q['attachment']))
        for item in q['attachment']:
            self.assertIn(item['source_url'], annotations)
            self.assertIn(str(item['pdf_page']), annotations)
        self.assertEqual(download.call_count,4)

    def test_missing_attachment_fails_instead_of_omitting_image(self):
        with self.assertRaises(FileNotFoundError):
            predict.build_attachment_content([{'file_path':'attachments/absent.png','mime_type':'image/png'}])

    def test_objective_predictions_resume_and_route_without_dr_keys(self):
        data=self.save('questions.json',[OBJECTIVE[0],OBJECTIVE[40]])
        args=self.options(data,self.base/'predictions')
        predict.args=args
        client=fake_client('answer')
        with patch.object(predict,'client',client):
            self.assertEqual(predict.main(args),0)
            self.assertEqual(predict.main(args),0)
        self.assertEqual(client.chat.completions.create.call_count,2)
        self.assertEqual(client.chat.completions.create.call_args.kwargs['model'],'openai/gpt-4o')
        self.assertEqual(len(json.loads((self.base/'predictions/gpt-4o.json').read_text())),2)

    def test_objective_partial_failure_preserves_success_and_retries(self):
        data=self.save('questions.json',[OBJECTIVE[0],OBJECTIVE[40]])
        args=self.options(data,self.base/'predictions'); predict.args=args
        good=fake_client('answer').chat.completions.create.return_value
        client=fake_client(side_effect=[good,RuntimeError('service unavailable')])
        with patch.object(predict,'client',client): self.assertEqual(predict.main(args),1)
        path=self.base/'predictions/gpt-4o.json'
        self.assertEqual(len(json.loads(path.read_text())),1)
        client=fake_client('recovered')
        with patch.object(predict,'client',client): self.assertEqual(predict.main(args),0)
        client.chat.completions.create.assert_called_once()
        self.assertEqual(len(json.loads(path.read_text())),2)

    def test_empty_prediction_is_a_failure(self):
        predict.args=self.options(None,self.base)
        with patch.object(predict,'client',fake_client('')):
            with self.assertRaisesRegex(ValueError,'empty content'):
                predict.attempt_question(OBJECTIVE[0])

    def test_image_rejection_does_not_produce_a_score_without_the_image(self):
        predict.args=self.options(None,self.base)
        q=next(q for q in OBJECTIVE if q['id']=='25')
        client=fake_client(side_effect=RuntimeError('no image input supported'))
        with patch.object(predict,'client',client), self.pdf_download():
            with self.assertRaisesRegex(RuntimeError,'image'):
                predict.attempt_question(q)
        client.chat.completions.create.assert_called_once()

    def test_subjective_reports_resume_after_partial_failure(self):
        data=self.save('topics.json',[SUBJECTIVE[0],SUBJECTIVE[20]])
        args=self.options(data,self.base/'reports'); write.args=args
        good=fake_client('report').chat.completions.create.return_value
        client=fake_client(side_effect=[good,RuntimeError('service unavailable')])
        with patch.object(write,'client',client): self.assertEqual(write.main(args),1)
        client=fake_client('report')
        with patch.object(write,'client',client):
            self.assertEqual(write.main(args),0)
            self.assertEqual(write.main(args),0)
        client.chat.completions.create.assert_called_once()
        self.assertEqual(client.chat.completions.create.call_args.kwargs['model'],'openai/gpt-4o')

    def test_real_cli_without_key_returns_failure_not_success(self):
        env={k:v for k,v in os.environ.items() if 'KEY' not in k and 'TOKEN' not in k}
        env['PYTHON_DOTENV_DISABLED']='1'
        for script,dataset in [('objective_eval/predict.py','objective_questions_public_80.json'),
                               ('subjective_eval/write.py','subjective_questions_public_40.json')]:
            p=subprocess.run([sys.executable,str(ROOT/'evaluation_toolkit'/script),'--local_dataset',
                              str(ROOT/'data'/dataset),'--model','gpt-4o','--num_workers','1','--max_samples','1',
                              '--output_dir',str(self.base/script.split('/')[0])],
                              env=env,cwd=self.base,text=True,capture_output=True,timeout=30)
            self.assertEqual(p.returncode,1,p.stdout+p.stderr)
            self.assertIn('OPENROUTER_API_KEY',p.stdout+p.stderr)
            self.assertNotIn('GEMINI_API_KEY',p.stdout+p.stderr)

    def test_objective_judge_with_single_worker(self):
        dataset=self.save('questions.json',[OBJECTIVE[0],OBJECTIVE[40]])
        predictions=self.save('predictions.json',{q['id']:{'model':'gpt-4o','response':'answer'} for q in [OBJECTIVE[0],OBJECTIVE[40]]})
        args=SimpleNamespace(local_dataset=dataset,predictions=predictions,judge='gpt-5.4',num_workers=1,
                             output_dir=str(self.base/'judged'),metrics_csv=str(self.base/'metrics.csv'))
        judge.args=args
        client=fake_client(json.dumps({'extracted_final_answer':'answer','reasoning':'mock','correct':'yes','confidence':100}))
        with patch.object(judge,'client',client): self.assertEqual(judge.main(args),0)
        self.assertEqual(client.chat.completions.create.call_args.kwargs['model'],'openai/gpt-5.4')
        self.assertEqual(float(next(csv.DictReader((self.base/'metrics.csv').open()))['Accuracy (%)']),100)

    def test_expert_cli_failure_is_nonzero(self):
        dataset=self.save('topics.json',[SUBJECTIVE[0]])
        self.save('reports/reports_gpt-4o.json',{SUBJECTIVE[0]['id']:{'model':'gpt-4o','response':'report'}})
        with patch.object(model_clients,'get_openrouter_client',return_value=fake_client(side_effect=RuntimeError('offline'))):
            code=self.cli('evaluation_toolkit/subjective_eval/ExpertCriteria/score_by_criteria.py',
                          ['--judge','gpt-5.4','--report_model','gpt-4o','--question_path',dataset,
                           '--report_dir',str(self.base/'reports'),'--output_dir',str(self.base/'scores'),'--max_workers','1'])
        self.assertEqual(code,1)

    def test_zero_score_included_and_unscored_entries_excluded(self):
        path=self.base/'scores.jsonl'
        path.write_text('\n'.join(json.dumps(x) for x in [
            {'task_id':'1','model':'A','total_score':80}, {'task_id':'2','model':'A','total_score':0},
            {'task_id':'3','model':'B','total_score':60}])+'\n')
        result=expert_stat.merge_multiple_jsonl_to_csv([str(path)],str(self.base/'leaderboard.csv'),None,{})
        scores=dict(zip(result['Model'],result['Avg Score']))
        self.assertEqual(scores,{'A':40,'B':60})

    def test_full_subjective_pipeline_bilingual_with_mock_services(self):
        questions=[SUBJECTIVE[0],SUBJECTIVE[20]]
        dataset=self.save('topics.json',questions)
        report_dir=self.base/'reports'; scores_dir=self.base/'scores'
        with patch.object(model_clients,'get_openrouter_client',return_value=fake_client('A cited report [1](https://www.sec.gov/test).')):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/write.py',[
                '--local_dataset',dataset,'--model','gpt-4o','--num_workers','1','--output_dir',str(report_dir)]),0)
        reports=json.loads((report_dir/'reports_gpt-4o.json').read_text())
        self.assertEqual(set(reports),{q['id'] for q in questions})
        with patch.object(model_clients,'get_openrouter_client',return_value=fake_expert_client(questions)):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/ExpertCriteria/score_by_criteria.py',[
                '--judge','gpt-5.4','--report_model','gpt-4o','--question_path',dataset,'--report_dir',str(report_dir),
                '--output_dir',str(scores_dir),'--max_workers','1']),0)
        expert_dir=scores_dir/'judge_gpt-5.4'
        for lang in [None,'zh','en']:
            expert_stat.stat(str(expert_dir),dataset,str(expert_dir/'leaderboard.csv'),lang)
        common=['--query_data_path',dataset,'--n_total_process','1']
        citations=json.dumps([{'fact':'fact','ref_idx':1,'url':'https://www.sec.gov/test'},
                             {'fact':'fact','ref_idx':1,'url':'https://www.sec.gov/test'}])
        extracted=self.base/'extracted.jsonl'; deduped=self.base/'deduped.jsonl'
        scraped=self.base/'scraped.jsonl'; validated=self.base/'validated.jsonl'
        with patch.object(model_clients,'get_openrouter_client',return_value=fake_client(citations)):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/FACT/extract.py',[
                '--raw_data_path',str(report_dir/'reports_gpt-4o.json'),'--output_path',str(extracted),*common]),0)
        with patch.object(model_clients,'get_openrouter_client',return_value=fake_client('[1]')):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/FACT/deduplicate.py',[
                '--raw_data_path',str(extracted),'--output_path',str(deduped),*common]),0)
        with patch('evaluation_toolkit.subjective_eval.FACT.utils.scrape_url',return_value={
                'url':'https://www.sec.gov/test','content':'fact','publish_time':'2099-01-01'}):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/FACT/scrape.py',[
                '--raw_data_path',str(deduped),'--output_path',str(scraped),'--n_total_process','1']),0)
        with patch.object(model_clients,'get_openrouter_client',return_value=fake_client('[{"idx":1,"result":"supported"}]')):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/FACT/validate.py',[
                '--raw_data_path',str(scraped),'--output_path',str(validated),*common]),0)
        with patch.object(model_clients,'get_openrouter_client',side_effect=AssertionError('Resume must not call a service')):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/FACT/validate.py',[
                '--raw_data_path',str(scraped),'--output_path',str(validated),*common]),0)
        self.assertEqual(len(validated.read_text().splitlines()),2)
        args=SimpleNamespace(input_path=str(validated),model_name='gpt-4o',language_filter=None)
        for lang in [None,'zh','en']:
            suffix='_'+lang if lang else ''
            fact_stat.run_stat(args,{q['id']:q['language'] for q in questions},str(self.base/f'fact{suffix}.csv'),lang,query_records=questions)
        run_all_versions([str(expert_dir)],str(self.base/'fact.csv'),str(self.base/'final'),.8,.1,.1)
        for lang in ['', '_zh','_en']:
            with (self.base/'final'/f'final_leaderboard{lang}.csv').open(encoding='utf-8-sig') as f:
                row=next(csv.DictReader(f))
            self.assertAlmostEqual(float(row['S_final']),84)

    def test_all_40_public_FACT_tasks_keep_original_report_provenance(self):
        from evaluation_toolkit.subjective_eval.scoring.provenance import report_fingerprint, report_set_fingerprint, task_set_fingerprint
        questions = SUBJECTIVE
        query = self.save('all40-questions.json', questions)
        reports = {q['id']: {'model': 'gpt-4o', 'response': 'A fact [1](https://www.sec.gov/test#:~:text=original).'} for q in questions}
        report_file = self.save('reports_openai.json', reports)
        common = ['--query_data_path', query, '--n_total_process', '1']
        extracted, deduped, scraped, validated = (self.base / name for name in ('all40-extracted.jsonl', 'all40-deduped.jsonl', 'all40-scraped.jsonl', 'all40-validated.jsonl'))
        citations = json.dumps([{'fact': 'fact', 'ref_idx': 1, 'url': 'https://www.sec.gov/test'}] * 2)
        with patch.object(model_clients, 'get_openrouter_client', return_value=fake_client(citations)):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/FACT/extract.py', ['--raw_data_path', report_file, '--output_path', str(extracted), *common]), 0)
        with patch.object(model_clients, 'get_openrouter_client', side_effect=AssertionError('Resume must not call API')):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/FACT/extract.py', ['--raw_data_path', report_file, '--output_path', str(extracted), *common]), 0)
        with patch.object(model_clients, 'get_openrouter_client', return_value=fake_client('[1]')):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/FACT/deduplicate.py', ['--raw_data_path', str(extracted), '--output_path', str(deduped), *common]), 0)
        with patch('evaluation_toolkit.subjective_eval.FACT.utils.scrape_url', return_value={'url': 'https://www.sec.gov/test', 'content': 'fact', 'publish_time': '2099-01-01'}):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/FACT/scrape.py', ['--raw_data_path', str(deduped), '--output_path', str(scraped), '--n_total_process', '1']), 0)
        with patch.object(model_clients, 'get_openrouter_client', return_value=fake_client('[{"idx":1,"result":"supported"}]')):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/FACT/validate.py', ['--raw_data_path', str(scraped), '--output_path', str(validated), *common]), 0)
        records = [json.loads(line) for line in validated.read_text().splitlines()]
        self.assertEqual(len(records), 40)
        self.assertTrue(all('#:~:text=' not in record['response'] for record in records))
        self.assertEqual({record['id']: record['report_fingerprint'] for record in records}, {tid: report_fingerprint(tid, report) for tid, report in reports.items()})
        args = SimpleNamespace(input_path=str(validated), model_name='gpt-4o', language_filter=None)
        for lang in (None, 'en', 'zh'):
            output = self.base / ('all40-fact' + (lang or '') + '.csv')
            self.assertEqual(fact_stat.run_stat(args, {q['id']: q['language'] for q in questions}, str(output), lang, query_records=questions), 0)
            with output.open() as handle:
                row = next(csv.DictReader(handle))
            selected = [q for q in questions if lang is None or q['language'] == lang]
            self.assertEqual(int(row['Evaluated Tasks']), len(selected))
            self.assertEqual(row['Provenance Verified'], 'True')
            self.assertEqual(row['Task Set Fingerprint'], task_set_fingerprint(selected))
            self.assertEqual(row['Report Set Fingerprint'], report_set_fingerprint({q['id']: report_fingerprint(q['id'], reports[q['id']]) for q in selected}))

    def test_fact_extraction_failure_is_recorded(self):
        path=self.base/'failed.jsonl'
        with patch.object(extract,'client',fake_client(side_effect=RuntimeError('offline'))):
            self.assertEqual(extract.run([{'id':'1','response':'report'}],str(path),{'1':'en'}),1)
        self.assertEqual(json.loads(path.read_text())['citations'],'extraction failed')

    def test_fact_deduplication_failure_is_retryable_and_retains_facts(self):
        path=self.base/'failed-dedup.jsonl'
        data=[{'id':'1','citations':[{'url':'https://x','fact':'one'},{'url':'https://x','fact':'two'}]}]
        with patch.object(deduplicate,'client',fake_client(side_effect=RuntimeError('offline'))):
            self.assertEqual(deduplicate.run(data,str(path),{'1':'en'}),1)
        result=json.loads(path.read_text())
        self.assertIn('dedup_error',result)
        self.assertEqual(result['citations_deduped']['https://x']['facts'],['one','two'])

    def test_canonical_ids_work_across_objective_cli_pipeline(self):
        dataset=self.save('questions.json',[OBJECTIVE[0],OBJECTIVE[40]])
        predictions=self.base/'predictions'; judged=self.base/'judged'; metrics=self.base/'metrics'
        with patch.object(model_clients,'get_openrouter_client',return_value=fake_client('answer')):
            self.assertEqual(self.cli('evaluation_toolkit/objective_eval/predict.py',[
                '--local_dataset',dataset,'--model','openai/gpt-4o','--num_workers','1','--output_dir',str(predictions)]),0)
        payload=json.dumps({'extracted_final_answer':'answer','reasoning':'mock','correct':'yes','confidence':100})
        with patch.object(model_clients,'get_openrouter_client',return_value=fake_client(payload)):
            self.assertEqual(self.cli('evaluation_toolkit/objective_eval/judge.py',[
                '--local_dataset',dataset,'--predictions',str(predictions/'gpt-4o.json'),
                '--judge','openai/gpt-5.4','--num_workers','1','--output_dir',str(judged),
                '--metrics_csv',str(self.base/'judge-metrics.csv')]),0)
        self.assertEqual(self.cli('evaluation_toolkit/objective_eval/metrics.py',[
            '--judge','openai/gpt-5.4','--query_file',dataset,'--judged_dir',str(judged),'--output_dir',str(metrics)]),0)
        with (metrics/'evaluation_results.csv').open() as f:
            self.assertEqual(float(next(csv.DictReader(f))['Accuracy (%)']),100)

    def test_canonical_ids_share_subjective_report_and_score_filenames(self):
        dataset=self.save('topics.json',[SUBJECTIVE[0]])
        reports=self.base/'reports'; scores=self.base/'scores'
        with patch.object(model_clients,'get_openrouter_client',return_value=fake_client('report')):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/write.py',[
                '--local_dataset',dataset,'--model','openai/gpt-4o','--num_workers','1','--output_dir',str(reports)]),0)
        with patch.object(model_clients,'get_openrouter_client',return_value=fake_expert_client([SUBJECTIVE[0]])):
            self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/ExpertCriteria/score_by_criteria.py',[
                '--judge','openai/gpt-5.4','--report_model','openai/gpt-4o','--question_path',dataset,
                '--report_dir',str(reports),'--output_dir',str(scores),'--max_workers','1']),0)
        self.assertTrue((scores/'judge_gpt-5.4/scores_gpt-4o.jsonl').is_file())

    def test_fact_validation_missing_reference_is_recorded(self):
        result=validate.validate(('https://x',{'url_content':None,'facts':['fact'],'article_id':'1'}),{'1':'en'})
        self.assertEqual(result['error'],'no reference')

    def test_time_decay_direct_cli_produces_output(self):
        source=self.base/'source.jsonl'
        source.write_text(json.dumps({'id':'1','citations_deduped':{'https://x':{'publish_time':'2099-01-01'}}})+'\n')
        self.assertEqual(self.cli('evaluation_toolkit/subjective_eval/FACT/time_decay.py',[
            '--input_path',str(source),'--output_path',str(self.base/'timed.jsonl')]),0)
        output=json.loads((self.base/'timed.jsonl').read_text())
        self.assertEqual(output['citations_deduped']['https://x']['time_decay_score'],1)

    def test_dr_plain_text_is_passed_to_each_service(self):
        from evaluation_toolkit.DR_clients.GeminiDR import GeminiDeepResearch
        from evaluation_toolkit.DR_clients.TavilyResearch import TavilyDeepResearch
        from evaluation_toolkit.DR_clients.OpenaiDR import OpenaiDeepResearch
        gemini=GeminiDeepResearch.__new__(GeminiDeepResearch)
        gemini.agent_name='test-agent'
        interactions=SimpleNamespace(create=Mock(return_value=SimpleNamespace(id='1')),
                       get=Mock(return_value=SimpleNamespace(status='completed',steps=[SimpleNamespace(
                           type='model_output',content=[SimpleNamespace(type='text',text='answer')])],output_text='answer')))
        gemini.client=SimpleNamespace(interactions=interactions)
        gemini.run_research('system','',multimodal_content='actual question')
        self.assertIn('actual question',interactions.create.call_args.kwargs['input'])
        tavily=TavilyDeepResearch.__new__(TavilyDeepResearch)
        tavily.client=SimpleNamespace(research=Mock(return_value={'request_id':'1'}),
                         get_research=Mock(return_value={'status':'completed','content':'answer'}))
        with patch('evaluation_toolkit.DR_clients.TavilyResearch.time.sleep'):
            tavily.run_research('system','',multimodal_content='actual question')
        self.assertIn('actual question',tavily.client.research.call_args.args[0])
        openai=OpenaiDeepResearch.__new__(OpenaiDeepResearch); openai.model_name='o3-deep-research'
        openai.client=SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=SimpleNamespace(output_text='answer'))))
        openai.run_research('system','',multimodal_content='actual question')
        self.assertIn('actual question',openai.client.responses.create.call_args.kwargs['input'])

    def test_tavily_sdk_has_research_methods(self):
        from tavily import TavilyClient
        self.assertTrue(callable(TavilyClient.research))
        self.assertTrue(callable(TavilyClient.get_research))


if __name__ == '__main__':
    unittest.main()
