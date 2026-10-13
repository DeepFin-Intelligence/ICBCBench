"""Exercise every public objective row through the real SDK without live calls."""
import contextlib
import csv
import importlib
import io
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
import openai
import pandas as pd
import pymupdf
from unittest.mock import Mock, patch

os.environ['PYTHON_DOTENV_DISABLED'] = '1'

from openai import OpenAI
from evaluation_toolkit.objective_eval import predict, judge, metrics
from evaluation_toolkit.subjective_eval import write
from evaluation_toolkit.subjective_eval.ExpertCriteria import score_by_criteria as expert, stat as expert_stat
from evaluation_toolkit.subjective_eval.scoring.compute_final_score import compute_final_score

ROOT = Path(__file__).resolve().parents[1]


class PublicPipelineTests(unittest.TestCase):
    def test_calibration_is_independent_of_worker_completion_order(self):
        entries = [(str(index), {'judge_result': {'correct': 'yes' if index < 10 else 'no', 'confidence': 50}})
                   for index in range(20)]
        shuffled = entries[::2] + entries[1::2]
        self.assertEqual(metrics.compute_metrics(dict(entries), 20), metrics.compute_metrics(dict(shuffled), 20))

    def test_final_aggregation_rejects_incomplete_fact_and_expert_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            expert_path, fact_path = base / 'expert.csv', base / 'fact.csv'
            for stage in ('expert', 'fact'):
                for coverage in (97.5, float('nan'), 101, 100):
                    expert_frame = pd.DataFrame({'Model': ['test'], 'Avg Score': [80],
                                                 'Total Tasks': [40], 'Evaluated Tasks': [40], 'Coverage (%)': [100]})
                    fact_frame = pd.DataFrame({'Model Name': ['test'], 'Avg Citation Score (Macro)': [100],
                                               'Avg Source Quality Score (S_source)': [100],
                                               'Total Tasks': [40], 'Evaluated Tasks': [40], 'Coverage (%)': [100]})
                    for frame in (expert_frame, fact_frame):
                        frame['Task Set Fingerprint'] = 'a' * 64
                        frame['Report Set Fingerprint'] = 'b' * 64
                        frame['Provenance Verified'] = True
                    target = expert_frame if stage == 'expert' else fact_frame
                    target['Coverage (%)'] = coverage
                    target['Evaluated Tasks'] = 39
                    expert_frame.to_csv(expert_path, index=False)
                    fact_frame.to_csv(fact_path, index=False)
                    with self.subTest(stage=stage, coverage=coverage), self.assertRaisesRegex(ValueError, 'incomplete'):
                        compute_final_score([str(expert_path)], str(fact_path), str(base / 'final.csv'), .8, .1, .1)
                    self.assertFalse((base / 'final.csv').exists())

    def test_all_40_subjective_questions_preserve_39_scores_and_report_rubric_conflict(self):
        http = importlib.import_module('httpx2' if int(openai.__version__.split('.')[0]) >= 3 else 'httpx')
        questions = json.loads((ROOT / 'data/subjective_questions_public_40.json').read_text())
        calls = []

        def handle(request):
            payload = json.loads(request.content)
            calls.append(payload)
            if payload['model'] == 'openai/gpt-4o':
                content = 'A synthetic research report [1](https://www.sec.gov/test).'
            else:
                prompt = payload['messages'][0]['content']
                question = next(item for item in questions if item['expert_evaluation_criteria'] in prompt)
                caps = expert.parse_criteria_dimensions(question['expert_evaluation_criteria'])
                content = json.dumps([{'dimension_id': key, 'score': round(cap * .8, 2)} for key, cap in caps.items()])
            return http.Response(200, json={
                'id': 'chatcmpl-offline', 'object': 'chat.completion', 'created': 1,
                'model': payload['model'], 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': content}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            base = Path(directory)
            output = io.StringIO()
            stack.enter_context(contextlib.redirect_stdout(output))
            stack.enter_context(contextlib.redirect_stderr(output))
            stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Live network forbidden')))
            sdk = OpenAI(api_key='offline-test', base_url='https://offline.invalid/v1',
                         http_client=http.Client(transport=http.MockTransport(handle)))
            stack.callback(sdk.close)
            dataset = str(ROOT / 'data/subjective_questions_public_40.json')
            options = SimpleNamespace(local_dataset=dataset, dataset=None, model='gpt-4o', num_workers=4,
                                      max_samples=None, temperature=0, max_completion_tokens=128,
                                      output_dir=str(base / 'reports'))
            write.args = options
            with patch.object(write, 'client', sdk):
                self.assertEqual(write.main(options), 0)
                self.assertEqual(write.main(options), 0)
            self.assertEqual(len(calls), 40)
            reports = json.loads((base / 'reports/reports_gpt-4o.json').read_text())
            self.assertEqual(set(reports), {question['id'] for question in questions})
            options = SimpleNamespace(question_path=dataset, judge='gpt-5.4', report_model='gpt-4o',
                                      report_dir=str(base / 'reports'), output_dir=str(base / 'scores'),
                                      max_workers=4, max_samples=None)
            with patch.object(expert, 'client', sdk):
                self.assertEqual(expert.main(options), 1)
                self.assertEqual(expert.main(options), 1)
            self.assertEqual(len(calls), 79)
            scores_dir = base / 'scores/judge_gpt-5.4'
            scores = [json.loads(line) for line in (scores_dir / 'scores_gpt-4o.jsonl').read_text().splitlines()]
            self.assertEqual(len(scores), 39)
            self.assertNotIn('36', {score['task_id'] for score in scores})
            self.assertIn('36.0 != 30.0', output.getvalue())
            expert_stat.stat(str(scores_dir), dataset, str(base / 'expert.csv'))
            with (base / 'expert.csv').open(encoding='utf-8-sig') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(int(row['Total Tasks']), 40)
            self.assertEqual(int(row['Evaluated Tasks']), 39)
            self.assertEqual(float(row['Coverage (%)']), 97.5)

    def test_all_80_objective_questions_real_sdk_prediction_judging_and_metrics(self):
        # Select the HTTP runtime actually used by this installed OpenAI version.
        http = importlib.import_module('httpx2' if int(openai.__version__.split('.')[0]) >= 3 else 'httpx')
        calls = []
        with pymupdf.open() as document:
            for index in range(8):
                page = document.new_page(width=64, height=64)
                page.insert_text((8, 25), f'Page {index + 1}', fontsize=8)
            pdf_response = Mock(content=document.tobytes())

        def handle(request):
            payload = json.loads(request.content)
            calls.append(payload)
            content = ('answer' if payload['model'] == 'openai/gpt-4o' else json.dumps({
                'extracted_final_answer': 'answer', 'reasoning': 'synthetic transport response',
                'correct': 'yes', 'confidence': 100}))
            return http.Response(200, json={
                'id': 'chatcmpl-offline', 'object': 'chat.completion', 'created': 1,
                'model': payload['model'], 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': content}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            base = Path(directory)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Live network forbidden')))
            http_client = http.Client(transport=http.MockTransport(handle))
            sdk = OpenAI(api_key='offline-test', base_url='https://offline.invalid/v1', http_client=http_client)
            stack.callback(sdk.close)
            dataset = str(ROOT / 'data/objective_questions_public_80.json')
            options = SimpleNamespace(local_dataset=dataset, dataset=None, model='gpt-4o', num_workers=4,
                                      max_samples=None, temperature=0, max_completion_tokens=128,
                                      output_dir=str(base / 'predictions'))
            predict.args = options
            with patch.object(predict, 'client', sdk), patch.object(predict.requests, 'get', return_value=pdf_response) as download:
                self.assertEqual(predict.main(options), 0)
                self.assertEqual(predict.main(options), 0)
            self.assertEqual(download.call_count, 2)
            self.assertEqual(len(calls), 80)
            questions = json.loads(Path(dataset).read_text())
            by_question = {question['question']: question for question in questions}
            seen = set()
            for call in calls:
                user = call['messages'][-1]['content']
                question_text = user[0]['text'] if isinstance(user, list) else user
                matching = [text for text in by_question if question_text.startswith(text)]
                self.assertEqual(len(matching), 1)
                original_text = matching[0]
                self.assertNotIn(original_text, seen)
                seen.add(original_text)
                if isinstance(user, list):
                    self.assertEqual(by_question[original_text]['id'], '25')
                    for attachment in by_question[original_text]['attachment']:
                        self.assertIn(attachment['source_url'], question_text)
                        self.assertIn(str(attachment['pdf_page']), question_text)
                        self.assertIn(attachment['description'], question_text)
                else:
                    self.assertEqual(question_text, original_text)
            self.assertEqual(seen, set(by_question))
            image_blocks = [block for call in calls for message in call['messages']
                            if isinstance(message['content'], list) for block in message['content']
                            if block['type'] == 'image_url']
            self.assertEqual(len(image_blocks), 2)
            self.assertTrue(all(block['image_url']['url'].startswith('data:image/') for block in image_blocks))

            judge_options = SimpleNamespace(local_dataset=dataset, predictions=str(base / 'predictions/gpt-4o.json'),
                                            judge='gpt-5.4', num_workers=4, output_dir=str(base / 'judged'),
                                            metrics_csv=str(base / 'judge-metrics.csv'))
            judge.args = judge_options
            with patch.object(judge, 'client', sdk):
                self.assertEqual(judge.main(judge_options), 0)
                self.assertEqual(judge.main(judge_options), 0)
            self.assertEqual(len(calls), 160)
            for question in questions:
                matching = [call for call in calls[80:] if question['question'] in call['messages'][0]['content']]
                self.assertEqual(len(matching), 1, question['id'])
                self.assertIn(str(question['answer']), matching[0]['messages'][0]['content'])
            metrics_options = SimpleNamespace(judge='gpt-5.4', query_file=dataset,
                                              judged_dir=str(base / 'judged'),
                                              output_dir=str(base / 'metrics'))
            self.assertEqual(metrics.main(metrics_options), 0)
            expected_totals = {'evaluation_results.csv': 80, 'en_evaluation_results.csv': 40,
                               'zh_evaluation_results.csv': 40, 'en_text_only_evaluation_results.csv': 36,
                               'zh_text_only_evaluation_results.csv': 34}
            for filename, expected_total in expected_totals.items():
                with self.subTest(filename=filename), (base / 'metrics' / filename).open() as stream:
                    row = next(csv.DictReader(stream))
                    self.assertEqual(int(row['Total']), expected_total)
                    self.assertEqual(int(row['Evaluated']), expected_total)
                    self.assertEqual(float(row['Coverage (%)']), 100)
                    self.assertEqual(float(row['Accuracy (%)']), 100)
                    self.assertEqual(float(row['CalibrationError (%)']), 0)


if __name__ == '__main__':
    unittest.main()
