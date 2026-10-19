"""Objective pipeline checks using the real SDK and a local HTTP transport."""
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pymupdf

from openai import OpenAI, _base_client

from evaluation_toolkit.objective_eval import judge, metrics, predict

# SDK 3.x uses httpx2; SDK 1.x/2.x use httpx. Match the installed SDK's client.
HTTP = getattr(_base_client, 'httpx2', None) or getattr(_base_client, 'httpx')
ROOT = Path(__file__).resolve().parents[1]
PUBLIC = json.loads((ROOT / 'data/objective_questions_public_80.json').read_text())


def judgment(correct='yes', confidence=100):
    return {'extracted_final_answer': 'answer', 'reasoning': 'local transport test',
            'correct': correct, 'confidence': confidence}


def judged_record(correct='yes', confidence=100, question=None, judge_model='judge'):
    record = {'model': 'gpt-4o', 'judge_model': judge_model, 'response': 'answer', 'judge_result': {'correct_answer': 'answer',
            'model_answer': 'answer', 'reasoning': 'local transport test',
            'correct': correct, 'confidence': confidence}}
    if question is not None:
        record['judge_fingerprint'] = metrics.judge_fingerprint(question, record, judge_model)
    return record


class ObjectiveReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='objective-review-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.output = io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.output))
        self.stack.enter_context(contextlib.redirect_stderr(self.output))
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Live network forbidden')))
        self.questions = [dict(PUBLIC[0]), dict(PUBLIC[40])]

    def save(self, filename, value):
        path = self.base / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        return str(path)

    def sdk(self, responses):
        requests, replies = [], iter(responses)
        def handle(request):
            requests.append(json.loads(request.content))
            reply = next(replies)
            if isinstance(reply, int):
                return HTTP.Response(reply, json={'error': {'message': 'local authentication failure', 'type': 'invalid_request_error'}})
            return HTTP.Response(200, json={
                'id': 'chatcmpl-local', 'object': 'chat.completion', 'created': 1, 'model': 'test',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': reply}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            })
        sdk = OpenAI(api_key='local-test', base_url='https://example.invalid/v1', max_retries=0,
                     http_client=HTTP.Client(transport=HTTP.MockTransport(handle)))
        self.addCleanup(sdk.close)
        return sdk, requests

    def prediction_options(self, questions=None, **overrides):
        selected = copy.deepcopy(questions or self.questions)
        for question in selected:
            for attachment in question.get('attachment') or []:
                if attachment.get('source_url'):
                    continue
                path = Path(attachment['file_path'])
                if not path.is_absolute():
                    attachment['file_path'] = str(ROOT / 'data' / path)
        options = dict(local_dataset=self.save('questions.json', selected), dataset=None,
                       model='gpt-4o', num_workers=1, max_samples=None, temperature=0,
                       max_completion_tokens=128, output_dir=str(self.base / 'predictions'))
        options.update(overrides)
        return SimpleNamespace(**options)

    def pdf_download(self):
        with pymupdf.open() as document:
            for index in range(8):
                page = document.new_page(width=64, height=64)
                page.insert_text((8, 25), f'Page {index + 1}', fontsize=8)
            response = Mock(content=document.tobytes())
        return patch.object(predict.requests, 'get', return_value=response)

    def judge_options(self, predictions=None, questions=None):
        return SimpleNamespace(local_dataset=self.save('questions.json', questions or self.questions), dataset=None,
                               predictions=self.save('predictions.json', predictions if predictions is not None else
                                  {q['id']: {'model': 'gpt-4o', 'response': 'answer'} for q in (questions or self.questions)}),
                               judge='gpt-5.4', num_workers=1, output_dir=str(self.base / 'judged'),
                               metrics_csv=str(self.base / 'judge_metrics.csv'))

    def csv_row(self, filename):
        with (self.base / filename).open(newline='') as handle:
            return next(csv.DictReader(handle))

    def test_public_modality_labels_recognize_all_ten_questions(self):
        ids = metrics.get_multimodal_ids_from_json(ROOT / 'data/objective_questions_public_80.json')
        self.assertEqual(set(ids), {q['id'] for q in PUBLIC if q['modality'] == 'Multi-Modal'})
        self.assertEqual(len(ids), 10)

    def test_text_only_subgroups_use_query_counts_and_ignore_unrelated_ids(self):
        query = self.save('public.json', PUBLIC)
        self.save('judged/judged_model_with_underscores.json',
                  {**{q['id']: judged_record(question=q) for q in PUBLIC}, 'unrelated': judged_record()})
        exclusions = metrics.get_multimodal_ids_from_json(query)
        for language in ('en', 'zh'):
            self.assertEqual(metrics.dump_all_metrics('judge', language, exclusions,
                             str(self.base / 'judged'), query, str(self.base / 'metrics')), 0)
            row = self.csv_row(f'metrics/{language}_text_only_evaluation_results.csv')
            expected = sum(q['language'] == language and q['modality'] == 'Text-Only' for q in PUBLIC)
            self.assertEqual(int(row['Total']), expected)
            self.assertEqual(int(row['Evaluated']), expected)
            self.assertEqual(float(row['Accuracy (%)']), 100)
            self.assertEqual(row['Model'], 'gpt-4o')
        self.assertEqual(sum(q['modality'] == 'Text-Only' for q in PUBLIC), 70)

    def test_partial_metrics_use_full_selected_dataset_denominator(self):
        query = self.save('public.json', PUBLIC)
        self.save('judged/judged_gpt-4o.json', {PUBLIC[0]['id']: judged_record(question=PUBLIC[0])})
        status = metrics.dump_all_metrics('judge', None, judged_dir=str(self.base / 'judged'),
                     query_file=query, results_dir=str(self.base / 'metrics'))
        self.assertEqual(status, 1)
        row = self.csv_row('metrics/evaluation_results.csv')
        self.assertEqual((row['Total'], row['Evaluated']), ('80', '1'))
        self.assertEqual(float(row['Accuracy (%)']), 1.25)
        self.assertEqual(float(row['Coverage (%)']), 1.25)

    def test_judge_and_standalone_calibration_are_identical(self):
        results = {str(i): judged_record('yes' if i < 10 else 'no', 0 if i < 10 else 100) for i in range(20)}
        judge.args = SimpleNamespace(predictions='gpt-4o.json', judge='gpt-5.4', metrics_csv=str(self.base / 'judge.csv'))
        judge.dump_metrics(results, 20)
        metrics.dump_metrics('gpt-4o', 'gpt-5.4', results, 20, str(self.base / 'metrics.csv'))
        self.assertEqual(self.csv_row('judge.csv'), self.csv_row('metrics.csv'))
        self.assertEqual(float(self.csv_row('judge.csv')['CalibrationError (%)']), 100)

    def test_real_sdk_serializes_predictions_and_image_blocks(self):
        options = self.prediction_options([next(q for q in PUBLIC if q['id'] == '25')])
        sdk, requests = self.sdk(['answer'])
        with patch.object(predict, 'client', sdk), self.pdf_download():
            self.assertEqual(predict.main(options), 0)
        request = requests[0]
        self.assertEqual(request['model'], 'openai/gpt-4o')
        images = [block for block in request['messages'][1]['content'] if block['type'] == 'image_url']
        self.assertEqual(len(images), 2)
        self.assertTrue(all(block['image_url']['url'].startswith('data:image/png;base64,') for block in images))
        cache = json.loads((self.base / 'predictions/gpt-4o.json').read_text())
        self.assertTrue(cache['25']['request_fingerprint'])
        self.assertEqual(cache['25']['usage']['total_tokens'], 2)

    def test_image_rejection_does_not_retry_without_required_images(self):
        options = self.prediction_options([next(q for q in PUBLIC if q['id'] == '25')])
        sdk, requests = self.sdk([400])
        with patch.object(predict, 'client', sdk), self.pdf_download():
            self.assertEqual(predict.main(options), 1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(json.loads((self.base / 'predictions/gpt-4o.json').read_text()), {})

    def test_prediction_authentication_failure_preserves_success_and_retries_only_failure(self):
        options = self.prediction_options()
        sdk, _ = self.sdk(['answer', 401])
        with patch.object(predict, 'client', sdk):
            self.assertEqual(predict.main(options), 1)
        path = self.base / 'predictions/gpt-4o.json'
        self.assertEqual(set(json.loads(path.read_text())), {self.questions[0]['id']})
        recovered, requests = self.sdk(['recovered'])
        with patch.object(predict, 'client', recovered):
            self.assertEqual(predict.main(options), 0)
            self.assertEqual(predict.main(options), 0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(json.loads(path.read_text())), 2)

    def test_prediction_cache_invalidates_changed_inputs_and_provider(self):
        options = self.prediction_options()
        sdk, requests = self.sdk(['answer', 'answer', 'changed', 'provider one', 'provider two'])
        with patch.object(predict, 'client', sdk):
            self.assertEqual(predict.main(options), 0)
            self.assertEqual(predict.main(options), 0)
            self.questions[0]['question'] += '\nNew instruction.'
            self.save('questions.json', self.questions)
            self.assertEqual(predict.main(options), 0)
            options.model = 'other/gpt-4o'
            self.assertEqual(predict.main(options), 1)
            self.assertEqual(len(requests), 3)
            options.output_dir = str(self.base / 'other-predictions')
            self.assertEqual(predict.main(options), 0)
        self.assertEqual(len(requests), 5)
        self.assertEqual(requests[-1]['model'], 'other/gpt-4o')

    def test_legacy_prediction_cache_without_signature_is_recomputed(self):
        options = self.prediction_options([self.questions[0]])
        self.save('predictions/gpt-4o.json', {self.questions[0]['id']: {'model': 'gpt-4o', 'response': 'old answer'}})
        sdk, requests = self.sdk(['new answer'])
        with patch.object(predict, 'client', sdk):
            self.assertEqual(predict.main(options), 0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(json.loads((self.base / 'predictions/gpt-4o.json').read_text())[self.questions[0]['id']]['response'], 'new answer')

    def test_attachment_changes_invalidate_prediction_signature(self):
        attachment = self.base / 'image.png'
        with pymupdf.open() as document:
            page = document.new_page(width=32, height=32)
            page.draw_rect(page.rect, color=(1, 0, 0), fill=(1, 0, 0))
            attachment.write_bytes(page.get_pixmap().tobytes('png'))
        question = {'id': 'image', 'language': 'en', 'question': 'Read the image',
                    'attachment': [{'file_path': str(attachment), 'mime_type': 'image/png', 'original_filename': 'image.png'}]}
        options = self.prediction_options([question])
        before = predict.request_fingerprint(question, options)
        attachment.write_bytes(attachment.read_bytes() + b'changed')
        self.assertNotEqual(before, predict.request_fingerprint(question, options))

    def test_real_sdk_judge_authentication_failure_saves_other_successes(self):
        options = self.judge_options()
        sdk, requests = self.sdk([json.dumps(judgment()), 401])
        with patch.object(judge, 'client', sdk):
            self.assertEqual(judge.main(options), 1)
        cache = json.loads((self.base / 'judged/judged_predictions.json').read_text())
        self.assertEqual(set(cache), {self.questions[0]['id']})
        self.assertEqual(requests[0]['model'], 'openai/gpt-5.4')
        self.assertEqual(requests[0]['max_completion_tokens'], 4096)
        self.assertEqual(self.csv_row('judge_metrics.csv')['Coverage (%)'], '50.0')
        recovered, requests = self.sdk([json.dumps(judgment())])
        with patch.object(judge, 'client', recovered):
            self.assertEqual(judge.main(options), 0)
            self.assertEqual(judge.main(options), 0)
        self.assertEqual(len(requests), 1)

    def test_invalid_judge_schema_is_retryable_and_cannot_corrupt_metrics(self):
        options = self.judge_options()
        invalid = json.dumps(judgment('yesterday', '90%'))
        sdk, _ = self.sdk([invalid, json.dumps(judgment())])
        with patch.object(judge, 'client', sdk):
            self.assertEqual(judge.main(options), 1)
        self.assertEqual(self.csv_row('judge_metrics.csv')['Evaluated'], '1')
        recovered, requests = self.sdk([json.dumps(judgment())])
        with patch.object(judge, 'client', recovered):
            self.assertEqual(judge.main(options), 0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.csv_row('judge_metrics.csv')['Evaluated'], '2')

    def test_invalid_existing_judge_cache_is_retried(self):
        options = self.judge_options(questions=[self.questions[0]])
        self.save('judged/judged_predictions.json', {self.questions[0]['id']: {'response': 'answer'}})
        sdk, requests = self.sdk([json.dumps(judgment())])
        with patch.object(judge, 'client', sdk):
            self.assertEqual(judge.main(options), 0)
        self.assertEqual(len(requests), 1)

    def test_judge_cache_invalidates_changed_responses_questions_and_judge(self):
        options = self.judge_options()
        sdk, requests = self.sdk([json.dumps(judgment())] * 6)
        with patch.object(judge, 'client', sdk):
            self.assertEqual(judge.main(options), 0)
            self.assertEqual(judge.main(options), 0)
            predictions = json.loads(Path(options.predictions).read_text())
            predictions[self.questions[0]['id']]['response'] = 'changed answer'
            self.save('predictions.json', predictions)
            self.assertEqual(judge.main(options), 0)
            self.questions[1]['answer'] = 'new reference answer'
            self.save('questions.json', self.questions)
            self.assertEqual(judge.main(options), 0)
            options.judge = 'other/gpt-5.4'
            self.assertEqual(judge.main(options), 1)
            self.assertEqual(len(requests), 4)
            options.output_dir = str(self.base / 'other-judged')
            self.assertEqual(judge.main(options), 0)
        self.assertEqual(len(requests), 6)

    def test_missing_predictions_are_incomplete_with_no_calibration_claim(self):
        options = self.judge_options({})
        sdk, requests = self.sdk([])
        with patch.object(judge, 'client', sdk):
            self.assertEqual(judge.main(options), 1)
        self.assertEqual(requests, [])
        row = self.csv_row('judge_metrics.csv')
        self.assertEqual((row['Total'], row['Evaluated'], row['Accuracy (%)']), ('2', '0', '0.0'))
        self.assertEqual(row['CalibrationError (%)'], '')

    def test_invalid_metric_records_are_not_accepted_as_yes(self):
        summary = metrics.compute_metrics({'1': judged_record('yesterday', 100),
                                            '2': judged_record('yes', True),
                                            '3': judged_record('yes', 101)}, 3)
        self.assertEqual(summary['Evaluated'], 0)
        self.assertIsNone(summary['CalibrationError (%)'])

    def test_empty_invalid_and_duplicate_datasets_fail(self):
        for data in ([], [{'id': '1'}], [self.questions[0], self.questions[0]]):
            with self.subTest(data=data):
                options = self.prediction_options()
                self.save('questions.json', data)
                self.assertEqual(predict.main(options), 1)
                options = self.judge_options()
                self.save('questions.json', data)
                self.assertEqual(judge.main(options), 1)

    def test_inconsistent_column_lengths_are_rejected_instead_of_truncating(self):
        incomplete = {'id': ['1', '2'], 'language': ['en', 'en'], 'question': ['only first question'], 'answer': ['one', 'two']}
        options = self.prediction_options()
        self.save('questions.json', incomplete)
        self.assertEqual(predict.main(options), 1)
        options = self.judge_options()
        self.save('questions.json', incomplete)
        self.assertEqual(judge.main(options), 1)

    def test_cli_requires_dataset_predictions_and_positive_limits(self):
        env = {key: value for key, value in os.environ.items() if 'KEY' not in key and 'TOKEN' not in key}
        env['PYTHON_DOTENV_DISABLED'] = '1'
        cases = [('predict.py', ['--model', 'gpt-4o']),
                 ('predict.py', ['--local_dataset', 'missing.json', '--model', 'gpt-4o', '--max_samples', '0']),
                 ('judge.py', ['--local_dataset', 'missing.json']),
                 ('judge.py', ['--local_dataset', 'missing.json', '--predictions', 'missing.json', '--num_workers', '0']),
                 ('metrics.py', ['--query_file', 'missing.json'])]
        for script, arguments in cases:
            with self.subTest(script=script, arguments=arguments):
                result = subprocess.run([sys.executable, str(ROOT / 'evaluation_toolkit/objective_eval' / script), *arguments],
                                        env=env, cwd=self.base, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_prediction_interruption_retains_completed_record(self):
        options = self.prediction_options()
        def interrupted(questions, on_result):
            on_result((questions[0]['id'], 'completed', None, predict.request_fingerprint(questions[0], options)))
            raise KeyboardInterrupt()
        with patch.object(predict, 'attempt_all', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                predict.main(options)
        cache = json.loads((self.base / 'predictions/gpt-4o.json').read_text())
        self.assertEqual(set(cache), {self.questions[0]['id']})

    def test_judge_interruption_retains_completed_record(self):
        options = self.judge_options()
        def interrupted(questions, predictions, on_result):
            record = {**predictions[questions[0]['id']], **judged_record(),
                      'judge_model': options.judge,
                      'judge_fingerprint': judge.judge_fingerprint(questions[0], predictions[questions[0]['id']])}
            on_result((questions[0]['id'], record))
            raise KeyboardInterrupt()
        with patch.object(judge, 'judge_all_responses', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                judge.main(options)
        self.assertEqual(set(json.loads((self.base / 'judged/judged_predictions.json').read_text())), {self.questions[0]['id']})


if __name__ == '__main__':
    unittest.main()
