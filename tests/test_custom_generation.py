"""Custom tested-system integration; no external model or network access."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from evaluation_toolkit import generation
from evaluation_toolkit.objective_eval import predict, judge
from evaluation_toolkit.subjective_eval import write

ROOT = Path(__file__).resolve().parents[1]


class CustomGenerationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('No network allowed')))
        self.stack.enter_context(patch.object(generation, 'load_project_env'))
        self.stack.enter_context(patch.dict(os.environ, {'OPENROUTER_API_KEY': 'judge-only-secret',
                                                       'ICBCBENCH_MODEL_API_KEY': 'tested-only-secret'}))
        self.requests = []
        self.client_options = []

    def options(self, **overrides):
        options = dict(model='gpt-my-own-model', model_backend='openai',
                       model_base_url='https://my-model.example/v1',
                       model_api_key_env='ICBCBENCH_MODEL_API_KEY', model_timeout=12.5,
                       agent=None, attachment_mode='url', num_workers=1,
                       max_samples=None, temperature=0.2, max_completion_tokens=100,
                       local_dataset=None, dataset=None, output_dir=str(self.base / 'outputs'))
        options.update(overrides)
        return SimpleNamespace(**options)

    def sdk_transport(self, *, content='An answer [source](https://example.org/report).',
                      finish_reason='stop', usage=None, choices=True):
        def respond(request):
            self.requests.append(request)
            body = {'id': 'response-1', 'object': 'chat.completion', 'created': 0,
                    'model': 'gpt-my-own-model', 'choices': [
                        {'index': 0, 'message': {'role': 'assistant', 'content': content},
                         'finish_reason': finish_reason}] if choices else []}
            if usage is not None:
                body['usage'] = usage
            return httpx.Response(200, json=body)
        def factory(**kwargs):
            self.client_options.append(kwargs)
            return OpenAI(http_client=httpx.Client(transport=httpx.MockTransport(respond)), **kwargs)
        return patch.object(generation, 'OpenAI', side_effect=factory)

    def save(self, name, value):
        path = self.base / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return str(path)

    def adapter(self, source):
        package = self.base / 'custom_test_adapter'
        package.mkdir(exist_ok=True)
        (package / '__init__.py').write_text('PACKAGE_IMPORTED = True\n', encoding='utf-8')
        path = package / 'worker.py'
        path.write_text(source, encoding='utf-8')
        self.stack.enter_context(patch.object(sys, 'path', [str(self.base), *sys.path]))
        self.addCleanup(lambda: sys.modules.pop('custom_test_adapter.worker', None))
        self.addCleanup(lambda: sys.modules.pop('custom_test_adapter', None))
        return self.options(model_backend='agent', model_base_url=None,
                            agent='custom_test_adapter.worker:generate'), path

    def test_generation_options_defaults_and_help_require_no_credentials(self):
        parser = argparse.ArgumentParser()
        generation.add_generation_arguments(parser)
        options = parser.parse_args([])
        self.assertEqual(options.model_backend, 'builtin')
        self.assertEqual(options.attachment_mode, 'image')
        self.assertEqual(options.model_timeout, 120)
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args(['--model_timeout', value])

    def test_custom_sdk_keeps_model_exact_and_does_not_override_judge(self):
        original_judge = judge.client
        with self.sdk_transport(usage={'prompt_tokens': 3, 'completion_tokens': 5, 'total_tokens': 8}):
            result = generation.generate_custom(self.options(), task_id='1', track='objective', language='en',
                                                messages=[{'role': 'user', 'content': 'question'}])
        body = json.loads(self.requests[0].content)
        self.assertEqual(body['model'], 'gpt-my-own-model')
        self.assertEqual(body['temperature'], 0.2)
        self.assertNotIn('extra_body', body)
        self.assertNotIn('reasoning', body)
        self.assertNotIn('thinking', body)
        self.assertEqual(self.requests[0].url.host, 'my-model.example')
        self.assertEqual(self.requests[0].headers['authorization'], 'Bearer tested-only-secret')
        self.assertEqual(self.client_options[0]['max_retries'], 1)
        self.assertEqual(self.client_options[0]['timeout'], 12.5)
        self.assertEqual(result['usage']['total_tokens'], 8)
        self.assertIs(judge.client, original_judge)
        self.assertEqual(os.environ['OPENROUTER_API_KEY'], 'judge-only-secret')

    def test_local_endpoint_can_omit_key_and_never_inherits_judge_key(self):
        os.environ.pop('ICBCBENCH_MODEL_API_KEY', None)
        with self.sdk_transport():
            generation.generate_custom(self.options(model_base_url='http://127.0.0.1:8000/v1'),
                                       task_id='1', track='objective', language='en', messages=[])
        self.assertEqual(self.requests[0].headers['authorization'], 'Bearer icbcbench-local-no-key')

    def test_custom_prediction_then_real_judge_sdk_uses_separate_credentials(self):
        question = {'id': '1', 'language': 'en', 'question': 'Compute two plus two', 'answer': '4'}
        options = self.options(local_dataset=self.save('objective.json', [question]))
        with self.sdk_transport(content='The final answer is 4.'):
            self.assertEqual(predict.main(options), 0)
        judge_requests = []
        def judge_response(request):
            judge_requests.append(request)
            return httpx.Response(200, json={
                'id': 'judge-1', 'object': 'chat.completion', 'created': 0, 'model': 'judge-model',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                    'role': 'assistant', 'content': json.dumps({'extracted_final_answer': '4',
                    'reasoning': 'Matches the reference.', 'correct': 'yes', 'confidence': 100})}}]})
        with OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1',
                    http_client=httpx.Client(transport=httpx.MockTransport(judge_response))) as judge_sdk:
            judge_options = SimpleNamespace(local_dataset=options.local_dataset, dataset=None, num_workers=1,
                                            predictions=str(Path(options.output_dir) / 'gpt-my-own-model.json'),
                                            judge='gpt-5.4', output_dir=str(self.base / 'judged'),
                                            metrics_csv=str(self.base / 'metrics.csv'))
            with patch.object(judge, 'client', judge_sdk):
                self.assertEqual(judge.main(judge_options), 0)
        self.assertEqual(self.requests[0].url.host, 'my-model.example')
        self.assertEqual(self.requests[0].headers['authorization'], 'Bearer tested-only-secret')
        self.assertEqual(judge_requests[0].url.host, 'openrouter.ai')
        self.assertEqual(judge_requests[0].headers['authorization'], 'Bearer judge-only-secret')
        self.assertEqual(json.loads(judge_requests[0].content)['model'], 'openai/gpt-5.4')

    def test_custom_remote_requires_separate_key(self):
        os.environ.pop('ICBCBENCH_MODEL_API_KEY', None)
        with self.assertRaisesRegex(ValueError, 'Missing ICBCBENCH_MODEL_API_KEY'):
            generation.validate_generation_credentials(self.options())

    def test_configuration_contains_no_key_values(self):
        config = generation.generation_config(self.options())
        serialized = json.dumps(config)
        self.assertNotIn('tested-only-secret', serialized)
        self.assertNotIn('judge-only-secret', serialized)
        self.assertEqual(config['model_api_key_env'], 'ICBCBENCH_MODEL_API_KEY')

    def test_rejects_credential_urls_before_logging_value(self):
        urls = ('https://user:secret@example.org/v1', 'https://example.org/v1?key=secret',
                'https://example.org/v1#secret', 'http://remote.example/v1',
                'file:///secret', 'https://example.org:secret/v1', 'https://example.org/v1\nsecret')
        for url in urls:
            with self.subTest(url=url), self.assertRaises(ValueError) as caught:
                generation.generation_config(self.options(model_base_url=url))
            self.assertNotIn('secret', str(caught.exception))

    def test_backend_arguments_cannot_be_silently_ignored(self):
        for options in (self.options(model_backend='builtin'),
                        self.options(agent='foo:bar'),
                        self.options(model_backend='agent', model_base_url=None, agent=None)):
            with self.assertRaises(ValueError):
                generation.generation_config(options)

    def test_custom_response_must_be_finished_nonempty_and_have_valid_usage(self):
        for overrides in ({'finish_reason': 'length'}, {'finish_reason': None}, {'choices': False},
                          {'content': ''}, {'content': None},
                          {'usage': {'prompt_tokens': -1, 'completion_tokens': 1, 'total_tokens': 0}}):
            with self.subTest(overrides=overrides), self.sdk_transport(**overrides), self.assertRaises(ValueError):
                generation.generate_custom(self.options(), task_id='1', track='objective', language='en', messages=[])

    def test_agent_config_hashes_source_without_importing_it(self):
        options, path = self.adapter('raise RuntimeError("must not run while validating")\n')
        first = generation.generation_config(options)
        self.assertNotIn('custom_test_adapter', sys.modules)
        self.assertNotIn('custom_test_adapter.worker', sys.modules)
        self.assertEqual(len(first['agent_source_files']), 2)
        path.write_text('raise RuntimeError("changed")\n')
        self.assertNotEqual(first, generation.generation_config(options))

    def test_agent_receives_only_sanitized_inputs_and_preserves_citations(self):
        options, _ = self.adapter('''
def generate(request):
    assert set(request) == {"id", "track", "language", "model", "messages"}
    assert request["track"] == "objective"
    assert "TOP-SECRET-ANSWER" not in str(request)
    assert "HIDDEN-RUBRIC" not in str(request)
    return {"content": "Answer [1](https://example.org/source)",
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            "citations": [{"url": "https://example.org/source"}]}
''')
        question = {'id': '8', 'question': 'Compute the ratio', 'language': 'en',
                    'answer': 'TOP-SECRET-ANSWER', 'expert_evaluation_criteria': 'HIDDEN-RUBRIC'}
        options.local_dataset = self.save('objective.json', [question])
        with patch.object(predict, 'create_dr_client', side_effect=AssertionError('Built-in provider forbidden')):
            self.assertEqual(predict.main(options), 0)
        result = json.loads((Path(options.output_dir) / 'gpt-my-own-model.json').read_text())['8']
        self.assertEqual(result['response'], 'Answer [1](https://example.org/source)')
        self.assertEqual(result['provider_citations'], [{'url': 'https://example.org/source'}])

    def test_agent_subjective_report_generation_no_rubric_leak(self):
        options, _ = self.adapter('''
def generate(request):
    assert request["track"] == "subjective"
    assert request["language"] == "en"
    assert "HIDDEN-RUBRIC" not in str(request)
    assert "UNWANTED-ANSWER" not in str(request)
    return {"content": "Report [source](https://example.org/source)", "citations": ["https://example.org/source"]}
''')
        question = {'id': '2', 'question': 'Research the market', 'language': 'en',
                    'answer': 'UNWANTED-ANSWER', 'expert_evaluation_criteria': 'HIDDEN-RUBRIC'}
        options.local_dataset = self.save('subjective.json', [question])
        with patch.object(write, 'create_dr_client', side_effect=AssertionError('Built-in provider forbidden')):
            self.assertEqual(write.main(options), 0)
            self.assertEqual(write.main(options), 0)
        result = json.loads((Path(options.output_dir) / 'reports_gpt-my-own-model.json').read_text())['2']
        self.assertEqual(result['provider_citations'], ['https://example.org/source'])

    def test_invalid_agent_outputs_fail_without_checkpoints(self):
        for expression in ('"string"', '{}', '{"content": ""}', '{"content": "answer", "usage": []}',
                           '{"content": "answer", "usage": {"total_tokens": -1}}',
                           '{"content": "answer", "citations": "bad"}'):
            # Remove module between subcases, just like separate CLI invocations.
            sys.modules.pop('custom_test_adapter.worker', None)
            sys.modules.pop('custom_test_adapter', None)
            options, _ = self.adapter(f'def generate(request):\n    return {expression}\n')
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                generation.generate_custom(options, task_id='1', track='objective', language='en', messages=[])

    def test_async_adapter_has_actionable_error(self):
        options, _ = self.adapter('async def generate(request):\n    return {"content": "answer"}\n')
        with self.assertRaisesRegex(ValueError, 'must be synchronous'):
            generation.generate_custom(options, task_id='1', track='objective', language='en', messages=[])

    def test_agent_cannot_hide_failure_behind_nonempty_content(self):
        for metadata in ({'error': 'provider failure'}, {'status': 'failed'}, {'finish_reason': 'length'},
                         {'failed': True}, {'success': False}, {'completed': False},
                         {'incomplete_details': {'reason': 'timeout'}}, {'truncated': True}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                generation._validated_result({'content': 'Partial answer', **metadata})
        result = generation._validated_result({'content': 'Complete answer', 'status': 'completed',
                                               'finish_reason': 'stop', 'success': True})
        self.assertEqual(result['content'], 'Complete answer')

    def test_url_attachment_custom_model_never_downloads_pdf(self):
        question = {'id': '25', 'language': 'en', 'question': 'Compute RWA', 'answer': 'PRIVATE-REFERENCE',
                    'attachment': [{'source_url': 'https://example.org/source.pdf', 'pdf_page': 8,
                                    'printed_page': 44, 'description': 'Dashboard', 'mime_type': 'application/pdf'}]}
        options = self.options(local_dataset=self.save('questions.json', [question]))
        with self.sdk_transport(), patch.object(predict.requests, 'get', side_effect=AssertionError('No PDF fetch')):
            self.assertEqual(predict.main(options), 0)
            self.assertEqual(predict.main(options), 0)
        self.assertEqual(len(self.requests), 1)
        body = json.loads(self.requests[0].content)
        self.assertEqual(body['model'], options.model)
        self.assertIn('https://example.org/source.pdf', body['messages'][1]['content'])
        self.assertNotIn('PRIVATE-REFERENCE', str(body))

    def test_cache_fingerprint_tracks_backend_endpoint_mode_and_agent_source(self):
        question = {'id': '1', 'language': 'en', 'question': 'q'}
        options = self.options()
        first = predict.request_fingerprint(question, options)
        for overrides in ({'model_base_url': 'https://another.example/v1'}, {'attachment_mode': 'image'},
                          {'model_backend': 'builtin', 'model_base_url': None}, {'model': 'openai/gpt-my-own-model'}):
            self.assertNotEqual(first, predict.request_fingerprint(question, self.options(**overrides)))
        options, path = self.adapter('def generate(request):\n    return {"content": "one"}\n')
        first = predict.request_fingerprint(question, options)
        path.write_text('def generate(request):\n    return {"content": "two"}\n')
        self.assertNotEqual(first, predict.request_fingerprint(question, options))

    def test_incomplete_custom_prediction_is_a_failure(self):
        options = self.options(local_dataset=self.save('questions.json', [{'id': '1', 'language': 'en', 'question': 'q'}]))
        with self.sdk_transport(finish_reason='length'):
            self.assertEqual(predict.main(options), 1)
        self.assertEqual(json.loads((Path(options.output_dir) / 'gpt-my-own-model.json').read_text()), {})

    def test_custom_subjective_model_does_not_send_provider_specific_reasoning(self):
        options = self.options(model='kimi-k2.6', local_dataset=self.save('topics.json', [{'id': '1', 'language': 'en', 'question': 'q'}]))
        with self.sdk_transport():
            self.assertEqual(write.main(options), 0)
        body = json.loads(self.requests[0].content)
        self.assertEqual(body['model'], 'kimi-k2.6')
        self.assertNotIn('thinking', body)
        self.assertNotIn('reasoning', body)


if __name__ == '__main__':
    unittest.main()
