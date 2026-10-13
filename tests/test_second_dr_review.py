"""Provider citations and interruption behavior, using SDK types or local transports."""
import contextlib
import io
import json
import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI, _base_client
from google import genai
from google.genai import types

from evaluation_toolkit.DR_clients.OpenaiDR import OpenaiDeepResearch
from evaluation_toolkit.DR_clients.GeminiDR import GeminiDeepResearch
from evaluation_toolkit.DR_clients.QwenDR import QwenDeepResearch
from evaluation_toolkit.DR_clients.TavilyResearch import TavilyDeepResearch
from evaluation_toolkit.DR_clients.citations import render_annotations, render_numbered
from evaluation_toolkit.subjective_eval import write
from evaluation_toolkit.subjective_eval.FACT import extract, utils
from evaluation_toolkit.objective_eval import predict, judge
from evaluation_toolkit.subjective_eval.ExpertCriteria import score_by_criteria as expert
from evaluation_toolkit.model_clients import require_complete_chat

HTTP = getattr(_base_client, 'httpx2', None) or getattr(_base_client, 'httpx')
URL = 'https://example.test/source'

class SecondDRReviewTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {'PYTHON_DOTENV_DISABLED': '1'}))
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Live network forbidden')))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def test_openai_real_sdk_preserves_annotations_and_renders_links(self):
        def handle(request):
            return HTTP.Response(200, json={'id': 'resp_local', 'object': 'response', 'created_at': 1,
                'model': 'o3-deep-research', 'status': 'completed', 'output': [
                {'id': 'msg_local', 'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [
                {'type': 'output_text', 'text': 'Verified fact.', 'annotations': [
                {'type': 'url_citation', 'start_index': 0, 'end_index': 14, 'title': 'Source', 'url': URL}]}]}]})
        dr = OpenaiDeepResearch(model_choose='o3-deep-research', api_key='fake', poll_interval=0)
        dr.client.close()
        dr.client = OpenAI(api_key='fake', max_retries=0, http_client=HTTP.Client(transport=HTTP.MockTransport(handle)))
        self.addCleanup(dr.client.close)
        result = dr.run_research('system', 'topic')
        self.assertEqual(result['content'], 'Verified fact. [Source](' + URL + ')')
        self.assertEqual(result['raw_citations'][0]['type'], 'url_citation')
        self.assertEqual(result['citations'][0]['url'], URL)

    def test_gemini_real_sdk_uses_utf8_byte_offsets(self):
        def handle(request):
            return httpx.Response(200, json={'id': 'interaction_local', 'object': 'interaction',
                'status': 'in_progress' if request.method == 'POST' else 'completed',
                'steps': [] if request.method == 'POST' else [
                {'type': 'model_output', 'content': [{'type': 'text', 'text': '事实。后文', 'annotations': [
                {'type': 'url_citation', 'start_index': 0, 'end_index': 9, 'title': 'Source', 'url': URL}]}]}]})
        dr = GeminiDeepResearch(api_key='fake', poll_interval=0)
        dr.client.close()
        dr.client = genai.Client(api_key='fake', http_options=types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handle))))
        self.addCleanup(dr.client.close)
        result = dr.run_research('system', 'topic')
        self.assertEqual(result['content'], '事实。 [Source](' + URL + ')后文')
        self.assertEqual(result['raw_citations'][0]['end_index'], 9)

    def test_invalid_offsets_do_not_fabricate_claim_mapping(self):
        text, citations = render_annotations('fact', [{'url': URL, 'start_index': 0, 'end_index': 99}])
        self.assertEqual(text, 'fact')
        self.assertEqual(citations[0]['url'], URL)
        self.assertEqual(render_annotations('中', [{'url': URL, 'start_index': 0, 'end_index': 1}], byte_offsets=True)[0], '中')

    def test_numbered_citations_use_explicit_ids_not_list_order(self):
        result, _ = render_numbered('Fact[7]; other[1].', [{'url': URL, 'index_number': 7}, {'url': 'https://other.test'}])
        self.assertEqual(result, 'Fact[7](' + URL + '); other[1].')
        result, _ = render_numbered('Fact[1]', [{'url': URL, 'index_number': 1}, {'url': 'https://other.test', 'index_number': 1}])
        self.assertEqual(result, 'Fact[1]')

    def test_qwen_final_answer_references_are_preserved(self):
        dr = QwenDeepResearch(api_key='fake')
        stream = iter([NS(output={'message': {'phase': 'WebResearch', 'content': 'ignored', 'extra': {'deep_research': {'references': [{'url': 'https://ignored.test', 'index_number': 1}]}}}}),
            NS(output={'message': {'phase': 'answer', 'status': 'finished', 'content': 'Fact[7]',
                'extra': {'deep_research': {'references': [{'url': URL, 'title': 'Source', 'index_number': 7}]}}}, 'finish_reason': 'stop'}, usage={})])
        content, _ = dr._collect_answer(stream, float('inf'))
        self.assertEqual(content, 'Fact[7](' + URL + ')')
        self.assertEqual(len(dr._last_raw_citations), 1)

    def test_qwen_truncation_cannot_be_reported_as_finished(self):
        dr = QwenDeepResearch(api_key='fake')
        for key in ('finish_reason', 'finished_reason', 'fininshed_reason'):
            message = {'phase': 'answer', 'status': 'finished', 'content': 'partial report', key: 'length'}
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'length'):
                dr._collect_answer(iter([NS(output={'message': message})]), float('inf'))

    def test_tavily_sources_without_ids_are_preserved_but_not_guessed(self):
        dr = object.__new__(TavilyDeepResearch)
        dr.timeout_seconds, dr.poll_interval = 30, 0
        dr.client = NS(research=Mock(return_value={'request_id': 'local'}), get_research=Mock(return_value={
            'status': 'completed', 'content': 'Fact[1]', 'sources': [{'url': URL, 'title': 'Source'}]}))
        result = dr.run_research('system', 'topic')
        self.assertEqual(result['content'], 'Fact[1]')
        self.assertEqual(result['raw_citations'][0]['url'], URL)
        self.assertIn('citation_mapping_error', result)
        call = Mock()
        with patch.object(extract, 'client', NS(chat=NS(completions=NS(create=call)))):
            record, error = extract.extract_report({'id': '1', 'response': result['content'],
                'citation_mapping_error': result['citation_mapping_error']}, {'1': 'en'})
        call.assert_not_called()
        self.assertEqual(error, 1)
        self.assertEqual(record['citations'], 'extraction failed')

    def test_tavily_explicit_reference_id_is_linked(self):
        dr = object.__new__(TavilyDeepResearch)
        dr.timeout_seconds, dr.poll_interval = 30, 0
        dr.client = NS(research=Mock(return_value={'request_id': 'local'}), get_research=Mock(return_value={
            'status': 'completed', 'content': 'Fact[7]', 'sources': [{'url': URL, 'index_number': 7}]}))
        result = dr.run_research('system', 'topic')
        self.assertEqual(result['content'], 'Fact[7](' + URL + ')')
        self.assertNotIn('citation_mapping_error', result)

    def test_tavily_late_completed_result_is_rejected(self):
        dr = object.__new__(TavilyDeepResearch)
        dr.timeout_seconds, dr.poll_interval = 30, 0
        dr.client = NS(research=Mock(return_value={'request_id': 'local'}), get_research=Mock(return_value={
            'status': 'completed', 'content': 'late report'}))
        with patch.object(dr, 'poll_wait'), patch.object(dr, 'check_deadline', side_effect=TimeoutError('expired')), self.assertRaises(TimeoutError):
            dr.run_research('system', 'topic')

    def test_report_writer_persists_citations_and_mapping_error(self):
        result = {'content': 'Fact[1]', 'citations': [{'url': URL}], 'raw_citations': [{'url': URL, 'title': 'Source'}],
                  'citation_mapping_error': 'missing explicit mapping'}
        with tempfile.TemporaryDirectory() as directory:
            query = Path(directory) / 'questions.json'
            query.write_text(json.dumps([{'id': '1', 'language': 'en', 'question': 'Topic'}]))
            options = NS(local_dataset=str(query), dataset=None, model='tavily-deep-research',
                num_workers=1, max_samples=None, temperature=0, max_completion_tokens=128, output_dir=directory)
            with patch.object(write, 'create_dr_client', return_value=NS(run_research=Mock(return_value=result))):
                self.assertEqual(write.main(options), 0)
            record = json.loads((Path(directory) / 'reports_tavily-deep-research.json').read_text())['1']
            self.assertEqual(record['provider_citations'], result['citations'])
            self.assertEqual(record['provider_raw_citations'], result['raw_citations'])
            self.assertEqual(record['citation_mapping_error'], result['citation_mapping_error'])

    def background_client(self, kind):
        if kind == 'openai':
            dr = object.__new__(OpenaiDeepResearch)
            dr.model_name = 'o3-deep-research'
            api = NS(create=Mock(return_value=NS(id='local', status='queued')), retrieve=Mock(), cancel=Mock())
            dr.client = NS(responses=api)
        else:
            dr = object.__new__(GeminiDeepResearch)
            dr.agent_name = 'test'
            api = NS(create=Mock(return_value=NS(id='local', status='in_progress')), get=Mock(return_value=NS(id='local', status='in_progress')), cancel=Mock())
            dr.client = NS(interactions=api)
        dr.timeout_seconds, dr.poll_interval = 30, 0
        return dr, api

    def test_background_poll_timeout_cancels_openai_and_gemini(self):
        for kind in ('openai', 'gemini'):
            dr, api = self.background_client(kind)
            failure = TimeoutError('expired')
            method = 'poll_wait'
            with self.subTest(kind=kind), patch.object(dr, method, side_effect=failure), self.assertRaises(Exception):
                dr.run_research('system', 'topic')
            api.cancel.assert_called_once_with('local', timeout=10)
            self.assertEqual(failure.research_task_id, 'local')

    def test_background_interrupt_cancels_without_masking_original(self):
        for kind in ('openai', 'gemini'):
            dr, api = self.background_client(kind)
            failure = KeyboardInterrupt()
            method = 'poll_wait'
            with self.subTest(kind=kind), patch.object(dr, method, side_effect=failure), self.assertRaises(KeyboardInterrupt):
                dr.run_research('system', 'topic')
            api.cancel.assert_called_once()

    def test_failed_cancellation_retains_task_id_and_original_failure(self):
        dr, api = self.background_client('openai')
        api.cancel.side_effect = RuntimeError('cancel unavailable')
        failure = TimeoutError('expired')
        with patch.object(dr, 'poll_wait', side_effect=failure), self.assertRaisesRegex(RuntimeError, 'expired') as raised:
            dr.run_research('system', 'topic')
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(failure.research_task_id, 'local')
        self.assertIn('cancel unavailable', failure.research_cancel_error)

    def test_late_completed_result_does_not_bypass_deadline(self):
        for kind in ('openai', 'gemini'):
            dr, api = self.background_client(kind)
            completed = NS(id='local', status='completed', output_text='late report')
            (api.retrieve if kind == 'openai' else api.get).return_value = completed
            with self.subTest(kind=kind), patch.object(dr, 'poll_wait'), patch.object(dr, 'check_deadline', side_effect=[None, None, TimeoutError('expired')]), self.assertRaises(Exception):
                dr.run_research('system', 'topic')
            api.cancel.assert_not_called()

    def test_parent_pipeline_interrupt_stops_research_worker_and_cancels_job(self):
        for pipeline in (write, predict):
            with self.subTest(pipeline=pipeline.__name__), tempfile.TemporaryDirectory() as directory:
                query = Path(directory) / 'questions.json'
                query.write_text(json.dumps([{'id': '1', 'language': 'en', 'question': 'Topic'}]))
                options = NS(local_dataset=str(query), dataset=None, model='o3-deep-research', num_workers=1,
                    max_samples=None, temperature=0, max_completion_tokens=128, output_dir=directory)
                dr, api = self.background_client('openai')
                dr.poll_interval = 30
                created, cancelled = threading.Event(), threading.Event()
                def create(**kwargs):
                    created.set()
                    return NS(id='local', status='queued')
                api.create.side_effect = create
                api.cancel.side_effect = lambda *args, **kwargs: cancelled.set()
                def interrupt(*args, **kwargs):
                    self.assertTrue(created.wait(2), 'Worker did not start')
                    raise KeyboardInterrupt()
                with patch.object(pipeline, 'create_dr_client', return_value=dr), patch.object(pipeline, 'as_completed', side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        pipeline.main(options)
                    self.assertTrue(cancelled.wait(2), 'Parent interruption did not cancel remote task')
                self.assertTrue(options.research_cancel_event.is_set())
                api.cancel.assert_called_once_with('local', timeout=10)

    def test_tavily_timeout_retains_uncancelled_remote_task_id(self):
        dr = object.__new__(TavilyDeepResearch)
        dr.timeout_seconds, dr.poll_interval = 30, 0
        dr.client = NS(research=Mock(return_value={'request_id': 'tavily-local'}))
        with patch.object(dr, 'poll_wait', side_effect=TimeoutError('expired')), self.assertRaises(TimeoutError) as raised:
            dr.run_research('system', 'topic')
        self.assertEqual(raised.exception.research_task_id, 'tavily-local')

    def truncated_client(self, content):
        response = NS(choices=[NS(finish_reason='length', message=NS(content=content))], usage=None)
        return NS(chat=NS(completions=NS(create=Mock(return_value=response))))

    def test_truncated_objective_prediction_is_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            query = Path(directory) / 'questions.json'
            query.write_text(json.dumps([{'id': '1', 'language': 'en', 'question': 'Topic'}]))
            options = NS(local_dataset=str(query), dataset=None, model='gpt-4o', num_workers=1,
                max_samples=None, temperature=0, max_completion_tokens=1, output_dir=directory)
            with patch.object(predict, 'client', self.truncated_client('nonempty partial answer')):
                self.assertEqual(predict.main(options), 1)
            self.assertEqual(json.loads((Path(directory) / 'gpt-4o.json').read_text()), {})

    def test_truncated_subjective_report_is_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            query = Path(directory) / 'questions.json'
            query.write_text(json.dumps([{'id': '1', 'language': 'en', 'question': 'Topic'}]))
            options = NS(local_dataset=str(query), dataset=None, model='gpt-4o', num_workers=1,
                max_samples=None, temperature=0, max_completion_tokens=1, output_dir=directory)
            with patch.object(write, 'client', self.truncated_client('nonempty partial report')):
                self.assertEqual(write.main(options), 1)
            self.assertEqual(json.loads((Path(directory) / 'reports_gpt-4o.json').read_text()), {})

    def test_valid_json_with_truncated_finish_cannot_be_expert_score(self):
        with patch.object(expert, 'args', NS(judge='gpt-5.4'), create=True), patch.object(expert, 'client', self.truncated_client('[{"dimension_id":"1.1","score":80}]')):
            with self.assertRaisesRegex(RuntimeError, 'length'):
                expert.call_llm([{'role': 'user', 'content': 'score report'}])

    def test_valid_json_with_truncated_finish_cannot_be_objective_judgment(self):
        payload = json.dumps({'extracted_final_answer': 'answer', 'reasoning': 'test', 'correct': 'yes', 'confidence': 100})
        with patch.object(judge, 'args', NS(judge='gpt-5.4'), create=True), patch.object(judge, 'client', self.truncated_client(payload)):
            self.assertIsNone(judge.extract_answer('question', 'answer', 'answer', 'en'))

    def test_filtered_and_tool_call_outputs_are_incomplete(self):
        for reason in ('length', 'content_filter', 'tool_calls', 'function_call'):
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                require_complete_chat(NS(choices=[NS(finish_reason=reason)]))

    def test_FACT_rejects_duplicate_ids_and_malformed_dedup_cache(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            utils.records_by_id([{'id': '1'}, {'id': 1}])
        current = {'citations': [{'fact': 'a', 'ref_idx': 1, 'url': URL}, {'fact': 'b', 'ref_idx': 1, 'url': URL}]}
        self.assertFalse(utils.valid_deduplication_cache(current, {'citations_deduped': {}}))
        self.assertFalse(utils.valid_deduplication_cache(current, {'citations_deduped': {URL: {'facts': []}}}))
        self.assertFalse(utils.valid_deduplication_cache(current, {'citations_deduped': {URL: {'facts': ['invented']}}}))
        self.assertTrue(utils.valid_deduplication_cache(current, {'citations_deduped': {URL: {'facts': ['a']}}}))

if __name__ == '__main__':
    unittest.main()
