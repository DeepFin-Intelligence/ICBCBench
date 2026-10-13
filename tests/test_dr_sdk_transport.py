"""DR request/response contracts through real SDKs and local HTTP transports."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI, _base_client
import requests
import pymupdf
from google import genai
from google.genai import types

from evaluation_toolkit.DR_clients.GeminiDR import GeminiDeepResearch
from evaluation_toolkit.DR_clients.OpenaiDR import OpenaiDeepResearch
from evaluation_toolkit.DR_clients.TavilyResearch import TavilyDeepResearch
from evaluation_toolkit.DR_clients.QwenDR import QwenDeepResearch

OPENAI_HTTP = getattr(_base_client, 'httpx2', None) or getattr(_base_client, 'httpx')
ROOT = Path(__file__).resolve().parents[1]


def two_images():
    content = [{'type': 'text', 'text': 'Read both required images.'}]
    with pymupdf.open() as document:
        for color in ((1, 0, 0), (0, 0, 1)):
            page = document.new_page(width=32, height=32)
            page.draw_rect(page.rect, color=color, fill=color)
            image = base64.b64encode(page.get_pixmap().tobytes('png')).decode()
            content.append({'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image}'}})
    return content


class LocalRequestsAdapter(requests.adapters.BaseAdapter):
    def __init__(self, handler):
        self.handler = handler

    def send(self, request, **options):
        status, body = self.handler(request, options)
        response = requests.Response()
        response.status_code, response.request, response.url = status, request, request.url
        response.encoding = 'utf-8'
        response.headers['Content-Type'] = 'text/event-stream' if isinstance(body, bytes) else 'application/json'
        response._content = body if isinstance(body, bytes) else json.dumps(body).encode()
        response.raw = io.BytesIO(response._content)
        return response

    def close(self):
        pass


class DRSDKTransportTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {'PYTHON_DOTENV_DISABLED': '1'}))
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Live network forbidden')))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def openai_run(self, content):
        requests_seen = []
        def handler(request):
            requests_seen.append({'method': request.method, 'path': request.url.path,
                                  'body': json.loads(request.content) if request.content else None})
            completed = request.method == 'GET'
            return OPENAI_HTTP.Response(200, json={
                'id': 'resp_local', 'object': 'response', 'created_at': 1,
                'model': 'o3-deep-research', 'status': 'completed' if completed else 'queued',
                'output': [{'id': 'message_local', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                            'content': [{'type': 'output_text', 'text': 'final report', 'annotations': []}]}] if completed else [],
                'usage': {'input_tokens': 1, 'output_tokens': 2, 'total_tokens': 3},
            })
        dr = OpenaiDeepResearch(model_choose='o3-deep-research', api_key='fake', timeout_seconds=30, poll_interval=0)
        dr.client.close()
        dr.client = OpenAI(api_key='fake', max_retries=0,
                           http_client=OPENAI_HTTP.Client(transport=OPENAI_HTTP.MockTransport(handler)))
        self.addCleanup(dr.client.close)
        result = dr.run_research('system instructions', '', multimodal_content=content)
        self.assertEqual(result['content'], 'final report')
        self.assertEqual(result['usage']['total_tokens'], 3)
        self.assertEqual([row['method'] for row in requests_seen], ['POST', 'GET'])
        self.assertEqual(requests_seen[1]['path'], '/v1/responses/resp_local')
        body = requests_seen[0]['body']
        self.assertTrue(body['background'])
        self.assertEqual(body['model'], 'o3-deep-research')
        self.assertEqual({tool['type'] for tool in body['tools']}, {'web_search_preview', 'code_interpreter'})
        self.assertEqual(body['tools'][1]['container'], {'type': 'auto'})
        return body

    def test_openai_background_create_retrieve_and_real_output_text(self):
        body = self.openai_run('actual research question')
        self.assertIn('actual research question', body['input'])
        self.assertIn('system instructions', body['input'])

    def test_openai_background_request_preserves_both_images(self):
        body = self.openai_run(two_images())
        content = body['input'][1]['content']
        images = [block for block in content if block['type'] == 'input_image']
        self.assertEqual(len(images), 2)
        self.assertTrue(all(block['image_url'].startswith('data:image/png;base64,') for block in images))
        self.assertTrue(any(block['type'] == 'input_text' for block in content))

    def gemini_run(self, content):
        requests_seen = []
        def handler(request):
            requests_seen.append({'method': request.method, 'path': request.url.path,
                                  'body': json.loads(request.content) if request.content else None})
            return httpx.Response(200, json={
                'id': 'interaction_local', 'object': 'interaction',
                'status': 'in_progress' if request.method == 'POST' else 'completed',
                'steps': [] if request.method == 'POST' else [
                    {'type': 'model_output', 'content': [{'type': 'text', 'text': 'final report'}]}],
            })
        dr = GeminiDeepResearch(api_key='fake', timeout_seconds=30, poll_interval=0)
        self.assertEqual(dr.agent_name, 'deep-research-preview-04-2026')
        dr.client.close()
        dr.client = genai.Client(api_key='fake', http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
        self.addCleanup(dr.client.close)
        self.assertEqual(dr.run_research('system instructions', '', multimodal_content=content)['content'], 'final report')
        self.assertEqual([row['method'] for row in requests_seen], ['POST', 'GET'])
        self.assertEqual(requests_seen[0]['path'], '/v1beta/interactions')
        self.assertEqual(requests_seen[1]['path'], '/v1beta/interactions/interaction_local')
        body = requests_seen[0]['body']
        self.assertEqual(body['agent'], 'deep-research-preview-04-2026')
        self.assertTrue(body['background'])
        return body

    def test_gemini_default_agent_real_create_get_and_steps_response(self):
        body = self.gemini_run('actual research question')
        self.assertIn('actual research question', body['input'])

    def test_gemini_real_sdk_serializes_both_images(self):
        body = self.gemini_run(two_images())
        # Current GAOS SDK normalizes a content list into a user_input step.
        content = body['input'][0]['content'] if body['input'][0].get('type') == 'user_input' else body['input']
        images = [block for block in content if block['type'] == 'image']
        self.assertEqual(len(images), 2)
        self.assertTrue(all(block['mime_type'] == 'image/png' and block['data'] for block in images))

    def test_tavily_real_research_and_pending_result_timeout(self):
        requests_seen = []
        def handler(request, options):
            requests_seen.append({'method': request.method, 'url': request.url,
                                  'body': json.loads(request.body) if request.body else None,
                                  'timeout': options.get('timeout')})
            if request.method == 'POST':
                return 200, {'request_id': 'research_local', 'status': 'pending'}
            if len(requests_seen) == 2:
                return 202, {'request_id': 'research_local', 'status': 'in_progress'}
            return 200, {'request_id': 'research_local', 'status': 'completed', 'content': 'final report'}
        dr = TavilyDeepResearch(api_key='fake', timeout_seconds=30, poll_interval=0)
        dr.client.session.mount('https://', LocalRequestsAdapter(handler))
        self.addCleanup(dr.client.session.close)
        self.assertEqual(dr.run_research('system instructions', '', multimodal_content='actual question')['content'], 'final report')
        self.assertEqual([row['method'] for row in requests_seen], ['POST', 'GET', 'GET'])
        self.assertEqual(requests_seen[0]['url'], 'https://api.tavily.com/research')
        self.assertTrue(requests_seen[1]['url'].endswith('/research/research_local'))
        self.assertIn('actual question', requests_seen[0]['body']['input'])
        self.assertTrue(all(0 < row['timeout'] <= 30 for row in requests_seen))

    def qwen_run(self, content):
        from dashscope.api_entities import http_request
        requests_seen = []
        def handler(request, options):
            requests_seen.append({'method': request.method, 'body': json.loads(request.body), 'timeout': options.get('timeout')})
            messages = requests_seen[-1]['body']['input']['messages']
            self.assertTrue(all(type(message['content']) is str for message in messages))
            self.assertTrue(all(message['role'] in ('user', 'assistant') for message in messages))
            answer = 'Which area should the report cover?' if len(requests_seen) == 1 else 'final report'
            events = [
                {'phase': 'KeepAlive', 'status': 'typing', 'content': 'progress is not an answer'},
                {'phase': 'answer', 'status': 'typing', 'content': answer},
                {'phase': 'answer', 'status': 'finished', 'content': ''},
            ]
            stream = []
            for message in events:
                event = {'status_code': 200, 'request_id': 'qwen_local', 'code': '', 'message': '',
                         'output': {'message': {'role': 'assistant', **message}},
                         'usage': {'input_tokens': 1, 'output_tokens': 2}}
                stream.append('data: ' + json.dumps(event) + '\n\n')
            return 200, ''.join(stream).encode()
        session = requests.Session()
        session.mount('https://', LocalRequestsAdapter(handler))
        self.addCleanup(session.close)
        dr = QwenDeepResearch(api_key='fake', timeout_seconds=30, poll_interval=0)
        with patch.object(http_request, '_get_shared_sync_session', return_value=session):
            result = dr.run_research('system instructions', '', multimodal_content=content)
        self.assertEqual(result, {'content': 'final report', 'usage': {'input_tokens': 2, 'output_tokens': 4, 'total_tokens': 6}, 'citations': [], 'raw_citations': []})
        self.assertEqual(len(requests_seen), 2)
        self.assertEqual(requests_seen[0]['body']['model'], 'qwen-deep-research')
        self.assertTrue(requests_seen[0]['body']['parameters']['incremental_output'])
        self.assertIn('system instructions', requests_seen[0]['body']['input']['messages'][0]['content'])
        self.assertEqual(requests_seen[1]['body']['input']['messages'][1]['content'], 'Which area should the report cover?')
        self.assertTrue(all(0 < row['timeout'] <= 30 for row in requests_seen))

    def test_qwen_real_sdk_text_clarification_and_answer_stream(self):
        self.qwen_run('actual question')

    def test_qwen_text_blocks_are_combined_into_string_messages(self):
        self.qwen_run([{'type': 'text', 'text': 'actual question'}, {'type': 'text', 'text': 'text attachment'}])


if __name__ == '__main__':
    unittest.main()
