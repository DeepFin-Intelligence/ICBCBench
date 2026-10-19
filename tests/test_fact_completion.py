"""Never accept syntactically valid FACT JSON from an unfinished model response."""
import contextlib
import io
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ['PYTHON_DOTENV_DISABLED'] = '1'
from evaluation_toolkit.subjective_eval.FACT import extract, deduplicate, validate


def client_for(content, reason):
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=reason, message=SimpleNamespace(content=json.dumps(content)))])
    create = Mock(return_value=response)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class FactCompletionTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(validate.time, 'sleep'))

    def test_extraction_requires_finished_output_even_when_json_is_valid(self):
        for reason in ('tool_calls', 'function_call', 'error', 'length', 'content_filter', 'stop'):
            with self.subTest(reason=reason):
                client = client_for([], reason)
                with patch.object(extract, 'client', client):
                    record, failed = extract.extract_report({'id': '1', 'response': 'A report.'}, {'1': 'en'})
                self.assertEqual(failed, int(reason != 'stop'))
                self.assertEqual(client.chat.completions.create.call_count, 1 if reason == 'stop' else 3)
                if reason != 'stop':
                    self.assertEqual(record['citations'], 'extraction failed')

    def test_deduplication_requires_finished_output_even_when_indices_are_valid(self):
        citations = [{'fact': fact, 'url': 'https://example.test', 'ref_idx': 0} for fact in ('First fact', 'Second fact')]
        for reason in ('tool_calls', 'function_call', 'error', 'length', 'content_filter', 'stop'):
            with self.subTest(reason=reason):
                client = client_for([1], reason)
                with patch.object(deduplicate, 'client', client):
                    record, failed = deduplicate.deduplicate_report({'id': '1', 'citations': citations}, {'1': 'en'})
                self.assertEqual(failed, int(reason != 'stop'))
                self.assertEqual(client.chat.completions.create.call_count, 1 if reason == 'stop' else 3)
                if reason != 'stop':
                    self.assertEqual(record['citations_deduped']['https://example.test']['facts'], ['First fact', 'Second fact'])

    def test_validation_requires_finished_output_even_when_judgments_are_valid(self):
        item = ('https://example.test', {'article_id': '1', 'url_content': 'Reference', 'facts': ['A fact']})
        for reason in ('tool_calls', 'function_call', 'error', 'length', 'content_filter', 'stop'):
            with self.subTest(reason=reason):
                client = client_for([{'idx': 1, 'result': 'supported'}], reason)
                with patch.object(validate, 'client', client):
                    result = validate.validate(item, {'1': 'en'})
                self.assertEqual(result['error'] is not None, reason != 'stop')
                self.assertEqual(client.chat.completions.create.call_count, 1 if reason == 'stop' else 3)
                if reason != 'stop':
                    self.assertEqual(result['validate_res'], [])


if __name__ == '__main__':
    unittest.main()
