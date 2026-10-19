"""Deduplication resume must preserve the current report and task identity."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ['PYTHON_DOTENV_DISABLED'] = '1'

from evaluation_toolkit.subjective_eval.scoring.provenance import question_fingerprint, report_fingerprint


ROOT = Path(__file__).resolve().parents[1]
URL = 'https://example.invalid/source'


class DedupProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.query = {'id': '1', 'language': 'en', 'question': 'Write a report.'}
        self.report = {'id': '1', 'model': 'gpt-4o', 'response': 'Original report.',
                       'citations': [{'fact': 'The recorded fact.', 'ref_idx': 1, 'url': URL}]}
        self.raw = self.base / 'extracted.jsonl'
        self.output = self.base / 'deduplicated.jsonl'
        self.queries = self.base / 'queries.json'

    def run_stage(self, query, report):
        current = copy.deepcopy(report)
        current['question_fingerprint'] = question_fingerprint(query)
        current['report_fingerprint'] = report_fingerprint('1', current)
        self.raw.write_text(json.dumps(current) + '\n', encoding='utf-8')
        self.queries.write_text(json.dumps([query]), encoding='utf-8')
        # One fact per URL needs no model call. Empty credentials also ensure a
        # regression cannot silently contact a paid provider during this test.
        result = subprocess.run(
            [sys.executable, '-m', 'evaluation_toolkit.subjective_eval.FACT.deduplicate',
             '--raw_data_path', str(self.raw), '--output_path', str(self.output),
             '--query_data_path', str(self.queries)],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
            env=dict(os.environ, OPENROUTER_API_KEY='', PYTHON_DOTENV_DISABLED='1'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        saved = json.loads(self.output.read_text(encoding='utf-8'))
        return current, saved, result.stdout

    def seed_cache(self):
        _, cached, _ = self.run_stage(self.query, self.report)
        cached['old_report_metadata'] = 'Must not survive a new extraction.'
        cached['citations_deduped'][URL].update(
            url_content='Previous source text.', publish_time='2000-01-01',
            validate_res=[{'idx': 0, 'result': 'supported'}],
            validation_model='previous-judge', validate_error=None)
        self.output.write_text(json.dumps(cached) + '\n', encoding='utf-8')

    def test_cached_fact_selection_keeps_current_report_question_and_model(self):
        changes = ('response', 'question', 'model')
        for field in changes:
            with self.subTest(changed=field):
                self.seed_cache()
                query, report = copy.deepcopy(self.query), copy.deepcopy(self.report)
                if field == 'question':
                    query['question'] = 'Revised research request.'
                elif field == 'model':
                    report['model'] = 'other-provider/other-model'
                else:
                    report['response'] = 'Regenerated report with the same cited claim.'
                current, saved, output = self.run_stage(query, report)
                self.assertIn('Processing 0 reports', output)
                for key, value in current.items():
                    self.assertEqual(saved[key], value, key)
                self.assertNotIn('old_report_metadata', saved)
                self.assertEqual(saved['citations_deduped'][URL],
                                 {'facts': ['The recorded fact.'], 'url_content': None})

    def test_changed_facts_and_language_invalidate_cached_decision(self):
        for change in ('fact', 'language'):
            with self.subTest(changed=change):
                self.seed_cache()
                query, report = copy.deepcopy(self.query), copy.deepcopy(self.report)
                if change == 'fact':
                    report['citations'][0]['fact'] = 'An edited fact.'
                else:
                    query['language'] = 'zh'
                current, saved, output = self.run_stage(query, report)
                self.assertIn('Processing 1 reports', output)
                self.assertEqual(saved['dedup_language'], query['language'])
                self.assertEqual(saved['report_fingerprint'], current['report_fingerprint'])
                self.assertEqual(saved['citations_deduped'][URL],
                                 {'facts': [report['citations'][0]['fact']], 'url_content': None})

    def test_failed_extraction_cannot_restore_cached_facts(self):
        self.seed_cache()
        failed = dict(self.report, citations='extraction failed', extraction_error='Provider failed')
        self.raw.write_text(json.dumps(failed) + '\n', encoding='utf-8')
        result = subprocess.run(
            [sys.executable, '-m', 'evaluation_toolkit.subjective_eval.FACT.deduplicate',
             '--raw_data_path', str(self.raw), '--output_path', str(self.output),
             '--query_data_path', str(self.queries)],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
            env=dict(os.environ, OPENROUTER_API_KEY='', PYTHON_DOTENV_DISABLED='1'))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        saved = json.loads(self.output.read_text(encoding='utf-8'))
        self.assertEqual(saved['citations_deduped'], {})
        self.assertIn('dedup_error', saved)


if __name__ == '__main__':
    unittest.main()
