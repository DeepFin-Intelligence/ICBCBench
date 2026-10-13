"""Second-pass cache and attachment regressions, without live model requests."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import pymupdf

from evaluation_toolkit.objective_eval import predict, judge, metrics
from evaluation_toolkit.subjective_eval import write


class SecondRuntimeReviewTests(unittest.TestCase):
    def test_objective_ids_are_normalized_and_invalid_types_rejected(self):
        for identifier in (True, False, [], {}, 1.5, ' '):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                metrics.validate_questions([{'id': identifier, 'language': 'en'}])
        with self.assertRaisesRegex(ValueError, 'unique'):
            metrics.validate_questions([{'id': ' 1 ', 'language': 'en'}, {'id': 1, 'language': 'en'}])
        self.assertEqual(metrics.validate_questions([{'id': ' 1 ', 'language': 'en'}])[0]['id'], '1')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='second-runtime-review-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.questions = [{'id': '1', 'language': 'en', 'question': 'first topic', 'answer': 'first answer'},
                          {'id': '2', 'language': 'zh', 'question': 'second topic', 'answer': 'second answer'}]
        self.dataset = self.base / 'questions.json'
        self.save(self.dataset, self.questions)

    def save(self, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def options(self, model='gpt-4o'):
        return SimpleNamespace(local_dataset=str(self.dataset), dataset=None, model=model,
                               max_samples=None, num_workers=1, temperature=0,
                               max_completion_tokens=128, output_dir=str(self.base / 'out'))

    def fake_prediction(self, options):
        def respond(questions, on_result):
            results = []
            for question in questions:
                result = (question['id'], 'answer from ' + options.model, None,
                          predict.request_fingerprint(question, options))
                on_result(result)
                results.append(result)
            return results
        return respond

    def fake_report(self, question):
        return question['id'], 'report for ' + question['question'], None

    def test_failed_changed_topic_replacement_removes_old_report(self):
        options = self.options()
        with patch.object(write, 'attempt_query', side_effect=self.fake_report):
            self.assertEqual(write.main(options), 0)
        self.questions[0]['question'] = 'changed topic'
        self.save(self.dataset, self.questions)
        with patch.object(write, 'attempt_query', return_value=None):
            self.assertEqual(write.main(options), 1)
        saved = json.loads((self.base / 'out/reports_gpt-4o.json').read_text())
        self.assertEqual(set(saved), {'2'})

    def test_interrupted_replacement_removes_old_report_before_request(self):
        options = self.options()
        with patch.object(write, 'attempt_query', side_effect=self.fake_report):
            write.main(options)
        self.questions[0]['question'] = 'changed topic'
        self.save(self.dataset, self.questions)
        with patch.object(write, 'as_completed', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                write.main(options)
        self.assertNotIn('1', json.loads((self.base / 'out/reports_gpt-4o.json').read_text()))

    def test_report_alias_reuses_canonical_model_cache(self):
        options = self.options()
        with patch.object(write, 'attempt_query', side_effect=self.fake_report) as request:
            self.assertEqual(write.main(options), 0)
            options.model = 'openai/gpt-4o'
            self.assertEqual(write.main(options), 0)
        self.assertEqual(request.call_count, 2)

    def test_report_subset_discards_stale_unselected_and_foreign_tasks(self):
        options = self.options()
        with patch.object(write, 'attempt_query', side_effect=self.fake_report):
            self.assertEqual(write.main(options), 0)
            options.max_samples = 1
            self.questions[1]['question'] = 'changed unselected topic'
            self.save(self.dataset, self.questions)
            self.assertEqual(write.main(options), 0)
        self.assertEqual(set(json.loads((self.base / 'out/reports_gpt-4o.json').read_text())), {'1'})
        self.save(self.dataset, [self.questions[0]])
        with patch.object(write, 'attempt_query', side_effect=self.fake_report):
            self.assertEqual(write.main(options), 0)
        self.assertEqual(set(json.loads((self.base / 'out/reports_gpt-4o.json').read_text())), {'1'})

    def test_prediction_different_provider_subset_cannot_mix_models(self):
        options = self.options('provider-a/shared-model')
        with patch.object(predict, 'attempt_all', side_effect=self.fake_prediction(options)):
            self.assertEqual(predict.main(options), 0)
            before = (self.base / 'out/shared-model.json').read_bytes()
            options.model, options.max_samples = 'provider-b/shared-model', 1
            self.assertEqual(predict.main(options), 1)
        self.assertEqual((self.base / 'out/shared-model.json').read_bytes(), before)

    def test_prediction_canonical_alias_is_reusable(self):
        options = self.options()
        with patch.object(predict, 'attempt_all', side_effect=self.fake_prediction(options)) as request:
            self.assertEqual(predict.main(options), 0)
            options.model = 'openai/gpt-4o'
            self.assertEqual(predict.main(options), 0)
        self.assertEqual(request.call_args.args[0], [])

    def test_prediction_subset_discards_stale_unselected_tasks(self):
        options = self.options()
        with patch.object(predict, 'attempt_all', side_effect=self.fake_prediction(options)):
            self.assertEqual(predict.main(options), 0)
            self.questions[1]['question'] = 'changed unselected topic'
            self.save(self.dataset, self.questions)
            options.max_samples = 1
            self.assertEqual(predict.main(options), 0)
        self.assertEqual(set(json.loads((self.base / 'out/gpt-4o.json').read_text())), {'1'})

    def record(self, question, judge_model='gpt-5.4'):
        record = {'model': 'gpt-4o', 'response': 'answer', 'judge_model': judge_model,
                  'judge_result': {'correct': 'yes', 'confidence': 100,
                                   'correct_answer': question['answer'], 'model_answer': 'answer', 'reasoning': 'evidence'}}
        record['judge_fingerprint'] = metrics.judge_fingerprint(question, record, judge_model)
        return record

    def standalone(self, records, judge_model='gpt-5.4'):
        self.save(self.base / 'judged/judged_gpt-4o.json', records)
        return metrics.dump_all_metrics(judge_model, None, judged_dir=str(self.base / 'judged'),
                                        query_file=str(self.dataset), results_dir=str(self.base / 'metrics'))

    def test_standalone_metrics_rejects_changed_question_or_reference_answer(self):
        for field in ('question', 'answer'):
            with self.subTest(field=field):
                questions = copy.deepcopy(self.questions)
                records = {question['id']: self.record(question) for question in questions}
                self.save(self.dataset, questions)
                self.assertEqual(self.standalone(records), 0)
                questions[0][field] += ' changed'
                self.save(self.dataset, questions)
                self.assertEqual(self.standalone(records), 1)
                row = pd.read_csv(self.base / 'metrics/evaluation_results.csv').iloc[0]
                self.assertEqual(row['Evaluated'], 1)
                self.assertEqual(row['Coverage (%)'], 50)

    def test_standalone_metrics_rejects_different_judge_and_legacy_results(self):
        records = {question['id']: self.record(question) for question in self.questions}
        self.assertEqual(self.standalone(records, 'other/gpt-5.4'), 1)
        for record in records.values():
            del record['judge_fingerprint']
        self.assertEqual(self.standalone(records), 1)

    def test_judge_rejects_mixed_prediction_models(self):
        prediction_path = self.base / 'predictions.json'
        self.save(prediction_path, {'1': {'model': 'provider-a/shared', 'response': 'first'},
                                    '2': {'model': 'provider-b/shared', 'response': 'second'}})
        options = SimpleNamespace(local_dataset=str(self.dataset), dataset=None,
                                  predictions=str(prediction_path), judge='gpt-5.4', num_workers=1,
                                  output_dir=str(self.base / 'judged'), metrics_csv=str(self.base / 'metrics.csv'))
        with patch.object(judge, 'judge_all_responses') as request:
            self.assertEqual(judge.main(options), 1)
        request.assert_not_called()

    def test_judge_output_cannot_be_reused_by_different_provider(self):
        prediction_path = self.base / 'predictions.json'
        self.save(prediction_path, {'1': {'model': 'gpt-4o', 'response': 'first'}})
        previous = self.record(self.questions[0], 'provider-a/shared-judge')
        path = self.base / 'judged/judged_predictions.json'
        self.save(path, {'1': previous})
        before = path.read_bytes()
        options = SimpleNamespace(local_dataset=str(self.dataset), dataset=None,
                                  predictions=str(prediction_path), judge='provider-b/shared-judge', num_workers=1,
                                  output_dir=str(self.base / 'judged'), metrics_csv=str(self.base / 'metrics.csv'))
        with patch.object(judge, 'judge_all_responses') as request:
            self.assertEqual(judge.main(options), 1)
        request.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_dr_report_preserves_provider_citations(self):
        options = self.options('tavily-research')
        response = {'content': 'a report', 'usage': {'total_tokens': 2},
                    'citations': [{'url': 'https://example.com/evidence', 'title': 'Evidence'}],
                    'raw_citations': [{'provider_id': 'source-1'}]}
        client = SimpleNamespace(run_research=lambda **kwargs: response)
        with patch.object(write, 'create_dr_client', return_value=client):
            self.assertEqual(write.main(options), 0)
        records = json.loads((self.base / 'out/reports_tavily-research.json').read_text())
        self.assertEqual(records['1']['provider_citations'], response['citations'])
        self.assertEqual(records['1']['provider_raw_citations'], response['raw_citations'])

    def pdf(self, filename, pages=1, table=False, blank=False):
        path = self.base / filename
        document = pymupdf.open()
        for index in range(pages):
            page = document.new_page()
            if blank:
                continue
            if table:
                for y in (50, 80, 110):
                    page.draw_line((40, y), (300, y))
                for x in (40, 170, 300):
                    page.draw_line((x, 50), (x, 110))
                for x, y, text in [(50, 70, 'HeaderA'), (180, 70, 'HeaderB'),
                                   (50, 100, 'ValueA'), (180, 100, 'ValueB')]:
                    page.insert_text((x, y), text)
            else:
                page.insert_text((50, 70), f'Content from page {index + 1}')
        document.save(path)
        document.close()
        return path

    def test_pdf_table_format_failure_preserves_table_text(self):
        path = self.pdf('table.pdf', table=True)
        with pymupdf.open(path) as document:
            self.assertEqual(len(document[0].find_tables().tables), 1)
        with patch.object(pd.DataFrame, 'to_markdown', side_effect=ImportError('missing tabulate')):
            text = predict.extract_pdf_text(path)
        self.assertIn('ValueA', text)
        self.assertIn('ValueB', text)

    def test_blank_pdf_is_not_valid_attachment_content(self):
        path = self.pdf('blank.pdf', blank=True)
        self.assertEqual(predict.extract_pdf_text(path), '')
        with self.assertRaisesRegex(ValueError, 'extract required PDF'):
            predict.build_attachment_content([{'file_path': str(path), 'mime_type': 'application/pdf',
                                               'original_filename': 'blank.pdf'}])

    def test_pdf_default_reads_beyond_ten_pages(self):
        self.assertIn('Content from page 12', predict.extract_pdf_text(self.pdf('long.pdf', pages=12)))

    def test_excel_default_reads_beyond_two_hundred_rows(self):
        path = self.base / 'long.xlsx'
        pd.DataFrame({'rows': [f'row-{index}' for index in range(211)]}).to_excel(path, index=False)
        self.assertIn('row-210', predict.extract_excel_text(path))

    def test_empty_excel_is_not_valid_attachment_content(self):
        path = self.base / 'empty.xlsx'
        pd.DataFrame().to_excel(path, index=False)
        self.assertEqual(predict.extract_excel_text(path), '')
        with self.assertRaisesRegex(ValueError, 'extract required Excel'):
            predict.build_attachment_content([{'file_path': str(path), 'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                               'original_filename': 'empty.xlsx'}])

    def test_relative_attachments_resolve_against_custom_dataset(self):
        source = self.pdf('relative.pdf')
        attachment = {'file_path': source.name, 'mime_type': 'application/pdf', 'original_filename': source.name}
        blocks, _ = predict.build_attachment_content([attachment], str(self.dataset))
        self.assertIn('Content from page 1', blocks[0]['text'])
        question = dict(self.questions[0], attachment=[attachment])
        self.assertTrue(predict.request_fingerprint(question, self.options()))


if __name__ == '__main__':
    unittest.main()
