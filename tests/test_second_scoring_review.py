"""Independent scoring checks: exact inputs must match across all stages."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PYTHON_DOTENV_DISABLED'] = '1'

from evaluation_toolkit.subjective_eval.ExpertCriteria import stat as expert_stat
from evaluation_toolkit.subjective_eval.FACT import stat as fact_stat
from evaluation_toolkit.subjective_eval.scoring.compute_final_score import compute_final_score
from evaluation_toolkit.subjective_eval.scoring.provenance import (
    question_fingerprint, task_set_fingerprint, report_fingerprint, report_set_fingerprint)


RUBRIC = '### 1. Quality (100 points)\n#### (1) Evidence (100 points)'


class SecondScoringReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.question = {'id': '1', 'language': 'en', 'question': 'Topic', 'expert_evaluation_criteria': RUBRIC}
        self.report = {'model': 'gpt-4o', 'response': 'A generated research report.'}
        self.metadata = {'Total Tasks': 1, 'Evaluated Tasks': 1, 'Coverage (%)': 100,
                         'Task Set Fingerprint': task_set_fingerprint([self.question]),
                         'Report Set Fingerprint': report_set_fingerprint({'1': report_fingerprint('1', self.report)}),
                         'Provenance Verified': True}

    def tearDown(self):
        self.temporary.cleanup()

    def inputs(self, expert_changes=None, fact_changes=None):
        expert_row = {'Model': 'gpt-4o', 'Avg Score': 80, **self.metadata, **(expert_changes or {})}
        fact_row = {'Model Name': 'openai/gpt-4o', 'Avg Citation Score (Macro)': 100,
                    'Avg Source Quality Score (S_source)': 100, **self.metadata, **(fact_changes or {})}
        expert_path, fact_path = self.base / 'expert.csv', self.base / 'fact.csv'
        pd.DataFrame([expert_row]).to_csv(expert_path, index=False)
        pd.DataFrame([fact_row]).to_csv(fact_path, index=False)
        return expert_path, fact_path

    def merge(self, experts, fact):
        return compute_final_score([str(path) for path in experts], str(fact), str(self.base / 'final.csv'), .8, .1, .1)

    def score_record(self, **changes):
        return {'task_id': '1', 'model': 'gpt-4o', 'total_score': 80,
                'dimensions_breakdown': [{'dimension_id': '1.1', 'score': 80}],
                'question_fingerprint': question_fingerprint(self.question),
                'report_fingerprint': report_fingerprint('1', self.report), **changes}

    def expert_statistics(self, records):
        directory = self.base / 'scores'
        directory.mkdir(exist_ok=True)
        (directory / 'scores.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))
        query = self.base / 'queries.json'
        query.write_text(json.dumps([self.question]))
        return expert_stat.stat(str(directory), str(query), str(self.base / 'expert_stat.csv'))

    def fact_statistics(self, record, queries=None):
        path = self.base / 'facts.jsonl'
        path.write_text(json.dumps(record) + '\n')
        args = SimpleNamespace(input_path=str(path), model_name='gpt-4o', language_filter=None)
        return fact_stat.run_stat(args, {'1': 'en'}, str(self.base / 'fact_stat.csv'), None,
                                  [self.question] if queries is None else queries)

    def test_identical_inputs_and_model_aliases_merge(self):
        expert, fact = self.inputs()
        result = self.merge([expert], fact)
        self.assertEqual(result.iloc[0]['S_final'], 84)
        self.assertTrue(result.iloc[0]['Provenance Verified'])

    def test_report_aliases_share_fingerprint(self):
        self.assertEqual(report_fingerprint('1', self.report),
                         report_fingerprint('1', {**self.report, 'model': ' OpenAI/GPT-4o '}))

    def test_query_order_does_not_affect_fingerprint(self):
        second = {**self.question, 'id': '2'}
        self.assertEqual(task_set_fingerprint([self.question, second]), task_set_fingerprint([second, self.question]))

    def test_different_counts_are_rejected_even_with_forged_same_hash(self):
        expert, fact = self.inputs(fact_changes={'Total Tasks': 40, 'Evaluated Tasks': 40})
        with self.assertRaisesRegex(ValueError, 'task counts'):
            self.merge([expert], fact)

    def test_same_count_different_question_text_is_rejected(self):
        other = {**self.question, 'question': 'A different topic'}
        expert, fact = self.inputs(fact_changes={'Task Set Fingerprint': task_set_fingerprint([other])})
        with self.assertRaisesRegex(ValueError, 'question sets'):
            self.merge([expert], fact)

    def test_old_FACT_for_regenerated_reports_is_rejected(self):
        other = {**self.report, 'response': 'Regenerated report'}
        expert, fact = self.inputs(fact_changes={'Report Set Fingerprint': report_set_fingerprint({'1': report_fingerprint('1', other)})})
        with self.assertRaisesRegex(ValueError, 'different generated reports'):
            self.merge([expert], fact)

    def test_expert_rounds_require_same_questions(self):
        expert, fact = self.inputs()
        other = self.base / 'second_round.csv'
        frame = pd.read_csv(expert)
        frame['Task Set Fingerprint'] = task_set_fingerprint([{**self.question, 'id': '2'}])
        frame.to_csv(other, index=False)
        with self.assertRaisesRegex(ValueError, 'question sets'):
            self.merge([expert, other], fact)

    def test_expert_rounds_require_same_model_set(self):
        expert, fact = self.inputs()
        other = self.base / 'second_round.csv'
        frame = pd.read_csv(expert)
        frame['Model'] = 'another/model'
        frame.to_csv(other, index=False)
        with self.assertRaisesRegex(ValueError, 'different model sets'):
            self.merge([expert, other], fact)

    def test_legacy_csv_cannot_claim_verified_final_score(self):
        expert, fact = self.inputs()
        frame = pd.read_csv(expert).drop(columns=['Task Set Fingerprint', 'Report Set Fingerprint', 'Provenance Verified'])
        frame.to_csv(expert, index=False)
        with self.assertRaisesRegex(ValueError, 'provenance metadata'):
            self.merge([expert], fact)

    def test_missing_model_or_score_does_not_become_successful_zero(self):
        for record in ({'task_id': '1'}, {'task_id': '1', 'model': 'gpt-4o'}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.expert_statistics([record])

    def test_duplicate_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate expert score'):
            self.expert_statistics([self.score_record(), self.score_record()])

    def test_expert_total_must_match_valid_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.expert_statistics([self.score_record(total_score=100)])

    def test_fresh_expert_output_has_verified_provenance(self):
        self.assertEqual(self.expert_statistics([self.score_record()]), 0)
        frame = pd.read_csv(self.base / 'expert_stat.csv')
        self.assertEqual(frame.iloc[0]['Report Set Fingerprint'], self.metadata['Report Set Fingerprint'])
        self.assertTrue(frame.iloc[0]['Provenance Verified'])

    def test_changed_question_cannot_relabel_stale_expert_scores_as_verified(self):
        self.assertEqual(self.expert_statistics([self.score_record(question_fingerprint='a' * 64)]), 1)
        self.assertFalse(pd.read_csv(self.base / 'expert_stat.csv').iloc[0]['Provenance Verified'])

    def test_fresh_FACT_output_matches_expert_fingerprints(self):
        record = {'id': '1', **self.report, 'citations_deduped': {},
                  'question_fingerprint': question_fingerprint(self.question),
                  'report_fingerprint': report_fingerprint('1', self.report)}
        self.assertEqual(self.fact_statistics(record), 0)
        frame = pd.read_csv(self.base / 'fact_stat.csv')
        self.assertEqual(frame.iloc[0]['Task Set Fingerprint'], self.metadata['Task Set Fingerprint'])
        self.assertEqual(frame.iloc[0]['Report Set Fingerprint'], self.metadata['Report Set Fingerprint'])

    def test_legacy_FACT_or_wrong_model_is_unverified(self):
        legacy = {'id': '1', 'citations_deduped': {}}
        self.assertGreater(self.fact_statistics(legacy), 0)
        wrong_model = {'id': '1', **self.report, 'model': 'another/model', 'citations_deduped': {},
                       'question_fingerprint': question_fingerprint(self.question),
                       'report_fingerprint': report_fingerprint('1', self.report)}
        self.assertGreater(self.fact_statistics(wrong_model), 0)

    def test_empty_FACT_question_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.fact_statistics({'id': '1', 'citations_deduped': {}}, queries=[])


if __name__ == '__main__':
    unittest.main()
