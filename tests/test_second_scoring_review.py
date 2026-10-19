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
from evaluation_toolkit.subjective_eval.ExpertCriteria.score_by_criteria import (
    parse_criteria_dimensions, validate_dimension_scores)
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
                    'Avg Source Quality Score (S_source)': 100,
                    'Evaluation As Of (UTC)': '2026-10-16T00:00:00+00:00',
                    **self.metadata, **(fact_changes or {})}
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
        args = SimpleNamespace(input_path=str(path), model_name='gpt-4o', language_filter=None,
                               as_of='2026-10-16')
        return fact_stat.run_stat(args, {'1': 'en'}, str(self.base / 'fact_stat.csv'), None,
                                  [self.question] if queries is None else queries)

    def test_identical_inputs_and_model_aliases_merge(self):
        expert, fact = self.inputs()
        result = self.merge([expert], fact)
        self.assertEqual(result.iloc[0]['S_final'], 84)
        self.assertTrue(result.iloc[0]['Provenance Verified'])

    def test_expert_intermediate_precision_survives_multiround_aggregation(self):
        score = 80.006
        record = self.score_record(total_score=score,
                                   dimensions_breakdown=[{'dimension_id': '1.1', 'score': score}])
        self.assertEqual(self.expert_statistics([record]), 0)
        first = self.base / 'expert_stat.csv'
        frame = pd.read_csv(first)
        self.assertEqual(frame.iloc[0]['1'], score)
        self.assertEqual(frame.iloc[0]['Avg Score'], score)
        rounds = [first]
        for number in (2, 3):
            path = self.base / f'round_{number}.csv'
            path.write_bytes(first.read_bytes())
            rounds.append(path)
        _, fact = self.inputs(fact_changes={'Avg Source Quality Score (S_source)': 90})
        result = self.merge(rounds, fact)
        self.assertEqual(result.iloc[0]['S_final'], 83.00)
        self.assertEqual(result.iloc[0]['Expert Rounds'], 3)

    def add_second_model(self, expert, fact, reference_date):
        report = {**self.report, 'model': 'vendor/second-model'}
        report_hash = report_set_fingerprint({'1': report_fingerprint('1', report)})
        for path, name_column in ((expert, 'Model'), (fact, 'Model Name')):
            frame = pd.read_csv(path)
            row = frame.iloc[0].to_dict()
            row[name_column] = report['model']
            row['Report Set Fingerprint'] = report_hash
            if name_column == 'Model Name':
                row['Evaluation As Of (UTC)'] = reference_date
            pd.concat([frame, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)

    def test_corrected_public_rubric_fraction_is_preserved_in_chinese_mean(self):
        questions = [question for question in json.loads(
            (ROOT / 'data/subjective_questions_public_40.json').read_text()) if question['language'] == 'zh']
        records = []
        for question in questions:
            criteria = question['expert_evaluation_criteria']
            caps = parse_criteria_dimensions(criteria)
            scores = [{'dimension_id': key, 'score': cap * .8} for key, cap in caps.items()]
            if question['id'] == '36':
                # Integer raw scores become 80 + 1/3 after the approved correction.
                remaining = {'1': 25, '2': 28, '3': 30, '4': 2}
                for dimension in scores:
                    key = dimension['dimension_id']
                    primary = key.split('.')[0]
                    dimension['score'] = min(caps[key], remaining[primary])
                    remaining[primary] -= dimension['score']
            records.append({'task_id': question['id'], 'model': self.report['model'],
                            'total_score': validate_dimension_scores(scores, criteria),
                            'dimensions_breakdown': scores,
                            'question_fingerprint': question_fingerprint(question),
                            'report_fingerprint': report_fingerprint(question['id'], self.report)})
        directory = self.base / 'chinese_scores'
        directory.mkdir()
        (directory / 'scores.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))
        query = self.base / 'chinese_questions.json'
        query.write_text(json.dumps(questions))
        expert = self.base / 'chinese.csv'
        self.assertEqual(expert_stat.stat(str(directory), str(query), str(expert)), 0)
        row = pd.read_csv(expert).iloc[0]
        self.assertAlmostEqual(row['36'], 80 + 1 / 3)
        self.assertAlmostEqual(row['Avg Score'], 80 + 1 / 60)
        _, fact = self.inputs(fact_changes={**{key: row[key] for key in self.metadata},
                                           'Avg Source Quality Score (S_source)': 90})
        result = self.merge([expert], fact)
        self.assertEqual(result.iloc[0]['S_final'], 83.01)

    def test_different_FACT_reference_dates_cannot_be_ranked_together(self):
        expert, fact = self.inputs()
        self.add_second_model(expert, fact, '2027-10-16')
        with self.assertRaisesRegex(ValueError, 'different evaluation reference dates.*--as_of'):
            self.merge([expert], fact)

    def test_equivalent_FACT_timezones_are_normalized_to_same_instant(self):
        expert, fact = self.inputs()
        self.add_second_model(expert, fact, '2026-10-15T20:00:00-04:00')
        result = self.merge([expert], fact)
        self.assertEqual(result['Evaluation As Of (UTC)'].tolist(),
                         ['2026-10-16T00:00:00+00:00'] * 2)

    def test_unused_FACT_models_do_not_constrain_reference_date(self):
        expert, fact = self.inputs()
        self.add_second_model(expert, fact, '2027-10-16')
        frame = pd.read_csv(expert).iloc[:1]
        frame.to_csv(expert, index=False)
        result = self.merge([expert], fact)
        self.assertEqual(len(result), 1)

    def test_missing_FACT_reference_column_requests_explicit_rerun(self):
        expert, fact = self.inputs()
        pd.read_csv(fact).drop(columns=['Evaluation As Of (UTC)']).to_csv(fact, index=False)
        with self.assertRaisesRegex(ValueError, 'lacks Evaluation As Of.*--as_of'):
            self.merge([expert], fact)

    def test_invalid_FACT_reference_dates_are_rejected(self):
        for reference_date in (None, '', ' ', 'not-a-date', '2026-02-30'):
            with self.subTest(reference_date=reference_date):
                expert, fact = self.inputs(fact_changes={'Evaluation As Of (UTC)': reference_date})
                with self.assertRaisesRegex(ValueError, 'invalid Evaluation As Of.*--as_of'):
                    self.merge([expert], fact)

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
