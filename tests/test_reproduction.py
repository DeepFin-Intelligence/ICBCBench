"""Fresh-checkout workflow and the explicitly versioned scoring convention."""
import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['PYTHON_DOTENV_DISABLED'] = '1'

from evaluation_toolkit import cli
from evaluation_toolkit.offline_demo import run_demo
from evaluation_toolkit.subjective_eval.FACT.time_decay import calculate_time_score, reference_time
from evaluation_toolkit.subjective_eval.scoring.compute_final_score import compute_final_score, run_all_versions


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(cli.platform, "platform", return_value="test-platform"))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def options(self, *extra):
        return cli.parser().parse_args(['run', '--model', 'gpt-4o', '--run_dir', str(self.base / 'run'),
                                       '--as_of', '2026-10-16', '--limit', '2', *extra])

    def test_fixed_date_and_timezone_define_reproducible_age(self):
        expected = round(math.exp(-.002 * 10), 4)
        self.assertEqual(calculate_time_score('2026-10-06', as_of='2026-10-16'), expected)
        self.assertEqual(calculate_time_score('2026-10-06T02:00:00+02:00', as_of='2026-10-15T20:00:00-04:00'), expected)
        self.assertEqual(calculate_time_score('2026', as_of='2026-01-11'), expected)
        self.assertEqual(calculate_time_score('unknown', as_of='2026-10-16'), -1)
        self.assertEqual(calculate_time_score('2030-01-01', as_of='2026-10-16'), 1)
        with self.assertRaises(ValueError):
            calculate_time_score('2026-10-06', as_of='not-a-date')
        self.assertEqual(reference_time('2026-10-16').isoformat(), '2026-10-16T00:00:00+00:00')

    def test_release_check_has_no_keys_or_network_dependency(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': ''}):
            self.assertEqual(cli.main(['check']), 0)

    def test_pipeline_subsets_are_shared_and_rounds_are_independent(self):
        config, datasets, steps = cli.build_plan(self.options())
        self.assertEqual(config['task_ids'], {'objective': ['1', '41'], 'subjective': ['1', '21']})
        for track, questions in datasets.items():
            self.assertEqual(len(questions), 2)
            self.assertEqual({q['language'] for q in questions}, {'en', 'zh'})
            for step in steps:
                command = step['command']
                for flag in ('--query_file', '--question_path', '--query_data_path', '--local_dataset'):
                    if flag in command and track in command[3]:
                        self.assertEqual(command[command.index(flag) + 1], str(self.base / 'run/inputs' / f'{track}.json'))
        scores = [s for s in steps if s['name'].startswith('expert-score-')]
        self.assertEqual(len(scores), 3)
        self.assertEqual(len({s['command'][s['command'].index('--output_dir') + 1] for s in scores}), 3)
        self.assertFalse((self.base / 'run').exists())

    def test_single_language_subset_does_not_require_absent_language_expert_scores(self):
        _, datasets, steps = cli.build_plan(self.options('--language', 'zh', '--limit', '1'))
        self.assertEqual(datasets['subjective'][0]['id'], '21')
        self.assertFalse(any(s['name'].startswith('expert-stat-') and s['name'].endswith('-en') for s in steps))
        self.assertTrue(any(s['name'] == 'expert-stat-1-zh' for s in steps))

    def test_single_language_statistics_commands_execute_end_to_end(self):
        import subprocess
        from evaluation_toolkit.subjective_eval.runtime import atomic_save
        from evaluation_toolkit.subjective_eval.ExpertCriteria.score_by_criteria import parse_criteria_dimensions
        from evaluation_toolkit.subjective_eval.scoring.provenance import question_fingerprint, report_fingerprint
        options = self.options('--track', 'subjective', '--language', 'en', '--limit', '1', '--expert_rounds', '1')
        _, datasets, steps = cli.build_plan(options)
        question = datasets['subjective'][0]
        report = {'model': 'gpt-4o', 'response': 'Synthetic CLI statistics fixture'}
        provenance = {'question_fingerprint': question_fingerprint(question),
                      'report_fingerprint': report_fingerprint(question['id'], report)}
        atomic_save(self.base / 'run/inputs/subjective.json', [question])
        caps = parse_criteria_dimensions(question['expert_evaluation_criteria'])
        atomic_save(self.base / 'run/subjective/expert_round_1/judge_gemini-3.1-pro/scores_gpt-4o.jsonl',
                    [{'task_id': question['id'], 'model': 'gpt-4o', 'total_score': 80,
                      'dimensions_breakdown': [{'dimension_id': key, 'score': cap * .8} for key, cap in caps.items()],
                      **provenance}], jsonl=True)
        atomic_save(self.base / 'run/subjective/fact/validate.jsonl',
                    [{'id': question['id'], 'model': 'gpt-4o', 'citations_deduped': {}, **provenance}], jsonl=True)
        for step in steps:
            if step['name'].startswith('expert-stat-') or step['name'] in ('fact-stat', 'subjective-final'):
                result = subprocess.run(step['command'], cwd=cli.ROOT, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with (self.base / 'run/subjective/final_leaderboard.csv').open(encoding='utf-8-sig') as stream:
            row = next(csv.DictReader(stream))
            self.assertEqual(float(row['S_final']), 64)
            self.assertEqual(int(row['Total Tasks']), 1)
        self.assertTrue((self.base / 'run/subjective/final_leaderboard_en.csv').exists())
        self.assertFalse((self.base / 'run/subjective/final_leaderboard_zh.csv').exists())

    def test_dry_run_writes_nothing_and_needs_no_credentials(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': ''}):
            self.assertEqual(cli.run_pipeline(self.options('--dry_run')), 0)
        self.assertFalse((self.base / 'run').exists())

    def test_subjective_pipeline_requires_explicit_reference_date(self):
        options = self.options()
        options.as_of = None
        with self.assertRaisesRegex(ValueError, '--as_of'):
            cli.build_plan(options)
        options.track = 'objective'
        self.assertEqual(len(cli.build_plan(options)[2]), 3)

    def test_credentials_check_rejects_placeholder_without_displaying_it(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'sk-...'}):
            with self.assertRaisesRegex(ValueError, 'OPENROUTER_API_KEY') as error:
                cli.check_keys('gpt-4o')
            self.assertNotIn('sk-...', str(error.exception))
        self.assertEqual(cli.required_keys('o3-deep-research'), ['OPENAI_API_KEY', 'OPENROUTER_API_KEY'])

    def test_live_runner_stops_at_failed_stage_and_resumes_same_inputs(self):
        options = self.options('--track', 'objective')
        with patch.object(cli, 'check_keys'), patch.object(cli.subprocess, 'run') as process:
            process.side_effect = [SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)]
            self.assertEqual(cli.run_pipeline(options), 1)
            self.assertEqual(process.call_count, 2)
            self.assertEqual(json.loads((self.base / "run/run_manifest.json").read_text())["status"], "failed")
            self.assertEqual(json.loads((self.base / 'run/inputs/objective.json').read_text())[1]['id'], '41')
            process.reset_mock(side_effect=True)
            process.return_value = SimpleNamespace(returncode=0)
            self.assertEqual(cli.run_pipeline(options), 0)
            self.assertEqual(process.call_count, 3)
            self.assertEqual(json.loads((self.base / "run/run_manifest.json").read_text())["status"], "complete")
            options.limit = 3
            with self.assertRaisesRegex(ValueError, 'configuration/code/environment changed'):
                cli.run_pipeline(options)
            self.assertEqual(process.call_count, 3)

    def test_edited_subset_and_nonempty_unmanaged_directory_are_rejected(self):
        options = self.options('--track', 'objective')
        with patch.object(cli, 'check_keys'), patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as process:
            cli.run_pipeline(options)
            subset = self.base / 'run/inputs/objective.json'
            subset.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'subset was modified'):
                cli.run_pipeline(options)
            self.assertEqual(process.call_count, 3)
            options.run_dir = str(self.base / 'unmanaged')
            Path(options.run_dir).mkdir()
            (Path(options.run_dir) / 'user-data').write_text('keep')
            with self.assertRaisesRegex(ValueError, 'must be empty'):
                cli.run_pipeline(options)
            self.assertEqual((Path(options.run_dir) / 'user-data').read_text(), 'keep')

    def test_demo_runs_real_aggregation_with_known_scores_and_no_network(self):
        output = self.base / 'demo'
        self.assertEqual(run_demo(output), 0)
        with (output / 'subjective/final_leaderboard.csv').open(encoding='utf-8-sig') as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(float(row['S_final']), 74)
        self.assertEqual(float(row['S_expert']), 80)
        self.assertEqual(int(row['Expert Rounds']), 3)
        self.assertEqual(row['Evaluation As Of (UTC)'], '2026-10-16T00:00:00+00:00')
        with (output / 'objective/evaluation_results.csv').open() as stream:
            self.assertEqual(float(next(csv.DictReader(stream))['Accuracy (%)']), 50)
        # Missing one requested scoring round must not silently become a two-round average.
        directories = [str(output / 'subjective' / f'expert_round_{i}') for i in (1, 2, 3)]
        fact = str(output / 'subjective/fact_result.csv')
        (Path(directories[2]) / 'leaderboard.csv').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing ExpertCriteria round'):
            run_all_versions(directories, fact, str(output / 'missing-round'), .8, .1, .1)
        first = str(Path(directories[0]) / 'leaderboard.csv')
        with self.assertRaisesRegex(ValueError, 'distinct CSV'):
            compute_final_score([first, first], fact, str(output / 'duplicate.csv'), .8, .1, .1)
        with self.assertRaisesRegex(ValueError, 'must be empty'):
            run_demo(output)


if __name__ == '__main__':
    unittest.main()
