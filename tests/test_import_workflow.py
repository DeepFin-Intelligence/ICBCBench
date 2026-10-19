"""Imported outputs run through the real scoring pipeline without generation."""
import contextlib
import csv
import importlib
import io
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import openai
from openai import OpenAI

from evaluation_toolkit import cli, model_clients
from evaluation_toolkit.subjective_eval.ExpertCriteria.score_by_criteria import parse_criteria_dimensions


class ImportWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.output = io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.output))
        self.stack.enter_context(contextlib.redirect_stderr(self.output))
        self.stack.enter_context(patch.dict(os.environ, {'PYTHON_DOTENV_DISABLED': '1'}))
        self.stack.enter_context(patch.object(cli.platform, 'platform', return_value='test-platform'))
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')))
        self.sources = {}
        for track in cli.DATASETS:
            path = self.base / f'{track}.json'
            path.write_text(json.dumps([
                {'id': q['id'], 'response': 'Synthetic response without citations. 合成测试文本。'}
                for q in cli.select_questions(track, limit=2)], ensure_ascii=False), encoding='utf-8')
            self.sources[track] = str(path)

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def options(self, *extra, tracks=('objective', 'subjective')):
        argv = ['score', '--model', 'my-lab/model-v1', '--run_dir', str(self.base / 'run'),
                '--as_of', '2026-10-19', '--limit', '2', '--workers', '1']
        for track in tracks:
            argv += ['--' + track + '_results', self.sources[track]]
        return cli.parser().parse_args(argv + list(extra))

    def test_plan_skips_generation_freezes_original_ids_and_keeps_three_rounds(self):
        config, datasets, steps = cli.build_plan(self.options())
        self.assertEqual(config['tracks'], ['objective', 'subjective'])
        self.assertEqual(config['task_ids'], {'objective': ['1', '41'], 'subjective': ['1', '21']})
        self.assertEqual(len([s for s in steps if s['name'].startswith('expert-score-')]), 3)
        self.assertFalse(any(s['name'] in ('objective-predict', 'subjective-write') for s in steps))
        self.assertEqual(config['evaluation_scope']['objective']['scope'], 'subset')
        self.assertEqual(config['imports']['subjective']['selected_count'], 2)
        self.assertEqual(len(datasets['subjective']), 2)
        self.assertFalse((self.base / 'run').exists())

    def test_imports_infer_single_track_and_objective_needs_no_date(self):
        options = self.options(tracks=('objective',))
        options.as_of = None
        self.assertEqual(cli.build_plan(options)[0]['tracks'], ['objective'])
        options.track = 'both'
        with self.assertRaisesRegex(ValueError, 'must match'):
            cli.build_plan(options)
        options.objective_results = None
        with self.assertRaisesRegex(ValueError, 'requires'):
            cli.build_plan(options)

    def test_dry_run_validates_inputs_without_keys_network_or_files(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': ''}):
            self.assertEqual(cli.run_pipeline(self.options('--dry_run')), 0)
        self.assertFalse((self.base / 'run').exists())
        Path(self.sources['objective']).write_text('[]', encoding='utf-8')
        with self.assertRaises(ValueError):
            cli.run_pipeline(self.options('--dry_run'))
        self.assertFalse((self.base / 'run').exists())

    def test_arbitrary_subset_requires_explicit_opt_in(self):
        Path(self.sources['subjective']).write_text('[{"id":"36","response":"A report"}]', encoding='utf-8')
        options = self.options(tracks=('subjective',))
        options.limit = None
        with self.assertRaises(ValueError):
            cli.build_plan(options)
        options.allow_subset = True
        config, datasets, steps = cli.build_plan(options)
        self.assertEqual(config['task_ids'], {'subjective': ['36']})
        self.assertEqual(len(config['imports']['subjective']['omitted_ids']), 39)
        self.assertEqual(config['evaluation_scope']['subjective']['selected_tasks'], 1)
        self.assertFalse(any(s['name'].endswith('-en') for s in steps))

    def test_import_only_checks_judge_credentials_even_for_provider_named_model(self):
        self.assertEqual(cli.required_keys('o3-deep-research', score_only=True), ['OPENROUTER_API_KEY'])
        self.assertEqual(cli.required_keys('o3-deep-research', {'model_backend': 'openai'}), ['OPENROUTER_API_KEY'])

    def test_custom_settings_only_reach_generation_stages(self):
        options = cli.parser().parse_args([
            'run', '--model', 'local-model', '--run_dir', str(self.base / 'custom'),
            '--model_backend', 'openai', '--model_base_url', 'http://localhost:8000/v1',
            '--attachment_mode', 'url', '--limit', '2', '--as_of', '2026-10-19'])
        config, _, steps = cli.build_plan(options)
        self.assertEqual(config['generation']['model_base_url'], 'http://localhost:8000/v1')
        for step in steps:
            if step['name'] in ('objective-predict', 'subjective-write'):
                self.assertIn('--model_base_url', step['command'])
                self.assertIn('url', step['command'])
            else:
                self.assertNotIn('--model_base_url', step['command'])
                self.assertNotIn('--agent', step['command'])

    def test_changed_import_or_frozen_results_cannot_reuse_previous_run(self):
        options = self.options(tracks=('objective',))
        with patch.object(cli, 'check_keys'), patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
            self.assertEqual(cli.run_pipeline(options), 0)
            source = Path(self.sources['objective'])
            original = source.read_bytes()
            source.write_bytes(original.replace(b'Synthetic response', b'Different response'))
            with self.assertRaisesRegex(ValueError, 'configuration/code/environment changed'):
                cli.run_pipeline(options)
            source.write_bytes(original)
            frozen = self.base / 'run/inputs/objective_results.json'
            frozen.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Frozen imported results'):
                cli.run_pipeline(options)

    def test_interrupted_import_initialization_is_resumable(self):
        from evaluation_toolkit.subjective_eval import runtime
        original_save = runtime.atomic_save

        def interrupt_import(path, data, **kwargs):
            if Path(path).name == 'objective_results.json':
                raise KeyboardInterrupt()
            return original_save(path, data, **kwargs)

        options = self.options(tracks=('objective',))
        with patch.object(cli, 'check_keys'), patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as stages:
            with patch.object(runtime, 'atomic_save', side_effect=interrupt_import):
                with self.assertRaises(KeyboardInterrupt):
                    cli.run_pipeline(options)
            stages.assert_not_called()
            manifest = self.base / 'run/run_manifest.json'
            self.assertEqual(json.loads(manifest.read_text(encoding='utf-8'))['status'], 'interrupted')
            self.assertEqual(cli.run_pipeline(options), 0)
            self.assertEqual(json.loads(manifest.read_text(encoding='utf-8'))['status'], 'complete')

    def test_both_tracks_score_and_resume_with_real_sdk_and_real_cli_stages(self):
        http = importlib.import_module('httpx2' if int(openai.__version__.split('.')[0]) >= 3 else 'httpx')
        calls = []
        questions = cli.select_questions('subjective', limit=2)

        def handle(request):
            payload = json.loads(request.content)
            calls.append(payload)
            model = payload['model']
            if model == 'openai/gpt-5.4':
                content = json.dumps({'extracted_final_answer': 'synthetic', 'reasoning': 'Offline fixture',
                                      'correct': 'yes' if len(calls) == 1 else 'no', 'confidence': 50})
            elif model == 'google/gemini-3.1-pro-preview':
                prompt = payload['messages'][0]['content']
                question = next(q for q in questions if q['expert_evaluation_criteria'] in prompt)
                caps = parse_criteria_dimensions(question['expert_evaluation_criteria'])
                content = json.dumps([{'dimension_id': key, 'score': cap * .8} for key, cap in caps.items()])
            elif model == 'google/gemini-3-flash-preview':
                content = '[]'
            else:
                raise AssertionError(f'Tested model must never be called: {model}')
            return http.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 1,
                'model': model, 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': content}}]})

        def execute_stage(command, **kwargs):
            # Execute the exact stage argv in-process, retaining the real SDK's
            # offline HTTP transport. Production launches these as subprocesses.
            with patch.object(sys, 'argv', [command[2], *command[3:]]), \
                    contextlib.redirect_stdout(kwargs['stdout']), contextlib.redirect_stderr(kwargs['stdout']):
                try:
                    runpy.run_module(command[2], run_name='__main__')
                except SystemExit as exc:
                    return SimpleNamespace(returncode=exc.code or 0)
            return SimpleNamespace(returncode=0)

        sdk = OpenAI(api_key='offline-fixture', base_url='https://judge.invalid/v1', max_retries=0,
                     http_client=http.Client(transport=http.MockTransport(handle)))
        self.addCleanup(sdk.close)
        with patch.object(model_clients, 'get_openrouter_client', return_value=sdk), \
                patch.object(cli.subprocess, 'run', side_effect=execute_stage), patch.object(cli, 'check_keys'):
            self.assertEqual(cli.run_pipeline(self.options()), 0, self.output.getvalue())
            self.assertEqual(len(calls), 10)  # 2 objective + 3*2 expert + 2 citation extraction
            self.assertEqual(cli.run_pipeline(self.options()), 0, self.output.getvalue())
            self.assertEqual(len(calls), 10, 'Resume must reuse validated judgments')
        with (self.base / 'run/objective/metrics/evaluation_results.csv').open(encoding='utf-8-sig') as handle:
            objective = next(csv.DictReader(handle))
            self.assertEqual(float(objective['Accuracy (%)']), 50)
            self.assertEqual(objective['Model'], 'my-lab/model-v1')
        with (self.base / 'run/subjective/final_leaderboard.csv').open(encoding='utf-8-sig') as handle:
            row = next(csv.DictReader(handle))
            self.assertEqual(float(row['S_final']), 64)
            self.assertEqual(int(row['Expert Rounds']), 3)
            self.assertEqual(int(row['Total Tasks']), 2)
        manifest = json.loads((self.base / 'run/run_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['config']['command'], 'score')
        self.assertEqual(set(manifest['config']['imports']), {'objective', 'subjective'})


if __name__ == '__main__':
    unittest.main()
