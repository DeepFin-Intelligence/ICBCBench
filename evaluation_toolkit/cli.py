"""Reproducible local-data workflows. No API calls during check or dry-run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'objective': 'objective_questions_public_80.json',
            'subjective': 'subjective_questions_public_40.json'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_release():
    from evaluation_toolkit.subjective_eval.runtime import validate_questions
    from evaluation_toolkit.subjective_eval.ExpertCriteria.score_by_criteria import (
        parse_criteria_dimensions, scoring_adjustments)
    manifest = json.loads((ROOT / 'data/huggingface_version.json').read_text())
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if path.stat().st_size != entry['size'] or sha256(path) != entry['sha256']:
            raise ValueError(f'Release file differs from manifest: {entry["path"]}')
    for track, filename in DATASETS.items():
        questions = validate_questions(json.loads((ROOT / 'data' / filename).read_text()))
        expected = 80 if track == 'objective' else 40
        if len(questions) != expected or any(sum(q['language'] == lang for q in questions) != expected // 2 for lang in ('en', 'zh')):
            raise ValueError(f'Unexpected public {track} task counts')
        if track == 'subjective':
            for question in questions:
                criteria = question['expert_evaluation_criteria']
                if not parse_criteria_dimensions(criteria):
                    raise ValueError(f'Missing structured rubric for task {question["id"]}')
                for adjustment in scoring_adjustments(criteria):
                    print(f'Task {question["id"]}: {adjustment["reason"]}')
        print(f'{track}: {expected} tasks, equal English/Chinese counts; release hashes verified')


def dependency_versions():
    versions = {}
    for line in (ROOT / 'requirements-lock.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError(f'Missing dependency {name}; install requirements-lock.txt') from None
        if versions[name] != expected:
            print(f'NOTE: {name}={versions[name]}, tested lock={expected}')
    return versions


def required_keys(model):
    providers = {'gemini-deep-research': 'GEMINI_API_KEY',
                 'qwen-deep-research': 'DASHSCOPE_API_KEY',
                 'tavily-research': 'TAVILY_API_KEY',
                 'o3-deep-research': 'OPENAI_API_KEY',
                 'o4-mini-deep-research': 'OPENAI_API_KEY'}
    # All pipeline judges use OpenRouter, even when generation uses another provider.
    return sorted({'OPENROUTER_API_KEY', providers.get(model.lower(), 'OPENROUTER_API_KEY')})


def check_keys(model):
    from evaluation_toolkit.model_clients import load_project_env
    load_project_env()
    missing = []
    for name in required_keys(model):
        value = os.getenv(name, '').strip()
        if not value or '...' in value or value.lower() in ('your-api-key', 'changeme'):
            missing.append(name)
    if missing:
        raise ValueError('Set real credentials in the project .env or environment: ' + ', '.join(missing))
    print('Required key names are configured (not tested against providers).')


def select_questions(track, language=None, limit=None):
    questions = json.loads((ROOT / 'data' / DATASETS[track]).read_text())
    if language:
        questions = [q for q in questions if q['language'] == language]
    if limit:
        # Alternate languages for bilingual smoke runs; preserve original public IDs.
        if not language:
            groups = [[q for q in questions if q['language'] == lang] for lang in ('en', 'zh')]
            questions = [q for pair in zip(*groups) for q in pair]
        questions = questions[:limit]
    return questions


def build_plan(options):
    from evaluation_toolkit.subjective_eval.FACT.time_decay import reference_time
    tracks = list(DATASETS) if options.track == 'both' else [options.track]
    if 'subjective' in tracks and not options.as_of:
        raise ValueError('--as_of YYYY-MM-DD is required for subjective evaluation')
    base = Path(options.run_dir).resolve()
    datasets = {track: select_questions(track, options.language, options.limit) for track in tracks}
    model_file = options.model.rsplit('/', 1)[-1]
    if not model_file or model_file in ('.', '..') or '\\' in options.model:
        raise ValueError('Invalid model ID')
    steps = []

    def stage(label, module, **kwargs):
        command = [sys.executable, '-m', 'evaluation_toolkit.' + module]
        for name, value in kwargs.items():
            if value is not None:
                command += ['--' + name] + [str(item) for item in (value if isinstance(value, list) else [value])]
        steps.append({'name': label, 'command': command})

    if 'objective' in tracks:
        directory = base / 'objective'
        query = base / 'inputs/objective.json'
        stage('objective-predict', 'objective_eval.predict', local_dataset=query, model=options.model,
              num_workers=options.workers, output_dir=directory / 'predictions')
        stage('objective-judge', 'objective_eval.judge', local_dataset=query,
              predictions=directory / 'predictions' / f'{model_file}.json', judge=options.objective_judge,
              num_workers=options.workers, output_dir=directory / 'judged', metrics_csv=directory / 'judge_metrics.csv')
        stage('objective-metrics', 'objective_eval.metrics', query_file=query, judged_dir=directory / 'judged',
              judge=options.objective_judge, output_dir=directory / 'metrics')
    if 'subjective' in tracks:
        directory = base / 'subjective'
        query = base / 'inputs/subjective.json'
        languages = sorted({q['language'] for q in datasets['subjective']})
        stage('subjective-write', 'subjective_eval.write', local_dataset=query, model=options.model,
              num_workers=options.workers, output_dir=directory / 'reports')
        expert_dirs = []
        for round_number in range(1, options.expert_rounds + 1):
            round_dir = directory / f'expert_round_{round_number}'
            scores_dir = round_dir / f'judge_{options.expert_judge.rsplit("/", 1)[-1]}'
            expert_dirs.append(scores_dir)
            stage(f'expert-score-{round_number}', 'subjective_eval.ExpertCriteria.score_by_criteria',
                  judge=options.expert_judge, report_model=options.model, question_path=query,
                  report_dir=directory / 'reports', output_dir=round_dir, max_workers=options.workers)
            for language in [None] + languages:
                stage(f'expert-stat-{round_number}-{language or "all"}', 'subjective_eval.ExpertCriteria.stat',
                      input_dir=scores_dir, query_file=query, output_csv=scores_dir / 'leaderboard.csv',
                      language_filter=language)
        raw = directory / 'reports' / f'reports_{model_file}.json'
        for name in ('extract', 'deduplicate', 'scrape', 'validate'):
            output = directory / 'fact' / f'{name}.jsonl'
            kwargs = dict(raw_data_path=raw, output_path=output, n_total_process=options.workers)
            if name != 'scrape':
                kwargs.update(query_data_path=query, judge=options.fact_judge)
            stage('fact-' + name, 'subjective_eval.FACT.' + name, **kwargs)
            raw = output
        stage('fact-stat', 'subjective_eval.FACT.stat', model_name=options.model, input_path=raw,
              query_file=query, output_csv=directory / 'fact_result.csv', as_of=options.as_of)
        stage('subjective-final', 'subjective_eval.scoring.compute_final_score', expert_dirs=expert_dirs,
              fact_csv=directory / 'fact_result.csv', output_dir=directory)
    config = {name: getattr(options, name) for name in ('track', 'model', 'language', 'limit',
              'objective_judge', 'expert_judge', 'fact_judge', 'expert_rounds', 'workers')}
    config['as_of'] = reference_time(options.as_of).isoformat() if options.as_of else None
    config['task_ids'] = {track: [q['id'] for q in items] for track, items in datasets.items()}
    config['dataset_sha256'] = {track: sha256(ROOT / 'data' / DATASETS[track]) for track in tracks}
    # Include implementation, prompts, authority rules and dependency lock in resume identity.
    files = list((ROOT / 'evaluation_toolkit').rglob('*.py')) + list((ROOT / 'evaluation_toolkit').rglob('*.json'))
    files += [ROOT / 'requirements-lock.txt']
    config['implementation_sha256'] = hashlib.sha256(json.dumps(
        {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}, sort_keys=True).encode()).hexdigest()
    return config, datasets, steps


def run_pipeline(options):
    config, datasets, steps = build_plan(options)
    if options.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        for step in steps:
            print(step['name'] + ': ' + shlex.join(step['command']))
        return 0
    check_release()
    versions = dependency_versions()
    check_keys(options.model)
    base = Path(options.run_dir).resolve()
    manifest_path = base / 'run_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['config'] != config or old['dependencies'] != versions or old['python'] != platform.python_version():
            raise ValueError('Run configuration/code/environment changed; choose a new --run_dir')
    elif base.exists() and any(base.iterdir()):
        raise ValueError('Run directory must be empty or contain a matching run_manifest.json')
    base.mkdir(parents=True, exist_ok=True)
    from evaluation_toolkit.subjective_eval.runtime import atomic_save
    for track, questions in datasets.items():
        path = base / 'inputs' / f'{track}.json'
        if path.exists() and json.loads(path.read_text()) != questions:
            raise ValueError(f'Frozen task subset was modified: {path}')
        atomic_save(str(path), questions)
    manifest = {'config': config, 'dependencies': versions, 'python': platform.python_version(),
                'platform': platform.platform(), 'steps': steps, 'status': 'running', 'completed_steps': []}
    atomic_save(str(manifest_path), manifest)
    (base / 'logs').mkdir(exist_ok=True)
    try:
        for index, step in enumerate(steps, 1):
            manifest['current_step'] = step['name']
            atomic_save(str(manifest_path), manifest)
            print(f'[{index}/{len(steps)}] {step["name"]}', flush=True)
            log_path = base / 'logs' / f'{step["name"]}.log'
            with log_path.open('a', encoding='utf-8') as log:
                result = subprocess.run(step['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                manifest.update(status='failed', returncode=result.returncode)
                print(f'Stage failed; see {log_path}. Rerun the same command to resume.', file=sys.stderr)
                return result.returncode if result.returncode > 0 else 1
            manifest['completed_steps'].append(step['name'])
        manifest.update(status='complete', current_step=None)
    except KeyboardInterrupt:
        manifest['status'] = 'interrupted'
        raise
    except Exception:
        manifest['status'] = 'failed'
        raise
    finally:
        atomic_save(str(manifest_path), manifest)
    print(f'Complete. Results and provenance: {base}')
    return 0


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', help='Offline release/dependency checks; optionally check key presence')
    check.add_argument('--model', help='Also check required key names for a full run, without API requests')
    run = commands.add_parser('run', help='Execute an isolated evaluation; live stages incur provider charges')
    run.add_argument('--model', required=True)
    run.add_argument('--run_dir', required=True)
    run.add_argument('--track', choices=['objective', 'subjective', 'both'], default='both')
    run.add_argument('--language', choices=['en', 'zh'])
    run.add_argument('--limit', type=positive, help='Maximum tasks PER TRACK; bilingual subsets alternate en/zh')
    run.add_argument('--workers', type=positive, default=2)
    run.add_argument('--expert_rounds', type=positive, default=3)
    run.add_argument('--objective_judge', default='gpt-5.4')
    run.add_argument('--expert_judge', default='gemini-3.1-pro')
    run.add_argument('--fact_judge', default='google/gemini-3-flash-preview')
    run.add_argument('--as_of', help='Required for subjective runs: ISO reference date for source timeliness')
    run.add_argument('--dry_run', action='store_true', help='Print planned commands; no files, keys or network needed')
    demo = commands.add_parser('demo', help='Offline synthetic scoring/aggregation example; no model performance claims')
    demo.add_argument('--output_dir', default='eval_result/offline_demo')
    return result


def main(argv=None):
    options = parser().parse_args(argv)
    try:
        if options.command == 'check':
            dependency_versions()
            check_release()
            if options.model:
                check_keys(options.model)
            print('Offline checks passed. No provider or source URLs were contacted.')
            return 0
        if options.command == 'demo':
            from evaluation_toolkit.offline_demo import run_demo
            return run_demo(Path(options.output_dir).resolve())
        return run_pipeline(options)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Interrupted. Successful stage records are retained for resume.', file=sys.stderr)
        return 130
