"""Synthetic inputs exercising real score validation and leaderboard aggregation.

No generation, judge, scraping or validation service is called. Every simulated
model/score is labelled synthetic; these are not benchmark performance results.
"""
import json
from pathlib import Path
from types import SimpleNamespace

from evaluation_toolkit.objective_eval import metrics
from evaluation_toolkit.subjective_eval.runtime import atomic_save
from evaluation_toolkit.subjective_eval.ExpertCriteria import score_by_criteria as expert
from evaluation_toolkit.subjective_eval.ExpertCriteria import stat as expert_stat
from evaluation_toolkit.subjective_eval.FACT import stat as fact_stat
from evaluation_toolkit.subjective_eval.scoring.compute_final_score import run_all_versions
from evaluation_toolkit.subjective_eval.scoring.provenance import question_fingerprint, report_fingerprint

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'synthetic-offline-demo'
AS_OF = '2026-10-16T00:00:00+00:00'


def run_demo(base):
    base = Path(base)
    if base.exists() and any(base.iterdir()):
        raise ValueError('Demo output directory must be empty; choose another --output_dir')
    base.mkdir(parents=True, exist_ok=True)
    atomic_save(base / 'demo_manifest.json', {
        'synthetic': True, 'model': MODEL, 'as_of': AS_OF,
        'description': 'Fabricated reports, judgments and citation contents; offline scoring/aggregation only.',
        'expected': {'objective_accuracy': 50, 'S_expert': 80, 'S_citation': 50,
                     'S_source': 50, 'S_final': 74}})
    objective = json.loads((ROOT / 'data/objective_questions_public_80.json').read_text(encoding='utf-8'))
    objective = [next(q for q in objective if q['language'] == lang) for lang in ('en', 'zh')]
    atomic_save(base / 'objective/questions.json', objective)
    judgments = {}
    for index, question in enumerate(objective):
        record = {'model': MODEL, 'response': 'Synthetic answer; not model output.',
                  'judge_model': MODEL, 'judge_result': {'correct': 'yes' if index == 0 else 'no', 'confidence': 50}}
        record['judge_fingerprint'] = metrics.judge_fingerprint(question, record, MODEL)
        judgments[question['id']] = record
    atomic_save(base / 'objective/judged/judged_synthetic-offline-demo.json', judgments)
    status = metrics.main(SimpleNamespace(query_file=base / 'objective/questions.json', judge=MODEL,
                          judged_dir=base / 'objective/judged', output_dir=base / 'objective'))
    questions = json.loads((ROOT / 'data/subjective_questions_public_40.json').read_text(encoding='utf-8'))
    # Include the unchanged public rubric 36 so the normalization is inspectable.
    questions = [questions[0], next(q for q in questions if q['id'] == '36')]
    query = base / 'subjective/questions.json'
    atomic_save(query, questions)
    reports = {q['id']: {'model': MODEL, 'response': 'Synthetic report for task ' + q['id']} for q in questions}
    atomic_save(base / 'subjective/reports.json', reports)
    directories = []
    for round_number, fraction in enumerate((.7, .8, .9), 1):
        directory = base / 'subjective' / f'expert_round_{round_number}'
        directories.append(str(directory))
        scores = []
        for question in questions:
            criteria = question['expert_evaluation_criteria']
            dimensions = [{'dimension_id': key, 'score': round(cap * fraction, 4)}
                          for key, cap in expert.parse_criteria_dimensions(criteria).items()]
            scores.append({'task_id': question['id'], 'model': MODEL, 'judge': MODEL,
                           'total_score': expert.validate_dimension_scores(dimensions, criteria),
                           'raw_total_score': sum(d['score'] for d in dimensions),
                           'scoring_adjustments': expert.scoring_adjustments(criteria),
                           'dimensions_breakdown': dimensions,
                           'question_fingerprint': question_fingerprint(question),
                           'report_fingerprint': report_fingerprint(question['id'], reports[question['id']])})
        atomic_save(directory / 'scores_synthetic.jsonl', scores, jsonl=True)
        for language in (None, 'en', 'zh'):
            status |= expert_stat.stat(str(directory), str(query), str(directory / 'leaderboard.csv'), language)
    citations = []
    for index, question in enumerate(questions):
        # A fabricated .gov source exercises authority=1. It is never fetched.
        sources = {'https://example.gov/synthetic-offline-fixture': {
            'facts': ['Synthetic fact'], 'url_content': 'Fabricated source text, not retrieved.',
            'publish_time': '2026-10-16', 'validate_res': [{'idx': 0, 'result': 'supported'}]}} if index == 0 else {}
        citations.append({'id': question['id'], 'model': MODEL, 'citations_deduped': sources,
                          'question_fingerprint': question_fingerprint(question),
                          'report_fingerprint': report_fingerprint(question['id'], reports[question['id']])})
    validated = base / 'subjective/validated_synthetic.jsonl'
    atomic_save(validated, citations, jsonl=True)
    options = SimpleNamespace(input_path=str(validated), model_name=MODEL, language_filter=None, as_of=AS_OF)
    for language in (None, 'en', 'zh'):
        filename = f'fact_result_{language}.csv' if language else 'fact_result.csv'
        status |= fact_stat.run_stat(options, {q['id']: q['language'] for q in questions},
                                    str(base / 'subjective' / filename), language, questions)
    if status:
        raise ValueError('Offline demonstration failed score/provenance validation')
    run_all_versions(directories, str(base / 'subjective/fact_result.csv'), str(base / 'subjective'), .8, .1, .1)
    print(f'SYNTHETIC OFFLINE DEMO: objective accuracy=50%; subjective final=74.00. Outputs: {base}')
    return 0
