# ICBCBench Evaluation Toolkit

This repository contains the evaluation toolkit for the ICBCBench project, including objective question evaluation and subjective research report evaluation.

## Directory Structure

```
evaluation_toolkit/
├── objective_eval/                 # Objective question evaluation
│   ├── predict.py                      # Generate model predictions
│   ├── judge.py                        # Judge model answers
│   └── metrics.py                      # Compute Accuracy / Calibration Error
├── subjective_eval/                # Subjective research report evaluation
│   ├── write.py                        # Generate research reports with models
│   ├── ExpertCriteria/                 # Expert-criteria based scoring
│   │   ├── score_by_criteria.py        # Score reports by criteria
│   │   └── stat.py                     # Aggregate jsonl scores to leaderboard CSV
│   ├── FACT/                           # Citation extraction, deduplication, scraping, validation, and scoring
│   │   ├── extract.py
│   │   ├── deduplicate.py
│   │   ├── scrape.py
│   │   ├── validate.py
│   │   ├── stat.py
│   │   ├── authority.py
│   │   ├── time_decay.py
│   │   └── utils.py
│   └── scoring/                        # Final score aggregation
│       └── compute_final_score.py      # Merge ExpertCriteria + FACT into final leaderboard
├── DR_clients/                     # Deep Research model clients
├── model_clients.py                # API client collection
└── utils.py                        # JSON / PDF utilities
```

## Installation

```bash
pip install -r requirements.txt
```

> Note: `subjective_eval/FACT/scrape.py` depends on `curl_cffi`.

## Environment Variables

API keys are loaded from a `.env` file in the project root. Copy `.env.example` to `.env` and fill in the required keys:

```bash
cp .env.example .env
```

Current clients use `OPENROUTER_API_KEY` as the default. Refer to `model_clients.py` and `.env.example` for all supported keys.

## Data Paths

Default data files are expected under the project root:

- Objective questions: `data/objective_questions_public_80.json`
- Subjective questions: `data/subjective_questions_public_40.json`
- Model reports: `eval_result/subjective_eval/dr_reports/reports_<model>.json`

Generated evaluation results are written to `eval_result/`:

- Objective predictions: `eval_result/objective_eval/prediction_results/<model>.json`
- Objective judged results: `eval_result/objective_eval/judged_results/judge_<judge>/judged_<model>.json`
- Objective metrics: `eval_result/objective_eval/evaluation_results.csv`
- Subjective scores: `eval_result/subjective_eval/scores/judge_<judge>/scores_<model>.jsonl`
- Subjective leaderboard: `eval_result/subjective_eval/scores/judge_<judge>/leaderboard.csv`
- FACT results: `eval_result/subjective_eval/fact_result.csv`
- Final leaderboard: `eval_result/subjective_eval/final_leaderboard.csv`

## Objective Evaluation Pipeline

### 1. Generate Predictions

```bash
python evaluation_toolkit/objective_eval/predict.py \
    --local_dataset data/objective_questions_public_80.json \
    --model gpt-4o \
    --num_workers 2
```

Outputs: `eval_result/objective_eval/prediction_results/gpt-4o.json`

### 2. Judge Predictions

```bash
python evaluation_toolkit/objective_eval/judge.py \
    --local_dataset data/objective_questions_public_80.json \
    --predictions eval_result/objective_eval/prediction_results/gpt-4o.json \
    --judge gemini-1.5-pro \
    --num_workers 5
```

Outputs:
- `eval_result/objective_eval/judged_results/judge_<judge>/judged_gpt-4o.json`
- `eval_result/objective_eval/evaluation_results.csv`

### 3. Compute Metrics

```bash
python evaluation_toolkit/objective_eval/metrics.py
```

Reads judged results from `eval_result/objective_eval/judged_results/judge_<judge>/` and writes metrics to `eval_result/objective_eval/`.

## Subjective Evaluation Pipeline

### 1. Generate Reports

```bash
python evaluation_toolkit/subjective_eval/write.py \
    --local_dataset data/subjective_questions_public_40.json \
    --model gpt-4o \
    --num_workers 2
```

Outputs: `eval_result/subjective_eval/dr_reports/reports_gpt-4o.json`

### 2. ExpertCriteria Scoring

```bash
python evaluation_toolkit/subjective_eval/ExpertCriteria/score_by_criteria.py \
    --judge gemini-1.5-pro \
    --report_model gpt-4o
```

Outputs: `eval_result/subjective_eval/scores/judge_<judge>/scores_gpt-4o.jsonl`

### 3. Aggregate ExpertCriteria Scores

```bash
python evaluation_toolkit/subjective_eval/ExpertCriteria/stat.py \
    --input_dir eval_result/subjective_eval/scores/judge_<judge> \
    --query_file data/subjective_questions_public_40.json \
    --output_csv eval_result/subjective_eval/scores/judge_<judge>/leaderboard.csv
```

### 4. FACT Evaluation

Run the FACT pipeline (`extract.py`, `deduplicate.py`, `scrape.py`, `validate.py`) to produce a validated citation file, then compute FACT metrics:

```bash
python evaluation_toolkit/subjective_eval/FACT/stat.py \
    --model_name gpt-4o \
    --input_path <path_to_validated_data.jsonl> \
    --output_csv eval_result/subjective_eval/fact_result.csv \
    --query_file data/subjective_questions_public_40.json
```

### 5. Compute Final Score

```bash
python evaluation_toolkit/subjective_eval/scoring/compute_final_score.py \
    --expert_dirs eval_result/subjective_eval/scores/judge_<judge> \
    --fact_csv eval_result/subjective_eval/fact_result.csv \
    --output_dir eval_result/subjective_eval \
    --alpha 0.8 --beta 0.1 --gamma 0.1
```

Outputs: `eval_result/subjective_eval/final_leaderboard.csv`

The final score formula is:

```
S_final = alpha * S_expert + beta * S_citation + gamma * S_source
```

## Notes

- All scripts can be run directly or as modules (`python -m evaluation_toolkit.objective_eval.predict ...`).
- Use `--max_samples` to limit the number of samples during testing.
- Set `--num_workers` / `--max_workers` according to your API rate limit.
- The FACT pipeline expects `citations_deduped` in the input data with fields such as `facts`, `validate_res`, `publish_time`, and `true_url`.
