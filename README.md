<h1 align="center">ICBCBench: An Industry Consortium Benchmark for Financial Deep Research</h1>

<div align="center">
<a href="https://arxiv.org/abs/2606.17458" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/datasets/DeepFin-Intelligence/ICBCBench"><img alt="Dataset" src="https://img.shields.io/badge/🤗%20Dataset-orange?color=FF6F00"></a>
<a href="https://huggingface.co/spaces/DeepFin-Intelligence/ICBCBench-Leaderboard"><img alt="Leaderboard" src="https://img.shields.io/badge/🏆%20Leaderboard-yellow?color=FFD700"></a>
<a href="https://github.com/DeepFin-Intelligence/ICBCBench/blob/main/LICENSE"><img src="https://img.shields.io/badge/Code_License-MIT-blue" alt="license"></a>
</div>


## Overview

ICBCBench is an industry consortium benchmark for evaluating financial Deep Research Agents in real-world research scenarios. It consists of bilingual objective and subjective tasks across major financial sectors, including capital markets, banking, insurance, and related financial services. Developed with over 50 contributors from more than 40 financial and academic organizations, ICBCBench combines verifiable question answering with expert-aligned long-form report evaluation to assess retrieval–reasoning accuracy, factual grounding, citation reliability, and end-to-end report quality.


 <p align="center">
  <img src="./images/overview.png" alt="ICBCBench Overview" width="90%"/>
</p>


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

The full offline suite is verified with Python 3.12.13. Use a separate virtual
environment and install the declared API SDK versions before running evaluation.

```bash
pip install -r requirements.txt
```

> Note: `subjective_eval/FACT/scrape.py` depends on `curl_cffi`.
> The official Tavily SDK is `tavily-python`. If an older environment has the
> unrelated `tavily` package installed, uninstall it before reinstalling requirements.

## Environment Variables

API keys are loaded from a `.env` file in the project root. Copy `.env.example` to `.env` and fill in the required keys:

```bash
cp .env.example .env
```

Current clients use `OPENROUTER_API_KEY` as the default. Refer to `model_clients.py` and `.env.example` for all supported keys.

Only the selected provider needs a key: ordinary models, Tongyi, and FACT use
`OPENROUTER_API_KEY`; Gemini DR uses `GEMINI_API_KEY`; OpenAI DR uses
`OPENAI_API_KEY`; Tavily uses `TAVILY_API_KEY`; Qwen DR uses `DASHSCOPE_API_KEY`.
Configuration is read from this project's `.env`, and `--help` does not require keys.
Short names such as `gpt-4o`, `gpt-5.4`, and `gemini-3.1-pro` are supported,
as are full OpenRouter IDs such as `openai/gpt-4o`. Output filenames use the
last component of a full model ID (for example, `gpt-4o.json`).

## License and Redistribution

The evaluation code is released under the [MIT License](LICENSE). The
benchmark-authored public task text, reference answers, and expert evaluation
rubrics are released under [Apache-2.0](data/LICENSE), as specified by the
existing Hugging Face dataset card. See [data license scope](data/DATA_LICENSE.md)
for the distinction between these materials and third-party sources.

Redistribution of the task data must comply with Apache-2.0, including providing
a copy of the license, retaining applicable notices, and identifying changes.
The data license does not grant rights to redistribute third-party reports,
charts, images, trademarks, or other source material.

The two figures referenced by objective question **25** are now represented by
official BNP Paribas PDF URLs and page locators. The prepared release contains
these references rather than copies of the third-party images or PDFs. Source
attribution and the publisher's terms are recorded in
[third-party sources](data/THIRD_PARTY_SOURCES.md).

## Data Paths

Default data files are expected under the project root:

- Objective questions: `data/objective_questions_public_80.json`
- Subjective questions: `data/subjective_questions_public_40.json`

The bundled JSON files are a URL-reference revision of the public
[Hugging Face dataset](https://huggingface.co/datasets/DeepFin-Intelligence/ICBCBench/tree/c1cbb5ed75daf6225035dbd89737aef1da86557f)
at commit `c1cbb5ed75daf6225035dbd89737aef1da86557f` (updated 2026-09-14).
This revision replaces question 25's two bundled image attachments with official
source URLs and page locators; it does not change the reference answer.
`data/huggingface_version.json` records the upstream source and the files in
the GitHub `Zhiwei` branch. Updating this GitHub branch does not update the
remote Hugging Face dataset; its URL-reference revision remains unpublished.

Use `--local_dataset` with the bundled JSON files to evaluate this revision. For a multimodal model, the toolkit retrieves each linked source PDF
and renders the designated page in memory for the request, without writing
third-party PDFs or page images to a disk cache. For a text-only model, the task
provides the source URLs and page descriptions. Source retrieval and rendering
do not grant redistribution rights to the publisher's material.

Historical Table 1 runs used the bundled screenshots. Replacing those inputs
with rendered PDF pages or URL references changes the evaluation input; the new
input format is not guaranteed to reproduce the published Table 1 scores.

Prediction, judging, and report generation also accept
`--dataset DeepFin-Intelligence/ICBCBench` in place of `--local_dataset`.
The objective entry points select `objective_questions`; report generation
selects `subjective_questions`. Both use the published `train` split.
Questions and any attachments in the selected remote revision are downloaded
from the same resolved HF snapshot. The following examples pin the historical
upstream revision, which still uses its original attachment format; they do not
select this branch's URL-reference revision:

```bash
python evaluation_toolkit/objective_eval/predict.py \
    --dataset DeepFin-Intelligence/ICBCBench \
    --dataset_revision c1cbb5ed75daf6225035dbd89737aef1da86557f \
    --model gpt-4o

python evaluation_toolkit/subjective_eval/write.py \
    --dataset DeepFin-Intelligence/ICBCBench \
    --dataset_revision c1cbb5ed75daf6225035dbd89737aef1da86557f \
    --model gpt-4o
```

`--dataset_config`, `--dataset_split`, and `--dataset_revision` override HF
loading options. Other HF repositories retain the default `test` split.
Local JSON loading does not download HF files. Models still require the
corresponding API credentials.

Generated evaluation results are written to `eval_result/`:

- Model reports: `eval_result/subjective_eval/dr_reports/reports_<model>.json`
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
    --judge gpt-5.4 \
    --num_workers 5
```

Outputs:
- `eval_result/objective_eval/judged_results/judge_<judge>/judged_gpt-4o.json`
- `eval_result/objective_eval/evaluation_results.csv`

### 3. Compute Metrics

```bash
python evaluation_toolkit/objective_eval/metrics.py
```

Pass `--judge <judge>` when using a different judge. `--query_file`,
`--judged_dir`, and `--output_dir` can be used for isolated subset evaluations.

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
    --judge gemini-3.1-pro \
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
python evaluation_toolkit/subjective_eval/FACT/extract.py \
    --raw_data_path eval_result/subjective_eval/dr_reports/reports_gpt-4o.json \
    --query_data_path data/subjective_questions_public_40.json \
    --output_path eval_result/subjective_eval/fact/extracted_gpt-4o.jsonl \
    --judge google/gemini-3-flash-preview --n_total_process 1

python evaluation_toolkit/subjective_eval/FACT/deduplicate.py \
    --raw_data_path eval_result/subjective_eval/fact/extracted_gpt-4o.jsonl \
    --query_data_path data/subjective_questions_public_40.json \
    --output_path eval_result/subjective_eval/fact/deduped_gpt-4o.jsonl \
    --judge google/gemini-3-flash-preview --n_total_process 1

python evaluation_toolkit/subjective_eval/FACT/scrape.py \
    --raw_data_path eval_result/subjective_eval/fact/deduped_gpt-4o.jsonl \
    --output_path eval_result/subjective_eval/fact/scraped_gpt-4o.jsonl \
    --n_total_process 1

python evaluation_toolkit/subjective_eval/FACT/validate.py \
    --raw_data_path eval_result/subjective_eval/fact/scraped_gpt-4o.jsonl \
    --query_data_path data/subjective_questions_public_40.json \
    --output_path eval_result/subjective_eval/fact/validated_gpt-4o.jsonl \
    --judge google/gemini-3-flash-preview --n_total_process 1
```

```bash
python evaluation_toolkit/subjective_eval/FACT/stat.py \
    --model_name gpt-4o \
    --input_path eval_result/subjective_eval/fact/validated_gpt-4o.jsonl \
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
- Prediction, report generation, expert scoring, and FACT model stages return a nonzero exit status when tasks fail. Successful records are saved for retry.
- Expert averages include genuine zero scores; tasks without a recorded score are excluded.
- Public subjective data uses `question` and `expert_evaluation_criteria` for both languages. Legacy bilingual fields remain supported.
- Objective accuracy uses every question in the selected dataset as its denominator;
  `Total`, `Evaluated`, and `Coverage (%)` make incomplete evaluations visible.
  Expert and FACT statistics also report task coverage. Incomplete stages return
  a nonzero status, and final aggregation rejects incomplete score inputs.
- Successful results are saved atomically as tasks complete. Changing a question,
  attachment, report, rubric, model parameters, or judge invalidates the relevant
  cached result. Legacy caches without input fingerprints are evaluated again.
- A failed replacement removes the stale result before the new request. Caches
  cannot combine different providers with the same model basename. Standalone
  objective metrics verify each judgment against its question and judge.
- Expert and FACT CSVs carry task-set and original-report fingerprints. Final
  aggregation requires complete coverage and matching provenance across all
  expert rounds and FACT. Regenerate older score files without these fields.
  Duplicate task records fail explicitly rather than overwriting one another.
- Truncated or filtered model responses are failed tasks even when their text
  or JSON is nonempty. OpenAI and Gemini clients try to cancel running background
  jobs when polling fails, times out, or the client invocation is interrupted.
  Parent-pipeline interruption signals research workers to stop polling. A failed
  cancellation reports the remote task ID; Tavily polling failures retain its
  task ID because this API does not expose cancellation.
- Provider citation annotations and sources are preserved in generated reports.
  Known offsets or explicit reference numbers become Markdown links. Tavily's
  source list does not document numeric reference IDs; when numeric citations
  lack URLs and an explicit mapping, FACT fails instead of guessing source order.
- PDF and Excel attachments are read in full by default. Attachment paths are
  relative to the selected dataset file. A failed PDF table conversion preserves
  its text; scanned PDFs without extractable text require preprocessing.
- Models that reject required image input fail explicitly; a text-only answer
  cannot replace the required image evaluation. Tavily and the current Qwen
  Deep Research client reject image tasks. Qwen's official API and main guide
  give inconsistent image schemas; this client uses the text API contract.
- `JINA_API_KEY` is optional for FACT scraping. Without it, or after Jina fails,
  the scraper uses its local HTML/PDF extraction path.
- Public subjective question **36** has an inconsistent rubric: primary dimension
  2 declares 30 points, but its five secondary dimensions total 36, making the
  full rubric 106 instead of 100. Scoring rejects it before calling a model and
  preserves the other results. The public rubric must be clarified before a
  complete 40-question subjective leaderboard can be produced.
- Additional public-data issues require author clarification: objective questions
  44, 51, and 53 refer to unspecified reports that are not attached; question 62
  lacks a target P/EV assumption; question 53's requested output layout conflicts
  with its answer layout. Question 28's JPMorgan answer conflicts with the 2024
  annual-report deposit figures under a deposit-amount insurance-coverage ratio.
  The nine Multi-Modal rows without attachments are 3, 28, 32, 51, 52, 55, 62,
  78, and 80; empty attachments alone do not prove an error where public source
  retrieval is intended. These answers have not been silently changed.

## Offline Regression Checks

```bash
python -m unittest discover -s tests -v
```

These checks use real SDKs with local HTTP transports, real HTML/PDF extraction,
and simulated service responses. They cover all 80 objective questions, all 40
subjective questions (including the known rubric conflict), bilingual data,
images, interrupted runs, cache invalidation, malformed responses, multiprocessing,
and final-score aggregation. They do not use live API keys or generate benchmark scores.


## Contact

For questions or feedback about ICBCBench, please contact:

- **Zhiwei Tang**: [zwtang1220@gmail.com](mailto:zwtang1220@gmail.com)
- **Weiya Li**: [weiyali126@outlook.com](mailto:weiyali126@outlook.com)
- **Yizhou He**: [jonah_he@163.com](mailto:jonah_he@163.com)
- **Li Guo**: [guo_li@fudan.edu.cn](mailto:guo_li@fudan.edu.cn)
- **Linfeng Zhang**: [zhanglinfeng@sjtu.edu.cn](mailto:zhanglinfeng@sjtu.edu.cn)

## Citation

```bibtex
@misc{li2026icbcbenchindustryconsortiumbenchmark,
      title={ICBCBench: An Industry Consortium Benchmark for Financial Deep Research}, 
      author={Weiya Li and Zhiwei Tang and Yizhou He and Chenghao Wang and Liang Feng and Xiao Sun and Dongrui Liu and Zichen Wen and Hu Wei and Jinghang Wang and Yi Luo and Li Guo and Linfeng Zhang},
      year={2026},
      eprint={2606.17458},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2606.17458}, 
}
``` 
