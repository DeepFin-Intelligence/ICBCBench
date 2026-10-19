# Evaluate your own model or research agent

Install the locked dependencies using the [README](../README.md#installation),
then run the commands below from the repository root. Choose one workflow:

| You already have | Entry point | What ICBCBench runs |
| --- | --- | --- |
| Answers or research reports | `score` | Judges, citation/source checks and aggregation |
| An OpenAI-compatible chat endpoint | `run --model_backend openai` | Generation through your endpoint, then scoring |
| A Python research agent | `run --model_backend agent` | Your callable for generation, then scoring |

The tested model and the scoring judges are separate. `--model` labels the tested
system. The default judges use OpenRouter through `OPENROUTER_API_KEY`: objective
judge `gpt-5.4`, expert-rubric judge `gemini-3.1-pro`, and FACT judge
`google/gemini-3-flash-preview`. Override them with `--objective_judge`,
`--expert_judge`, and `--fact_judge`. Keep the judge configuration identical when
comparing systems.

## 1. Import existing answers and reports

This workflow does not require access to your model or its serving framework.
Generate responses with your own system, save one response per original task ID,
and provide separate files for the objective and subjective tracks. Do not submit
reference answers, rubrics, or scores in place of model responses.

### Supported result formats

Use UTF-8 JSON or JSONL. Each record has:

| Field | Meaning |
| --- | --- |
| `id` | Original task ID; required in JSON arrays and JSONL, supplied by the key in a keyed JSON object |
| `response` | Nonempty answer or complete report as a string |
| `model` | Optional system ID; when present it must agree with `--model` |

Keyed JSON object:

```json
{
  "1": {"response": "Explanation: ...\nExact Answer: ...\nConfidence: 80%"},
  "41": {"response": "解析：...\n确切答案：...\n置信度：80%"}
}
```

JSON array:

```json
[
  {"id": 1, "response": "The complete English report, including source links."},
  {"id": 21, "response": "完整中文报告，包含来源链接。"}
]
```

JSONL uses one record per line, with the same `id` and `response` fields; see
[objective.jsonl](../examples/results/objective.jsonl). Do not renumber a subset
to consecutive IDs: objective IDs `1` and `41`, and subjective IDs `1` and `21`,
are the first English and Chinese tasks in the respective datasets. IDs belong
to their own track; objective task 1 and subjective task 1 are different tasks.

For objective answers, preserve the explanation, exact final answer, and
confidence from the model. The judge extracts confidence from the response text;
the existing judge convention uses 100 when confidence is absent. For reports,
preserve the full report and actual source URLs in Markdown links or a reference
list that resolves inline citations. A bare `[1]` without a corresponding URL
cannot identify a source for FACT. Do not add citations that the system did not
use. A report with no citations has zero citation/source scores.

### Validate a small import without API calls

The bundled examples are **synthetic format fixtures**, not historical outputs
or results from any evaluated model. This command validates their format and
selected task coverage and prints the scoring plan:

```bash
python -m evaluation_toolkit score \
    --model synthetic-example \
    --objective_results examples/results/objective.json \
    --subjective_results examples/results/subjective.json \
    --limit 2 --as_of 2026-10-19 \
    --run_dir eval_result/import-example --dry_run
```

`--dry_run` does not call APIs or create run outputs. Removing it starts real
judge calls and, when citations exist, source retrieval. Scoring imported results
is **not offline evaluation**. Use `python -m evaluation_toolkit demo` for the
separate offline, synthetic scoring/aggregation demonstration.

### Score your complete results

Replace the paths with your own files, set `OPENROUTER_API_KEY` in the repository
`.env`, and choose an explicit source-timeliness reference date:

```bash
python -m evaluation_toolkit score \
    --model my-model \
    --objective_results /path/to/my-objective.json \
    --subjective_results /path/to/my-reports.jsonl \
    --as_of 2026-10-19 \
    --run_dir eval_result/my-model-import
```

The default selection is all 80 objective and 40 subjective tasks. The command
infers the track from the supplied result paths. To score just reports, supply
only `--subjective_results`; to score just objective answers, supply only
`--objective_results` (`--as_of` is not required for objective-only scoring).
An explicit `--track objective`, `subjective`, or `both` must match the supplied
files. The result files must cover every selected task exactly once. Duplicate,
unknown, extra, empty, and mismatched-model records are rejected before scoring.

`--language en` or `--language zh` selects one language. `--limit N` selects at
most N tasks **per track**, alternating English and Chinese when both languages
are selected. Supply results for exactly that selected set, as in the two-task
example above. For an arbitrary subset, explicitly opt in:

```bash
python -m evaluation_toolkit score \
    --model my-model --objective_results /path/to/my-subset.json \
    --allow_subset --run_dir eval_result/my-model-subset --dry_run
```

`--allow_subset` permits missing selected IDs and evaluates the nonempty set
actually supplied; it does not permit unknown IDs or IDs outside a language/limit
selection. The saved task subset determines all score denominators. Report its
original IDs and compare results only on the same task set. Subset scores are
not full-benchmark scores.

### Outputs, provenance and resume

The imported results feed the same judges and aggregation as `run`, skipping
answer and report generation. The runner freezes the selected questions and
records imported-file provenance in `run_manifest.json`. Preserve that manifest,
the input files and all scored outputs with an experiment. Do not edit imported
responses or reuse a run directory for different inputs or configuration.

Useful outputs under `--run_dir`:

- `objective/metrics/evaluation_results.csv`: accuracy, calibration and coverage;
- `subjective/expert_round_*/judge_<judge>/scores_<model>.jsonl`: per-task expert
  rubric scores, including raw and normalized values;
- `subjective/final_leaderboard.csv` and language-specific CSVs: final weighted scores;
- `inputs/`, `logs/` and `run_manifest.json`: selected tasks and execution records.

Here `<judge>` and `<model>` in filenames use the last component of a slash-delimited
ID. Three expert-rubric rounds are model-judge calls on the **same reports**, not
three human experts or three new reports. The default final score is
`0.8 * expert + 0.1 * citation + 0.1 * source`. The task 36 correction and rounding
conventions in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) apply to imported reports too.

A failed stage stops the pipeline and preserves successful records. Rerun the
same command to resume. Only `status: complete` in the run manifest denotes a
finished run; partial CSVs must not be reported as a completed leaderboard.

## 2. Connect a custom chat endpoint

Start your own server implementing the OpenAI Chat Completions API, then supply
its base URL (including the usual `/v1` API prefix) and exact model ID:

```bash
python -m evaluation_toolkit run \
    --model_backend openai --model my-model \
    --model_base_url http://127.0.0.1:8000/v1 \
    --model_api_key_env ICBCBENCH_MODEL_API_KEY --model_timeout 120 \
    --attachment_mode url --track both --limit 2 \
    --as_of 2026-10-19 --run_dir eval_result/my-model-endpoint --dry_run
```

If your endpoint requires authentication, put its key in
`ICBCBENCH_MODEL_API_KEY` in `.env`, or use `--model_api_key_env` to select a
different environment-variable name. Do not put a secret directly in a command.
For an unauthenticated loopback service, omit `--model_api_key_env`. Remote
endpoints require HTTPS and an explicitly configured key environment variable.
The local
generation service needs no provider key, but actual scoring still needs
`OPENROUTER_API_KEY`. Your custom base URL and key apply only to generation;
judges continue to use their independently configured OpenRouter client.

Remove `--dry_run` to run the experiment. The endpoint must return a nonempty
completed chat response. Truncated, filtered or failed requests are failures;
they are not silently counted as valid answers. Endpoint support for messages,
token limits and generation parameters must be compatible with Chat Completions.
There is no automatic retrieval/search tool attached to a plain chat endpoint.
If your system needs its own tool loop, use the agent interface below or import
results generated by that system.

`--attachment_mode url` passes the source URL and page locator as text. It does
not fetch the source on behalf of your model. Use `--attachment_mode image` for
an endpoint that accepts image content blocks; the referenced PDF page is then
rendered for the request. Record the mode, because these are different model
inputs. Neither mode recreates the exact screenshots in historical Table 1.

## 3. Connect a Python research agent

Implement a callable importable from the repository root. The built-in
[example adapter](../examples/custom_agent.py) can be selected as follows:

```bash
python -m evaluation_toolkit run \
    --model_backend agent --agent examples.custom_agent:generate \
    --model synthetic-agent --attachment_mode url \
    --track both --limit 2 --as_of 2026-10-19 \
    --run_dir eval_result/synthetic-agent --dry_run
```

This adapter generates synthetic placeholders without network access. It is an
integration example, not a working financial research system. Removing
`--dry_run` invokes it and then starts real paid judge calls. Replace its body
with your agent before reporting model performance. You can first inspect its
contract locally without judges:

```bash
python -c 'from examples.custom_agent import generate; print(generate({"id": "1", "track": "objective", "language": "en", "model": "synthetic-agent", "messages": []}))'
```

The callable receives one dictionary:

```python
{
    "id": "1",
    "track": "objective",       # or "subjective"
    "language": "en",           # or "zh"
    "model": "my-agent",
    "messages": [               # formatted task prompt and attachments
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ],
}
```

The task passed to the callable does not include reference answers or expert
rubrics. Use `messages` as the generation input; it may include image content
blocks when image mode is selected. Your implementation owns its model calls,
retrieval, timeouts, retries, and service credentials. `--model_timeout` controls
the custom chat client; it does not enforce a timeout on your Python callable.
Return:

```python
{
    "content": "Completed answer or report with actual source URL citations.",
    "usage": {"input_tokens": 100, "output_tokens": 200},  # optional
    "citations": [],                                       # optional metadata
}
```

`content` is required and must be a nonempty string. Include usable source URLs
in the report itself; a separate metadata list is not a substitute for citations
attached to claims. Raise an exception when generation fails. Do not return an
error string as if it were a completed report. The runner can use multiple
workers, so make your callable safe for concurrent tasks or use `--workers 1`.
Keep your adapter and its dependencies fixed for a reproducible experiment.

## Reporting a comparison

Keep the dataset revision, original task IDs, judges, expert rounds, source
reference date, attachment mode and aggregation policy identical across systems.
Record whether the tested system had retrieval/tools, the model/service version,
generation configuration, and your agent's source revision. The manifest records
runner settings; it cannot infer a service's internal version or agent search
policy. Report these separately. Online providers and cited sources can change,
so a new run is not guaranteed to reproduce historical Table 1 scores.
