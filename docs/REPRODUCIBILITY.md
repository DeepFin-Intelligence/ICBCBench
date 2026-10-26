# Reproduction conventions

## Scope and versions

The bundled release contains 80 objective and 40 subjective tasks, equally split
between English and Chinese. No private hold-out tasks or private model outputs
are required by the public pipeline. The source dataset revision, current file
hashes and URL attachment provenance are in `data/huggingface_version.json`.
The URL-reference revision is available on GitHub branch `Zhiwei`; it has not
been published to the remote Hugging Face dataset.

Use Python 3.12 and `requirements-lock.txt`. The unified `run` and `score` commands record
Python/dependency versions, model and judge IDs, source dataset hashes, selected
original task IDs, the code/prompt/configuration hash and reference date in
`run_manifest.json`. It saves exact subset JSONs under `inputs/`. Preserve the
manifest, intermediate JSON/JSONL records and CSVs with an experiment. API keys
are never included in the manifest. Task files, manifests and stage logs use
explicit UTF-8 encoding, including when the host default encoding differs.

Provider model aliases and online sources can change. Even with the same prompts,
model IDs and temperature, live outputs and judge scores may vary. The public
pipeline provides rerunnable evaluation and auditable aggregation; reproducing
historical Table 1 numbers additionally requires the historical generated
reports, judgments, source snapshots and original evaluation conventions. The
repository does not fabricate these missing artifacts.

## Imported results and custom systems

`score` evaluates existing objective answers and/or subjective reports without
regenerating them. Its judges, three expert-rubric rounds, citation/source checks,
task 36 correction and aggregation are the same as `run`. Scoring still requires
judge credentials and network access; only `--dry_run` is an offline import
validation and plan preview. Formats and commands are in
[CUSTOM_MODELS.md](CUSTOM_MODELS.md).

Imports use original task IDs within each track. By default, the file must cover
every selected task exactly once; the default selection is all public tasks in
the supplied tracks. `--language` and `--limit` select a smaller fixed set.
`--allow_subset` explicitly evaluates a nonempty partial set from those selected
tasks, with that set frozen for every downstream stage. Unknown or extra IDs are
rejected even in subset mode. Report the selected IDs with subset scores and
compare only identical task sets. The import manifest records source-file
provenance; changing source responses or run settings requires a new run folder.

`run --model_backend openai` sends generation requests to an explicitly supplied
Chat Completions endpoint, with its own optional local-service credential. It does
not change the judge endpoint or credentials. `run --model_backend agent` calls
a developer-provided Python adapter with task ID, track, language, model label and
formatted messages; reference answers and rubrics are excluded from this input.
The adapter owns its research tools, provider configuration and retry policy.
Record the tested service's actual model version and the adapter's code/dependency
revision alongside the runner manifest. A model label alone does not pin a remote
deployment or describe an agent's retrieval setting.

Custom generation supports an explicit attachment mode: `url` passes source URLs
and page locators as text without fetching them, while `image` renders the
referenced PDF pages for an image-capable interface. Record this setting and
keep it consistent across compared systems. The example agent and imported
fixture files are synthetic integration examples, not benchmark measurements.

## Public subjective task 36: explicit correction

The original rubric is preserved byte for byte. Its primary dimension maxima
are 25, 30, 30 and 15. The five secondary maxima within primary dimension 2 are
8, 10, 6, 6 and 6, totaling 36 rather than 30. The raw rubric therefore permits
106 points. Under the author-approved correction:

```
S_expert(task 36) = sum(raw scores in primary dimensions 1, 3, 4)
                 + (30 / 36) * sum(raw scores in primary dimension 2)
```

The judge returns raw secondary scores and is explicitly instructed not to
normalize them. The evaluator validates every raw score against its original
cap and applies the factor once, without integer rounding. A perfect raw score
of 106 becomes 100; a raw score of 53 at half of every cap becomes 50.

Each score record retains `dimensions_breakdown` (raw), `raw_total_score`,
`total_score` (normalized) and `scoring_adjustments`. The policy identifier is
`public-subjective-36-primary-2-v1`. It applies only when the rubric SHA-256
matches `RUBRIC_36_SHA256` in `score_by_criteria.py`, preventing accidental
application to another task or an edited rubric. Other inconsistent structured
rubrics remain errors. The scoring cache fingerprint includes this policy.

This correction is a release convention, not a claim about how historical
Table 1 was scored. Historical results are not overwritten or silently rescaled.

## Three expert rounds and source reference date

The default runner saves three separate model-judge rounds using expert rubrics,
all judging the same generated or imported reports. These are not human scoring
sessions. Their model-level averages are averaged before applying
`S_final = 0.8 * S_expert + 0.1 * S_citation + 0.1 * S_source`. A repeated command
resumes completed tasks; it does not create an additional scoring round. Missing
requested round files and duplicate round-file inputs are rejected. Intermediate
expert CSVs preserve numeric precision for individual scores and their means;
only the final leaderboard rounds displayed scores to two decimals. This avoids
changing the last displayed digit through repeated rounding.

`--as_of YYYY-MM-DD` is required for subjective runs through the unified runner.
It denotes midnight UTC; ISO timestamps with offsets are also accepted. The
FACT statistics CLI supports the same option. When that individual CLI is used
without the option, it captures the current UTC time once and records it.
FACT and final CSVs include `Evaluation As Of (UTC)`; final CSVs also include
`Expert Rounds`. Final aggregation requires this date field and rejects missing,
invalid or different reference dates among the models being compared; equivalent
timezone offsets normalize to the same UTC instant. For older FACT CSVs without
this metadata, rerun FACT statistics with an explicit `--as_of` date. Reuse the
same reference date for comparisons.

Source timeliness remains `exp(-0.002 * age_in_whole_days)`, rounded to four
decimals before aggregation. Datetimes are converted to UTC. Missing components
in a parseable partial publication date default to year 1970, January and day 1.
Unparsable or missing publication dates contribute zero timeliness/source quality. Future
publication dates have age zero. Changing the reference date does not refetch
or rejudge sources. Save scraped contents and publication dates for later audits;
redistributing third-party contents requires the corresponding rights.

## Retrieval and failures in this release

Ordinary OpenRouter chat models receive the task prompt and supported attachments;
the toolkit does not attach a search tool to these baseline chat requests.
The same applies to custom chat endpoints. A custom agent supplies its own search
or retrieval workflow; result import does not establish how responses were
generated, so the developer must describe that setting separately.
Deep Research clients use their provider's research API or configured agent
workflow. Report/judgment metadata preserves citation annotations where supported.
The objective task 25 PDFs are fetched and selected pages rendered as attachment
inputs for supported image interfaces. Text-only research interfaces receive
source URLs/page locators. This is different from the historical screenshot input.

The OpenRouter client permits one SDK retry for transient API failures. Citation
extraction, semantic deduplication and citation validation additionally allow up
to three application attempts. These are bounded attempts, not retries until
every task succeeds.
Provider-specific research clients have their own polling timeout/cancellation
handling. Failed, empty, truncated or invalid outputs do not become successful
cached responses; the stage returns a nonzero status and retains completed work.
The runner stops at that stage. Rerun the same command to retry remaining work.
Deduplication may reuse unchanged fact selections, but always retains the current
report, model and question provenance. A changed input language invalidates that
decision cache; legacy caches without language metadata regenerate once. Source
retrieval and validation each apply their own cache checks.

Objective accuracy uses the full selected task set as denominator, so missing
judgments lower partial accuracy and coverage. Expert statistics exclude missing
scores from their partial mean and disclose coverage; genuine recorded zero
scores remain included. FACT reports with no citations are complete zero-score
tasks. Extraction, scraping and validation failures are incomplete tasks, not
valid zero judgments. Final subjective aggregation requires complete coverage
and matching task/report fingerprints across every expert round and FACT. A
partial CSV is diagnostic and cannot be presented as a completed leaderboard.

These statements describe the released implementation. Do not infer a historical
experiment's retrieval or failure handling solely from a later code revision.

## Verification and remaining data questions

`python -m evaluation_toolkit check` verifies local release hashes, bilingual
counts, all structured rubrics and installed packages without contacting any
provider. `demo` uses fabricated inputs and runs real score validation and
aggregation, producing 50% objective accuracy and a 74.00 subjective final score.
It is not an evaluation of any model. The regression suite also exercises the
real SDK request/response interfaces with offline transports and all public
tasks. It has been tested on Ubuntu and macOS with Python 3.12.

The README lists unresolved content questions in several objective tasks. The
release preserves their wording and answers rather than inventing missing
financial assumptions. These content limitations are separate from whether the
software workflow runs. Source URLs still require network access and continued
publisher availability; inaccessible sources cause an explicit stage failure.
