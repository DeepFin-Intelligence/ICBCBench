import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p.name != "evaluation_toolkit" and _p.parent != _p:
    _p = _p.parent
if _p.name == "evaluation_toolkit":
    _root = _p.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import argparse
import pandas as pd
import os
import math

from evaluation_toolkit.utils import get_project_root
from evaluation_toolkit.model_clients import resolve_openrouter_model
from evaluation_toolkit.subjective_eval.scoring.provenance import valid_fingerprint


def normalize_model_name(name):
    """Normalize model name for matching: lowercase and strip whitespace"""
    return resolve_openrouter_model(str(name).strip().lower())


def compute_final_score(expert_csvs, fact_csv, output_csv, alpha, beta, gamma):
    """
    Merge ExpertCriteria (averaged across multiple judgments) and FACT scores to compute the final score.

    S_expert = mean(Avg Score across multiple expert judgments)
    S_final = alpha * S_expert + beta * S_citation + gamma * S_source

    expert_csvs: list of paths to ExpertCriteria leaderboard CSVs
    """
    weights = (alpha, beta, gamma)
    if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not math.isclose(sum(weights), 1):
        raise ValueError("Final score weights must be finite, nonnegative, and sum to 1")
    # 1. Read multiple ExpertCriteria CSVs and compute average Avg Score
    expert_dfs = []
    for csv_path in expert_csvs:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Expert CSV does not exist: {csv_path}")
        df = pd.read_csv(csv_path, encoding='utf-8')
        if 'Model' not in df.columns or 'Avg Score' not in df.columns:
            raise ValueError(f"Expert CSV missing Model or Avg Score: {csv_path}")
        df['_norm_model'] = df['Model'].apply(normalize_model_name)
        validate_scores(df, ['Avg Score'], csv_path)
        validate_coverage(df, csv_path)
        validate_provenance(df, csv_path)
        if expert_dfs:
            check_matching_inputs(expert_dfs[0], df, csv_path, require_same_models=True)
        expert_dfs.append(df[['Model', 'Avg Score', '_norm_model', 'Total Tasks',
                              'Task Set Fingerprint', 'Report Set Fingerprint']])

    if not expert_dfs:
        raise ValueError("No expert score inputs were provided")

    combined_expert = pd.concat(expert_dfs, ignore_index=True)
    df_expert = (
        combined_expert
        .groupby('_norm_model', as_index=False)
        .agg({'Model': 'first', 'Avg Score': 'mean', 'Total Tasks': 'first',
              'Task Set Fingerprint': 'first', 'Report Set Fingerprint': 'first'})
    )

    # 2. Read FACT CSV
    df_fact = pd.read_csv(fact_csv, encoding='utf-8')
    # Required columns: Model Name, Avg Citation Score (Macro), Avg Source Quality Score (S_source)
    required_fact_cols = ['Model Name', 'Avg Citation Score (Macro)', 'Avg Source Quality Score (S_source)']
    for col in required_fact_cols:
        if col not in df_fact.columns:
            raise ValueError(f"FACT CSV missing required column '{col}', current columns: {list(df_fact.columns)}")

    # 3. Normalize model names for matching
    df_fact['_norm_model'] = df_fact['Model Name'].apply(normalize_model_name)
    validate_scores(df_fact, required_fact_cols[1:], fact_csv)
    validate_coverage(df_fact, fact_csv)
    validate_provenance(df_fact, fact_csv)
    missing_models = set(df_expert['_norm_model']) - set(df_fact['_norm_model'])
    if missing_models:
        raise ValueError(f"FACT scores are missing for models: {sorted(missing_models)}")
    check_matching_inputs(df_expert, df_fact, fact_csv)

    # 4. Merge by normalized model name (using ExpertCriteria as base, fill 0 for missing FACT models)
    df_merged = pd.merge(
        df_expert[['Model', 'Avg Score', '_norm_model', 'Total Tasks',
                   'Task Set Fingerprint', 'Report Set Fingerprint']],
        df_fact[['Model Name', 'Avg Citation Score (Macro)', 'Avg Source Quality Score (S_source)', '_norm_model']],
        on='_norm_model',
        how='left',
        suffixes=('', '_fact')
    )

    if df_merged.empty:
        raise ValueError("Expert CSV contains no models")

    # Fill 0 for models missing in FACT
    df_merged['Avg Citation Score (Macro)'] = df_merged['Avg Citation Score (Macro)'].fillna(0)
    df_merged['Avg Source Quality Score (S_source)'] = df_merged['Avg Source Quality Score (S_source)'].fillna(0)

    # 5. Compute final score
    df_merged['S_expert'] = df_merged['Avg Score']
    df_merged['S_citation'] = df_merged['Avg Citation Score (Macro)']
    df_merged['S_source'] = df_merged['Avg Source Quality Score (S_source)']
    df_merged['S_final'] = (
        alpha * df_merged['S_expert']
        + beta * df_merged['S_citation']
        + gamma * df_merged['S_source']
    )

    # 6. Select output columns
    df_out = df_merged[[
        'Model',
        'S_expert',
        'S_citation',
        'S_source',
        'S_final', 'Total Tasks', 'Task Set Fingerprint', 'Report Set Fingerprint'
    ]].copy()
    df_out['Provenance Verified'] = True

    # Sort by S_final descending
    df_out = df_out.sort_values(by='S_final', ascending=False).reset_index(drop=True)

    # Round to two decimal places
    for col in ['S_expert', 'S_citation', 'S_source', 'S_final']:
        df_out[col] = df_out[col].round(2)

    # 7. Write to CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"\nMerge complete! Total models: {len(df_out)}")
    print(f"Weights: alpha={alpha}, beta={beta}, gamma={gamma}")
    print(f"Output file: {output_csv}")

    return df_out


def validate_coverage(frame, source):
    required = {'Coverage (%)', 'Total Tasks', 'Evaluated Tasks'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Score input lacks verified coverage metadata; regenerate {source}')
    if 'Coverage (%)' in frame:
        coverage = pd.to_numeric(frame['Coverage (%)'], errors='raise')
        if not coverage.map(lambda value: math.isfinite(value) and value == 100).all():
            raise ValueError(f"Score input is incomplete: Coverage (%) must be 100 in {source}")
    if {'Total Tasks', 'Evaluated Tasks'}.issubset(frame.columns):
        total = pd.to_numeric(frame['Total Tasks'], errors='raise')
        evaluated = pd.to_numeric(frame['Evaluated Tasks'], errors='raise')
        if (not total.map(lambda value: math.isfinite(value) and value > 0 and value == int(value)).all()
                or not total.eq(evaluated).all()):
            raise ValueError(f"Score input is incomplete: all tasks must be evaluated in {source}")


def validate_provenance(frame, source):
    columns = {'Task Set Fingerprint', 'Report Set Fingerprint', 'Provenance Verified'}
    if not columns.issubset(frame.columns):
        raise ValueError(f'Score input lacks provenance metadata; regenerate {source}')
    if not frame['Provenance Verified'].map(lambda value: str(value).strip().lower() == 'true').all():
        raise ValueError(f'Score input has unverified provenance; regenerate {source}')
    for column in ('Task Set Fingerprint', 'Report Set Fingerprint'):
        if not frame[column].map(valid_fingerprint).all():
            raise ValueError(f'Invalid {column} in {source}')
    if frame['Task Set Fingerprint'].nunique() != 1:
        raise ValueError(f'Score input mixes different task sets: {source}')


def check_matching_inputs(base, other, source, require_same_models=False):
    if require_same_models and set(base['_norm_model']) != set(other['_norm_model']):
        raise ValueError(f'Expert score rounds contain different model sets: {source}')
    if set(base['Task Set Fingerprint']) != set(other['Task Set Fingerprint']):
        raise ValueError(f'Score inputs refer to different question sets: {source}')
    others = other.set_index('_norm_model')
    for _, row in base.iterrows():
        if row['_norm_model'] not in others.index:
            continue
        matched = others.loc[row['_norm_model']]
        if row['Total Tasks'] != matched['Total Tasks']:
            raise ValueError(f'Score inputs have inconsistent task counts: {source}')
        if row['Report Set Fingerprint'] != matched['Report Set Fingerprint']:
            raise ValueError(f'Score inputs refer to different generated reports for model {row["_norm_model"]}: {source}')


def validate_scores(frame, columns, source):
    if frame.empty:
        raise ValueError(f"Score CSV is empty: {source}")
    if frame['_norm_model'].eq('nan').any() or frame['_norm_model'].eq('').any():
        raise ValueError(f"Missing model name in {source}")
    if frame['_norm_model'].duplicated().any():
        raise ValueError(f"Duplicate normalized model names in {source}")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not frame[column].map(lambda value: math.isfinite(value) and 0 <= value <= 100).all():
            raise ValueError(f"Scores in {column} must be finite numbers from 0 to 100: {source}")


def run_all_versions(expert_dirs, fact_csv_base, output_dir, alpha, beta, gamma):
    """
    Run ALL, zh, and en versions, consistent with ExpertCriteria/stat.py.
    expert_dirs: list of directories containing ExpertCriteria leaderboard CSVs
    """
    completed = 0
    for lang in [None, "zh", "en"]:
        # Expert CSV paths (multiple rounds)
        # Look for leaderboard.csv or leaderboard_<lang>.csv in each expert directory
        expert_csvs = []
        for d in expert_dirs:
            if lang:
                p = os.path.join(d, f"leaderboard_{lang}.csv")
            else:
                p = os.path.join(d, "leaderboard.csv")
            if os.path.exists(p):
                expert_csvs.append(p)

        if not expert_csvs:
            print(f"Skip: No Expert CSV found (lang={lang})")
            continue

        if lang:
            output_csv = os.path.join(output_dir, f"final_leaderboard_{lang}.csv")
        else:
            output_csv = os.path.join(output_dir, "final_leaderboard.csv")

        # FACT CSV path: prefer language-specific variant
        fact_csv = fact_csv_base
        if lang:
            fact_csv_lang = fact_csv_base.replace('.csv', f'_{lang}.csv')
            fact_csv = fact_csv_lang

        if not os.path.exists(fact_csv):
            raise FileNotFoundError(f"FACT CSV does not exist for language {lang or 'ALL'}: {fact_csv}")

        print(f"\n{'='*50}")
        lang_label = lang.upper() if lang else "ALL"
        print(f"Processing: {lang_label}")
        print(f"Expert CSVs: {expert_csvs}")
        print(f"{'='*50}")
        compute_final_score(expert_csvs, fact_csv, output_csv, alpha, beta, gamma)
        completed += 1
    if not completed:
        raise FileNotFoundError("No ExpertCriteria leaderboard files were found")


if __name__ == "__main__":
    project_root = get_project_root()
    default_expert_dirs = [
        os.path.normpath(os.path.join(project_root, "eval_result", "subjective_eval", "scores", "judge_gemini-3.1-pro-preview")),
    ]
    default_fact_csv = os.path.normpath(os.path.join(project_root, "eval_result", "subjective_eval", "fact_result.csv"))
    default_output_dir = os.path.normpath(os.path.join(project_root, "eval_result", "subjective_eval"))

    parser = argparse.ArgumentParser(description="Merge ExpertCriteria (multi-round average) + FACT scores to compute final score")
    parser.add_argument("--expert_dirs", type=str, nargs='+',
                        default=default_expert_dirs,
                        help="List of directories containing ExpertCriteria leaderboard CSVs (supports multi-round averaging)")
    parser.add_argument("--fact_csv", type=str,
                        default=default_fact_csv,
                        help="FACT score CSV path (base path; will auto-try _zh/_en language-specific variants)")
    parser.add_argument("--output_dir", type=str,
                        default=default_output_dir,
                        help="Final score output directory")
    parser.add_argument("--language_filter", type=str, default=None, choices=["zh", "en"],
                        help="Language filter: zh/en; if not specified, run all three versions")
    parser.add_argument("--alpha", type=float, default=0.8, help="Expert weight")
    parser.add_argument("--beta", type=float, default=0.1, help="Citation weight")
    parser.add_argument("--gamma", type=float, default=0.1, help="Source weight")

    args = parser.parse_args()

    if args.language_filter:
        # Run only the specified language
        expert_csvs = []
        for d in args.expert_dirs:
            p = os.path.join(d, f"leaderboard_{args.language_filter}.csv")
            if os.path.exists(p):
                expert_csvs.append(p)

        output_csv = os.path.join(args.output_dir, f"final_leaderboard_{args.language_filter}.csv")

        # FACT CSV: prefer language-specific version
        fact_csv = args.fact_csv
        fact_csv_lang = fact_csv.replace('.csv', f'_{args.language_filter}.csv')
        fact_csv = fact_csv_lang

        if expert_csvs and os.path.exists(fact_csv):
            compute_final_score(expert_csvs, fact_csv, output_csv, args.alpha, args.beta, args.gamma)
        else:
            raise FileNotFoundError(f"Input files do not exist. Expert={expert_csvs}, FACT={fact_csv}")
    else:
        # Default: run all three versions
        run_all_versions(args.expert_dirs, args.fact_csv, args.output_dir, args.alpha, args.beta, args.gamma)
