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

from evaluation_toolkit.utils import get_eval_data_dir


def normalize_model_name(name):
    """标准化模型名用于匹配：小写、去除多余空格"""
    return str(name).strip().lower()


def compute_final_score(expert_csvs, fact_csv, output_csv, alpha, beta, gamma):
    """
    整合 ExpertCriteria（多轮次取平均）和 FACT 的评分，计算最终总分。

    S_expert = mean(Avg Score across multiple expert judgments)
    S_final = alpha * S_expert + beta * S_citation + gamma * S_source

    expert_csvs: list of paths to ExpertCriteria leaderboard CSVs
    """
    # 1. 读取多轮 ExpertCriteria CSV 并计算平均 Avg Score
    expert_dfs = []
    for csv_path in expert_csvs:
        if not os.path.exists(csv_path):
            print(f"跳过不存在的 Expert CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path, encoding='utf-8')
        if 'Model' not in df.columns or 'Avg Score' not in df.columns:
            print(f"跳过: Expert CSV 缺少必要列，当前列: {list(df.columns)} | 文件: {csv_path}")
            continue
        df['_norm_model'] = df['Model'].apply(normalize_model_name)
        expert_dfs.append(df[['Model', 'Avg Score', '_norm_model']])

    if not expert_dfs:
        print("警告: 没有有效的 Expert CSV。")
        return

    combined_expert = pd.concat(expert_dfs, ignore_index=True)
    df_expert = (
        combined_expert
        .groupby('_norm_model', as_index=False)
        .agg({'Model': 'first', 'Avg Score': 'mean'})
    )

    # 2. 读取 FACT CSV
    df_fact = pd.read_csv(fact_csv, encoding='utf-8')
    # 关键列：Model Name, Avg Citation Score (Macro), Avg Source Quality Score (S_source)
    required_fact_cols = ['Model Name', 'Avg Citation Score (Macro)', 'Avg Source Quality Score (S_source)']
    for col in required_fact_cols:
        if col not in df_fact.columns:
            raise ValueError(f"FACT CSV 缺少必要列 '{col}'，当前列: {list(df_fact.columns)}")

    # 3. 标准化模型名用于匹配
    df_fact['_norm_model'] = df_fact['Model Name'].apply(normalize_model_name)

    # 4. 按标准化模型名合并（以 ExpertCriteria 为基准，FACT 缺失的模型填 0）
    df_merged = pd.merge(
        df_expert[['Model', 'Avg Score', '_norm_model']],
        df_fact[['Model Name', 'Avg Citation Score (Macro)', 'Avg Source Quality Score (S_source)', '_norm_model']],
        on='_norm_model',
        how='left',
        suffixes=('', '_fact')
    )

    if df_merged.empty:
        print("警告: Expert CSV 为空。")
        return

    # FACT 中缺失的模型，分数填 0
    df_merged['Avg Citation Score (Macro)'] = df_merged['Avg Citation Score (Macro)'].fillna(0)
    df_merged['Avg Source Quality Score (S_source)'] = df_merged['Avg Source Quality Score (S_source)'].fillna(0)

    # 5. 计算最终分数
    df_merged['S_expert'] = df_merged['Avg Score']
    df_merged['S_citation'] = df_merged['Avg Citation Score (Macro)']
    df_merged['S_source'] = df_merged['Avg Source Quality Score (S_source)']
    df_merged['S_final'] = (
        alpha * df_merged['S_expert']
        + beta * df_merged['S_citation']
        + gamma * df_merged['S_source']
    )

    # 6. 选择输出列并重命名
    df_out = df_merged[[
        'Model',
        'S_expert',
        'S_citation',
        'S_source',
        'S_final'
    ]].copy()

    # 按 S_final 降序排序
    df_out = df_out.sort_values(by='S_final', ascending=False).reset_index(drop=True)

    # 数值保留两位小数
    for col in ['S_expert', 'S_citation', 'S_source', 'S_final']:
        df_out[col] = df_out[col].round(2)

    # 7. 写入 CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"\n整合完成！共 {len(df_out)} 个模型")
    print(f"权重: alpha={alpha}, beta={beta}, gamma={gamma}")
    print(f"输出文件: {output_csv}")

    return df_out


def run_all_versions(expert_dirs, fact_csv_base, output_dir, alpha, beta, gamma):
    """
    运行全部、zh、en 三个版本，与 ExpertCriteria/stat.py 保持一致。
    expert_dirs: list of directories containing ExpertCriteria leaderboard CSVs
    """
    for lang in [None, "zh", "en"]:
        # Expert CSV 路径（多轮次）
        # 文件名前缀与目录名前缀一致，如 1st_judge_xxx/1st_leaderboard.csv
        expert_csvs = []
        for d in expert_dirs:
            prefix = os.path.basename(d).split('_')[0]  # e.g., '1st', '2nd', '3rd'
            if lang:
                p = os.path.join(d, f"{prefix}_leaderboard_{lang}.csv")
            else:
                p = os.path.join(d, f"{prefix}_leaderboard.csv")
            if os.path.exists(p):
                expert_csvs.append(p)

        if not expert_csvs:
            print(f"跳过: 未找到任何 Expert CSV (lang={lang})")
            continue

        if lang:
            output_csv = os.path.join(output_dir, f"final_leaderboard_{lang}.csv")
        else:
            output_csv = os.path.join(output_dir, "final_leaderboard.csv")

        # FACT CSV 路径：优先尝试带语言后缀的版本
        fact_csv = fact_csv_base
        if lang:
            fact_csv_lang = fact_csv_base.replace('.csv', f'_{lang}.csv')
            if os.path.exists(fact_csv_lang):
                fact_csv = fact_csv_lang
                print(f"使用分语言 FACT CSV: {fact_csv}")

        if not os.path.exists(fact_csv):
            print(f"跳过: FACT CSV 不存在: {fact_csv}")
            continue

        print(f"\n{'='*50}")
        lang_label = lang.upper() if lang else "ALL"
        print(f"Processing: {lang_label}")
        print(f"Expert CSVs: {expert_csvs}")
        print(f"{'='*50}")
        compute_final_score(expert_csvs, fact_csv, output_csv, alpha, beta, gamma)


if __name__ == "__main__":
    eval_dir = get_eval_data_dir()
    default_expert_dirs = [
        os.path.join(eval_dir, "subjective_eval", "final_test", "1st_judge_gemini-3.1-pro-preview"),
        os.path.join(eval_dir, "subjective_eval", "final_test", "2nd_judge_gemini-3.1-pro-preview"),
        os.path.join(eval_dir, "subjective_eval", "final_test", "3rd_judge_gemini-3.1-pro-preview"),
    ]
    default_fact_csv = os.path.join(eval_dir, "subjective_eval", "final_test", "scores", "FACT", "fact_result.csv")
    default_output_dir = os.path.join(eval_dir, "subjective_eval", "final_test")

    parser = argparse.ArgumentParser(description="整合 ExpertCriteria（多轮次平均）+ FACT 评分，计算最终总分")
    parser.add_argument("--expert_dirs", type=str, nargs='+',
                        default=default_expert_dirs,
                        help="ExpertCriteria leaderboard 所在目录列表（支持多轮次取平均）")
    parser.add_argument("--fact_csv", type=str,
                        default=default_fact_csv,
                        help="FACT 评分 CSV 路径（基础路径，会自动尝试带 _zh/_en 后缀的分语言版本）")
    parser.add_argument("--output_dir", type=str,
                        default=default_output_dir,
                        help="最终评分输出目录")
    parser.add_argument("--language_filter", type=str, default=None, choices=["zh", "en"],
                        help="语言过滤: zh/en，不指定则运行全部三个版本")
    parser.add_argument("--alpha", type=float, default=0.8, help="Expert 权重")
    parser.add_argument("--beta", type=float, default=0.1, help="Citation 权重")
    parser.add_argument("--gamma", type=float, default=0.1, help="Source 权重")

    args = parser.parse_args()

    if args.language_filter:
        # 只运行指定语言
        expert_csvs = []
        for d in args.expert_dirs:
            prefix = os.path.basename(d).split('_')[0]
            p = os.path.join(d, f"{prefix}_leaderboard_{args.language_filter}.csv")
            if os.path.exists(p):
                expert_csvs.append(p)

        output_csv = os.path.join(args.output_dir, f"final_leaderboard_{args.language_filter}.csv")

        # FACT CSV：优先尝试分语言版本
        fact_csv = args.fact_csv
        fact_csv_lang = fact_csv.replace('.csv', f'_{args.language_filter}.csv')
        if os.path.exists(fact_csv_lang):
            fact_csv = fact_csv_lang
            print(f"使用分语言 FACT CSV: {fact_csv}")

        if expert_csvs and os.path.exists(fact_csv):
            compute_final_score(expert_csvs, fact_csv, output_csv, args.alpha, args.beta, args.gamma)
        else:
            print(f"错误: 输入文件不存在。Expert={expert_csvs}, FACT={fact_csv}")
    else:
        # 默认运行全部三个版本
        run_all_versions(args.expert_dirs, args.fact_csv, args.output_dir, args.alpha, args.beta, args.gamma)
