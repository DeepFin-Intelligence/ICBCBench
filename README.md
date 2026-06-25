# FinHLE Evaluation Toolkit

本目录整理自 `eval/`，包含 FinHLE / FinResearch Benchmark 中可独立分享的核心评估代码。

## 目录结构

```
evaluation_toolkit/
├── objective_eval/                 # 客观题评估
│   ├── predict.py                      # 异步调用模型答题
│   ├── judge.py                        # 调用 Judge 模型批分
│   ├── metrics.py                      # 指标计算（Accuracy / Calibration Error）
│   └── subset_metrics.py               # v2 客观题子集统计
├── subjective_eval/                # 主观题（研究报告）评估
│   ├── write.py                        # 调用模型撰写报告
│   ├── ExpertCriteria/                 # 专家评分标准评分
│   │   ├── score_by_static_criteria.py
│   │   └── stat.py                     # 将 jsonl 评分结果合并为 leaderboard CSV
│   ├── FACT/                           # 引用提取、去重、爬取、验证、统计
│   │   ├── extract.py
│   │   ├── deduplicate.py
│   │   ├── scrape.py
│   │   ├── validate.py
│   │   ├── stat.py
│   │   ├── authority.py
│   │   ├── time_decay.py
│   │   └── utils.py
│   └── scoring/                        # 分数汇总与统计
│       ├── compute_final_score.py      # 整合 ExpertCriteria + FACT 计算最终总分
│       └── fact_stat.py                # FACT 指标统计
├── DR_clients/                     # Deep Research 模型客户端
├── model_clients.py                # API 客户端集合
└── utils.py                        # JSON / PDF 等工具函数
```

## 安装依赖

```bash
pip install -r evaluation_toolkit/requirements.txt
```

> 注意：原项目根目录的 `requirements.txt` 也包含相关依赖，二者可任选其一安装。

## 运行方式

所有脚本均可直接运行，也可以通过模块方式运行。

### 直接运行（推荐）

```bash
cd FinHLE
python evaluation_toolkit/objective_eval/predict.py \
    --local_dataset "..." --model gpt-4o --num_workers 2
```

### 模块方式运行

```bash
cd FinHLE
python -m evaluation_toolkit.objective_eval.predict \
    --local_dataset "..." --model gpt-4o --num_workers 2
```

## 数据路径说明

脚本中的数据路径默认基于项目根目录下的 `data/` 目录。可通过环境变量统一修改数据根目录：

```bash
export EVAL_DATA_DIR="/path/to/your/eval_data"
```

- 客观题数据：`$EVAL_DATA_DIR/objective_eval/v2_obj_public-private_test/`
- 主观题数据：`$EVAL_DATA_DIR/subjective_eval/final_test/`

所有原 `D:\AI_Lab_ICBC\FinHLE\...` 绝对路径已替换为基于 `EVAL_DATA_DIR` 的相对路径。如需进一步自定义，可通过各脚本的命令行参数覆盖。

## 环境变量

API Key 通过 `.env` 文件读取，请在项目根目录创建 `.env` 并配置：

```bash
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
KIMI_API_KEY=...
# 其他需要的 key
```

具体 key 名请参考 `model_clients.py`。

## 注意事项

- `subjective_eval/FACT/scrape.py` 依赖 `curl_cffi`。
