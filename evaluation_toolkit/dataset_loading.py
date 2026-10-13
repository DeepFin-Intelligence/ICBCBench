"""Hugging Face loading shared by objective and subjective entry points."""
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


ICBCBENCH_REPO = "DeepFin-Intelligence/ICBCBench"
CONFIG_FILES = {
    "objective_questions": "objective_questions_public_80.json",
    "subjective_questions": "subjective_questions_public_40.json",
}


def add_hf_dataset_arguments(parser):
    parser.add_argument("--dataset_config", default=None, help="HF configuration; inferred for ICBCBench")
    parser.add_argument("--dataset_split", default=None, help="HF split; ICBCBench defaults to train, other datasets to test")
    parser.add_argument("--dataset_revision", default=None, help="Optional HF commit, tag, or branch")


def load_hf_dataset(options, task):
    repo = getattr(options, "dataset", None)
    if not repo:
        raise ValueError("Provide --local_dataset or --dataset")
    config = getattr(options, "dataset_config", None)
    split = getattr(options, "dataset_split", None)
    revision = getattr(options, "dataset_revision", None)
    options._hf_dataset_path = None

    if repo == ICBCBENCH_REPO:
        expected_config = f"{task}_questions"
        config = config or expected_config
        if config in CONFIG_FILES and config != expected_config:
            raise ValueError(f"{task} evaluation requires ICBCBench configuration {expected_config}, got {config}")
        split = split or "train"
        # One snapshot binds questions and attachments to the same resolved commit.
        snapshot = Path(snapshot_download(
            repo_id=repo, repo_type="dataset", revision=revision,
            allow_patterns=["README.md", "data/**"],
        ))
        options._hf_dataset_revision = snapshot.name
        dataset = load_dataset(str(snapshot), name=config, split=split)
        if config in CONFIG_FILES:
            options._hf_dataset_path = str(snapshot / "data" / CONFIG_FILES[config])
    else:
        kwargs = {"split": split or "test"}
        if config is not None:
            kwargs["name"] = config
        if revision is not None:
            kwargs["revision"] = revision
        dataset = load_dataset(repo, **kwargs)
    return dataset.to_dict()
