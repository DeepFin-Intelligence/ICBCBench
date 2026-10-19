"""Keep custom model namespaces intact when producing objective metric CSVs."""

import contextlib
import csv
import io
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation_toolkit.objective_eval import judge, metrics


class ObjectiveModelIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("No network")))
        self.questions = [{"id": "1", "question": "Question?", "answer": "A", "language": "en"},
                          {"id": "2", "question": "问题？", "answer": "B", "language": "zh"}]

    def save(self, name, value):
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def record(self, question, model="my-lab/model-v1"):
        record = {"response": "A", "judge_model": "gpt-5.4",
                  "judge_result": {"correct": "yes", "confidence": 100,
                                   "correct_answer": "A", "model_answer": "A", "reasoning": "fixture"}}
        if model is not None:
            record["model"] = model
        record["judge_fingerprint"] = metrics.judge_fingerprint(question, record, "gpt-5.4")
        return record

    def rows(self, name):
        with (self.base / name).open(encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    def metric_rows(self, records, name="judged_model-v1.json"):
        query = self.save("questions.json", self.questions)
        self.save("judged/" + name, records)
        status = metrics.dump_all_metrics("gpt-5.4", None, judged_dir=self.base / "judged",
                                          query_file=query, results_dir=self.base / "metrics")
        return status, self.rows("metrics/evaluation_results.csv")

    def test_both_objective_csv_paths_keep_full_model_name(self):
        records = {question["id"]: self.record(question) for question in self.questions}
        status, rows = self.metric_rows(records)
        self.assertEqual(status, 0)
        self.assertEqual(rows[0]["Model"], "my-lab/model-v1")
        with patch.object(judge, "args", SimpleNamespace(predictions="model-v1.json", judge="gpt-5.4",
                                                        metrics_csv=str(self.base / "judge.csv")), create=True):
            judge.dump_metrics(records, len(records))
        self.assertEqual(self.rows("judge.csv")[0]["Model"], "my-lab/model-v1")

    def test_legacy_records_without_model_use_filename(self):
        records = {question["id"]: self.record(question, None) for question in self.questions}
        status, rows = self.metric_rows(records, "judged_legacy_model_name.json")
        self.assertEqual(status, 0)
        self.assertEqual(rows[0]["Model"], "legacy_model_name")
        self.assertEqual(metrics.prediction_model_name({}, "legacy_empty"), "legacy_empty")

    def test_same_basename_in_different_namespaces_remains_separate_rows(self):
        first = {question["id"]: self.record(question, "lab-a/model-v1") for question in self.questions}
        second = {question["id"]: self.record(question, "lab-b/model-v1") for question in self.questions}
        self.save("judged/judged_file_a.json", first)
        _, rows = self.metric_rows(second, "judged_file_b.json")
        self.assertEqual({row["Model"] for row in rows}, {"lab-a/model-v1", "lab-b/model-v1"})

    def test_mixed_model_names_are_rejected_instead_of_combined(self):
        records = {"1": self.record(self.questions[0], "lab-a/model-v1"),
                   "2": self.record(self.questions[1], "lab-b/model-v1")}
        with self.assertRaisesRegex(ValueError, "multiple models"):
            self.metric_rows(records)
        self.assertFalse((self.base / "metrics/evaluation_results.csv").exists())

    def test_known_aliases_use_stable_canonical_label_when_mixed(self):
        first = {"1": {"model": "gpt-4o"}, "2": {"model": "openai/gpt-4o"}}
        self.assertEqual(metrics.prediction_model_name(first, "ignored"), "openai/gpt-4o")
        self.assertEqual(metrics.prediction_model_name(dict(reversed(list(first.items()))), "ignored"), "openai/gpt-4o")
        self.assertEqual(metrics.prediction_model_name({"1": {"model": "gpt-4o"}}, "ignored"), "gpt-4o")
        self.assertEqual(metrics.prediction_model_name({"1": {"model": "vendor/gpt-4o"}}, "ignored"), "vendor/gpt-4o")

    def test_invalid_explicit_model_names_cannot_silently_fall_back(self):
        for value in (None, "", " ", 123, False, " model "):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Prediction model names"):
                metrics.prediction_model_name({"1": {"model": value}}, "filename")

    def test_judge_keeps_full_identity_when_every_judgment_fails(self):
        query = self.save("questions.json", self.questions)
        predictions = self.save("model-v1.json", {question["id"]: {"model": "my-lab/model-v1", "response": "A"}
                                                  for question in self.questions})
        options = SimpleNamespace(local_dataset=str(query), dataset=None, predictions=str(predictions),
                                  num_workers=1, judge="gpt-5.4", output_dir=str(self.base / "judged"),
                                  metrics_csv=str(self.base / "judge.csv"))
        with patch.object(judge, "judge_all_responses", return_value=[]):
            self.assertEqual(judge.main(options), 1)
        row = self.rows("judge.csv")[0]
        self.assertEqual(row["Model"], "my-lab/model-v1")
        self.assertEqual(row["Evaluated"], "0")

    def test_judge_rejects_mixed_models_before_calls(self):
        query = self.save("questions.json", self.questions)
        predictions = self.save("model-v1.json", {"1": {"model": "lab-a/model-v1", "response": "A"},
                                                   "2": {"model": "lab-b/model-v1", "response": "B"}})
        options = SimpleNamespace(local_dataset=str(query), dataset=None, predictions=str(predictions),
                                  num_workers=1, judge="gpt-5.4", output_dir=str(self.base / "judged"),
                                  metrics_csv=str(self.base / "judge.csv"))
        with patch.object(judge, "judge_all_responses") as calls:
            self.assertEqual(judge.main(options), 1)
        calls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
