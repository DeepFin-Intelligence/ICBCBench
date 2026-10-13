"""Exercise the published HF layout and entry points without live services."""
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import datasets
import pymupdf

from evaluation_toolkit import dataset_loading
from evaluation_toolkit.objective_eval import predict, judge
from evaluation_toolkit.subjective_eval import write


ROOT = Path(__file__).resolve().parents[1]


class HFDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="icbc-hf-test-")
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        version = json.loads((ROOT / "data/huggingface_version.json").read_text())
        self.snapshot = self.base / "snapshots" / version['release_id']
        self.snapshot.mkdir(parents=True)
        for item in version["files"]:
            destination = self.snapshot / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / item["path"], destination)
        # Reproduce real HF Hub cache files, which point into a shared blobs dir.
        blobs = self.base / "blobs"
        blobs.mkdir()
        for item in version["files"]:
            source = self.snapshot / item["path"]
            blob = blobs / item["sha256"]
            source.rename(blob)
            source.symlink_to(blob)
        (self.snapshot / "README.md").write_text(
            "---\nconfigs:\n"
            "- config_name: objective_questions\n  data_files:\n"
            "  - data/objective_questions_public_80.json\n"
            "- config_name: subjective_questions\n  data_files:\n"
            "  - data/subjective_questions_public_40.json\n---\n",
            encoding="utf-8",
        )
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Live network forbidden")))
        self.stack.enter_context(patch.object(datasets.config, "HF_DATASETS_CACHE", self.base / "cache"))
        self.download = self.stack.enter_context(patch.object(
            dataset_loading, "snapshot_download", return_value=str(self.snapshot)))

    def options(self, **kwargs):
        values = dict(local_dataset=None, dataset=dataset_loading.ICBCBENCH_REPO,
                      dataset_config=None, dataset_split=None, dataset_revision=self.snapshot.name,
                      model="gpt-4o", max_samples=None, num_workers=1, temperature=0,
                      max_completion_tokens=128, output_dir=str(self.base / "output"))
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_objective_hf_loads_all_rows_and_binds_source_references_to_snapshot(self):
        source = self.snapshot / 'data/objective_questions_public_80.json'
        changed = json.loads(source.read_text())
        changed_question = next(row for row in changed if row['id'] == '25')
        changed_question['attachment'][0]['pdf_page'] = 7
        source.write_text(json.dumps(changed))
        options = self.options()
        rows = predict.load_questions(options, ("id", "language", "question", "answer"))
        self.assertEqual(len(rows), 80)
        self.assertEqual([row["id"] for row in rows if isinstance(row["answer"], list)], ["6", "21", "27"])
        question = next(row for row in rows if row["id"] == "25")
        predict.args = options
        with pymupdf.open() as document:
            for index in range(8):
                page = document.new_page(width=64, height=64)
                page.insert_text((8, 25), f'Page {index + 1}', fontsize=8)
            response = Mock(content=document.tobytes())
        with patch.object(predict.requests, 'get', return_value=response) as download:
            messages = predict.format_message(question)
        images = [block for block in messages[1]['content'] if block['type'] == 'image_url']
        self.assertEqual(len(images), 2)
        self.assertEqual(download.call_count, 2)
        for image in images:
            png = base64.b64decode(image['image_url']['url'].split(',', 1)[1])
            self.assertEqual(png[:8], b'\x89PNG\r\n\x1a\n')
        text = messages[1]['content'][0]['text']
        for attachment in question['attachment']:
            self.assertIn(attachment['source_url'], text)
            self.assertIn(str(attachment['pdf_page']), text)
        self.assertEqual(question['attachment'][0]['pdf_page'], 7)
        local = self.options(local_dataset=str(ROOT / "data/objective_questions_public_80.json"))
        local_question = next(row for row in json.loads(Path(local.local_dataset).read_text()) if row['id'] == '25')
        self.assertNotEqual(predict.request_fingerprint(question, options), predict.request_fingerprint(local_question, local))
        self.assertIsNone(options.local_dataset)
        self.assertEqual(options._hf_dataset_revision, self.snapshot.name)
        self.download.assert_called_once_with(
            repo_id=dataset_loading.ICBCBENCH_REPO, repo_type="dataset",
            revision=self.snapshot.name, allow_patterns=["README.md", "data/**"])

    def test_hf_subjective_generation_loads_and_saves_all_40_topics(self):
        options = self.options()
        with patch.object(write, "attempt_query", side_effect=lambda q: (q["id"], "Report " + q["question"], None)) as generate:
            self.assertEqual(write.main(options), 0)
        reports = json.loads((self.base / "output/reports_gpt-4o.json").read_text())
        self.assertEqual(len(reports), 40)
        self.assertEqual(generate.call_count, 40)
        self.assertIn("36", reports)

    def test_hf_objective_judging_loads_all_80_questions_and_preserves_list_answers(self):
        rows = json.loads((ROOT / "data/objective_questions_public_80.json").read_text())
        predictions = self.base / "predictions.json"
        predictions.write_text(json.dumps({q["id"]: {"model": "gpt-4o", "response": "synthetic answer"} for q in rows}))
        options = self.options(predictions=str(predictions), judge="gpt-5.4", metrics_csv=str(self.base / "metrics.csv"))
        seen = []

        def judge_all(questions, predictions, on_result):
            for question in questions:
                seen.append(question)
                on_result((question["id"], {"judge_result": {"correct": "yes", "confidence": 50,
                           "correct_answer": question["answer"], "model_answer": "synthetic answer", "reasoning": "offline"}}))

        with patch.object(judge, "judge_all_responses", side_effect=judge_all):
            self.assertEqual(judge.main(options), 0)
        self.assertEqual(len(seen), 80)
        self.assertEqual(sum(isinstance(q["answer"], list) for q in seen), 3)
        self.assertEqual(len(json.loads((self.base / "output/judged_predictions.json").read_text())), 80)

    def test_local_sources_do_not_download_or_retain_old_snapshot_attachment_paths(self):
        options = self.options(local_dataset=str(ROOT / "data/objective_questions_public_80.json"))
        options._hf_dataset_path = "stale/remote.json"
        self.assertEqual(len(predict.load_questions(options)), 80)
        self.assertIsNone(options._hf_dataset_path)
        subjective = self.options(local_dataset=str(ROOT / "data/subjective_questions_public_40.json"), max_samples=1)
        with patch.object(write, "attempt_query", return_value=("1", "local report", None)):
            self.assertEqual(write.main(subjective), 0)
        self.download.assert_not_called()

    def test_other_hf_repositories_keep_test_default_and_accept_explicit_options(self):
        options = self.options(dataset="example/questions", dataset_revision=None)
        with patch.object(dataset_loading, "load_dataset") as load:
            load.return_value.to_dict.return_value = {"id": ["1"]}
            dataset_loading.load_hf_dataset(options, "objective")
            load.assert_called_once_with("example/questions", split="test")
            load.reset_mock()
            options.dataset_config, options.dataset_split, options.dataset_revision = "custom", "validation[:2]", "commit"
            dataset_loading.load_hf_dataset(options, "objective")
            load.assert_called_once_with("example/questions", name="custom", split="validation[:2]", revision="commit")
        self.download.assert_not_called()

    def test_invalid_revision_and_wrong_task_config_fail_without_fallback(self):
        with patch.object(dataset_loading, "snapshot_download", side_effect=RuntimeError("invalid revision")), patch.object(dataset_loading, "load_dataset") as load:
            with self.assertRaisesRegex(RuntimeError, "invalid revision"):
                predict.load_questions(self.options(dataset_revision="missing"))
            load.assert_not_called()
        with self.assertRaisesRegex(ValueError, "objective_questions"):
            predict.load_questions(self.options(dataset_config="subjective_questions"))
        self.download.assert_not_called()

    def test_prepared_release_manifest_preserves_historical_source_and_current_hashes(self):
        version = json.loads((ROOT / "data/huggingface_version.json").read_text())
        self.assertEqual(version['source_revision'], 'c1cbb5ed75daf6225035dbd89737aef1da86557f')
        self.assertEqual(version['release_id'], self.snapshot.name)
        self.assertEqual(version['status'], 'github_branch_hf_unpublished')
        self.assertNotIn('revision', version)
        self.assertEqual({item['path'] for item in version['files']}, {
            'data/objective_questions_public_80.json', 'data/subjective_questions_public_40.json',
            'data/LICENSE', 'data/DATA_LICENSE.md', 'data/THIRD_PARTY_SOURCES.md'})
        for item in version["files"]:
            with self.subTest(path=item["path"]):
                content = (ROOT / item["path"]).read_bytes()
                self.assertEqual(len(content), item["size"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])
        # Historical provenance is recorded as metadata. The third-party image
        # bytes are deliberately absent from the public repository and fixtures.
        self.assertEqual(version['source_files'], [
            {'path': 'data/attachments/en_25_CIB.png', 'size': 163080,
             'sha256': '23c2dbf081be5b0d68bb6fd325d2b53a4ea5041725ea221425588171165ff4d3'},
            {'path': 'data/attachments/en_25_CIB占比.png', 'size': 165651,
             'sha256': '31f5c75889df912511301f1e6acd3a4fedfb1d94ddd083757f05ec40182a4374'},
            {'path': 'data/objective_questions_public_80.json', 'size': 103902,
             'sha256': 'd8c0e465fc38cc8c9c0c91d2e02c6eaf22ee07b8c79d94da5b9d3b7794e74dcb'},
            {'path': 'data/subjective_questions_public_40.json', 'size': 386289,
             'sha256': '1d630d72d1bb4b7ba5cbc050cb4c224a76396e0b5c39c499897e87f411e36897'},
        ])
        rows = json.loads((ROOT / 'data/objective_questions_public_80.json').read_text())
        references = next(row for row in rows if row['id'] == '25')['attachment']
        self.assertEqual([item['pdf_page'] for item in references], [8, 6])
        self.assertTrue(all('file_path' not in item for item in references))
        self.assertEqual(list((ROOT / 'data').rglob('*.png')), [])
        self.assertEqual(list((self.snapshot / 'data').rglob('*.png')), [])


if __name__ == "__main__":
    unittest.main()
