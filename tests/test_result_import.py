"""External outputs must be complete, attributable, and re-judged locally."""

import hashlib
import json
import socket
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation_toolkit.result_import import SCHEMA_VERSION, load_results
from evaluation_toolkit.subjective_eval.runtime import valid_report


class ResultImportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.questions = [{"id": "25", "question": "English question", "language": "en"},
                          {"id": "3", "question": "中文问题", "language": "zh"}]
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("Network is forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def write(self, value, filename="outputs.json"):
        path = self.base / filename
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def load(self, value, **kwargs):
        return load_results(self.write(value), model="my-lab/model-v1", questions=self.questions, **kwargs)

    def records(self):
        return {"25": {"response": "A"}, "3": {"response": "中文报告"}}

    def test_native_object_preserves_response_exactly_and_uses_question_order(self):
        text = "  报告\r\nClaim [1](https://example.org/source).\n\t"
        normalized, questions, metadata = self.load({"3": {"response": text}, "25": {"response": "0"}})
        self.assertEqual(list(normalized), ["25", "3"])
        self.assertEqual(normalized["3"]["response"], text)
        self.assertEqual(normalized["25"]["response"], "0")
        self.assertEqual(questions, self.questions)
        self.assertTrue(all(valid_report(record, "my-lab/model-v1") for record in normalized.values()))
        self.assertEqual(metadata["selected_count"], 2)
        self.assertEqual(metadata["source_count"], 2)
        self.assertEqual(metadata["omitted_ids"], [])

    def test_array_and_bom_jsonl_normalize_to_same_content(self):
        rows = [{"id": 3, "response": "中文报告"}, {"id": "25", "response": "A"}]
        array_result = self.load(rows)
        path = self.base / "outputs.jsonl"
        path.write_bytes(b"\xef\xbb\xbf" + ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
                         + "\n\n").encode("utf-8"))
        line_result = load_results(path, model="my-lab/model-v1", questions=self.questions)
        self.assertEqual(array_result[:2], line_result[:2])
        self.assertEqual(array_result[2]["normalized_fingerprint"], line_result[2]["normalized_fingerprint"])
        self.assertNotEqual(array_result[2]["source_sha256"], line_result[2]["source_sha256"])
        self.assertEqual(array_result[2]["source_format"], "json_array")
        self.assertEqual(line_result[2]["source_format"], "jsonl")

    def test_json_bom_is_accepted_and_source_hash_covers_original_bytes(self):
        path = self.base / "outputs.json"
        source = b"\xef\xbb\xbf" + json.dumps(self.records()).encode()
        path.write_bytes(source)
        result, _, metadata = load_results(path, model="my-model", questions=self.questions)
        self.assertEqual(metadata["source_sha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(metadata["schema_version"], SCHEMA_VERSION)
        self.assertEqual(len(metadata["normalized_fingerprint"]), 64)
        self.assertEqual(result["25"]["model"], "my-model")

    def test_missing_tasks_require_explicit_subset(self):
        records = {"3": {"response": "report"}}
        with self.assertRaisesRegex(ValueError, "Missing.*25"):
            self.load(records)
        normalized, selected, metadata = self.load(records, allow_subset=True)
        self.assertEqual(list(normalized), ["3"])
        self.assertEqual(selected, [self.questions[1]])
        self.assertEqual(metadata["omitted_ids"], ["25"])
        self.assertEqual(metadata["selected_count"], 1)

    def test_unknown_ids_rejected_even_in_subset_mode(self):
        for subset in (False, True):
            with self.subTest(subset=subset), self.assertRaisesRegex(ValueError, "Unknown task ID 999"):
                self.load({"999": {"response": "not an official task"}}, allow_subset=subset)

    def test_empty_subsets_and_empty_files_rejected(self):
        for value in ({}, []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "no task outputs"):
                self.load(value, allow_subset=True)
        path = self.base / "outputs.jsonl"
        path.write_text("\n  \n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            load_results(path, model="model", questions=self.questions, allow_subset=True)

    def test_duplicate_json_keys_rejected_before_overwrite(self):
        for source in ('{"25":{"response":"first"},"25":{"response":"second"}}',
                       '{"25":{"response":"first","response":"second"}}',
                       '{"25":{"response":"first","usage":{"n":1,"n":2}}}'):
            path = self.base / "duplicate.json"
            path.write_text(source, encoding="utf-8")
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "Duplicate JSON"):
                load_results(path, model="model", questions=self.questions, allow_subset=True)

    def test_duplicate_ids_in_array_jsonl_and_normalized_object_keys(self):
        rows = [{"id": "25", "response": "A"}, {"id": 25, "response": "B"}]
        with self.assertRaisesRegex(ValueError, "Duplicate result task ID"):
            self.load(rows, allow_subset=True)
        with self.assertRaisesRegex(ValueError, "Duplicate result task ID"):
            self.load({"25": {"response": "A"}, " 25 ": {"response": "B"}}, allow_subset=True)
        path = self.base / "outputs.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate result task ID"):
            load_results(path, model="model", questions=self.questions, allow_subset=True)

    def test_record_ids_must_match_dictionary_keys(self):
        with self.assertRaisesRegex(ValueError, "key and record id disagree"):
            self.load({"25": {"id": "3", "response": "A"}}, allow_subset=True)
        normalized, _, _ = self.load({"25": {"id": 25, "response": "A"}}, allow_subset=True)
        self.assertEqual(normalized, {"25": {"model": "my-lab/model-v1", "response": "A"}})

    def test_invalid_record_ids_rejected(self):
        for task_id in (None, True, False, "", " \t ", 25.0, [], {}):
            with self.subTest(task_id=task_id), self.assertRaisesRegex(ValueError, "task ID"):
                self.load([{"id": task_id, "response": "A"}], allow_subset=True)
        with self.assertRaisesRegex(ValueError, "task ID"):
            self.load([{"response": "A"}], allow_subset=True)

    def test_non_object_records_and_malformed_json_rejected(self):
        for value in ({"25": "A"}, ["A"], 25, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(value, allow_subset=True)
        path = self.base / "outputs.jsonl"
        path.write_text('{"id":"25","response":"A"}\n{bad JSON}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 2"):
            load_results(path, model="model", questions=self.questions, allow_subset=True)
        path.write_text('[{"id":"25","response":"A"}]', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 1 must be a JSON object"):
            load_results(path, model="model", questions=self.questions, allow_subset=True)

    def test_non_utf8_input_rejected(self):
        path = self.base / "outputs.json"
        path.write_bytes(b'\xff\xfe{\x00}\x00')
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            load_results(path, model="model", questions=self.questions)

    def test_missing_empty_and_non_text_responses_rejected(self):
        for response in (None, "", " \n\t", 0, False, [], {"text": "A"}):
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, "nonempty textual response"):
                self.load({"25": {"response": response}}, allow_subset=True)
        with self.assertRaisesRegex(ValueError, "nonempty textual response"):
            self.load({"25": {}}, allow_subset=True)

    def test_known_aliases_are_consistent_but_mixed_or_invalid_models_rejected(self):
        records = {"25": {"model": "gpt-4o", "response": "A"},
                   "3": {"model": "openai/gpt-4o", "response": "B"}}
        normalized, _, _ = load_results(self.write(records), model="gpt-4o", questions=self.questions)
        self.assertEqual({record["model"] for record in normalized.values()}, {"gpt-4o"})
        for model in (None, "", 12, "another-model", "my-lab/model-v2", " my-lab/model-v1 "):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "model does not match"):
                self.load({"25": {"model": model, "response": "A"}}, allow_subset=True)

    def test_completion_flags_are_validated_before_discarding(self):
        failure_flags = [{"status": "failed"}, {"generation_status": "running"}, {"status": ""},
                         {"success": False}, {"completed": False}, {"failed": True}, {"is_error": True},
                         {"truncated": True}, {"is_truncated": "false"}, {"finish_reason": "length"},
                         {"finish_reason": "tool_calls"}, {"stop_reason": "max_tokens"},
                         {"finish_reason": "content_filter"}, {"error": "timeout"},
                         {"errors": ["timeout"]}, {"incomplete_details": {"reason": "max_output_tokens"}},
                         {"generation_error": "error"}, {"citation_mapping_error": "unmapped citations"}]
        for flags in failure_flags:
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.load({"25": {"response": "partial output", **flags}}, allow_subset=True)
        normalized, _, _ = self.load({"25": {"response": "complete", "status": "completed", "success": True,
                                            "finish_reason": "STOP", "stop_reason": "end_turn", "failed": False,
                                            "truncated": False, "error": None}}, allow_subset=True)
        self.assertEqual(normalized["25"], {"model": "my-lab/model-v1", "response": "complete"})

    def test_literal_failure_words_in_response_are_not_guessed_to_be_errors(self):
        # These can be valid answers; completion metadata is the reliable signal.
        text = "The failure rate is 0; the error is within tolerance."
        normalized, _, _ = self.load({"25": {"response": text}}, allow_subset=True)
        self.assertEqual(normalized["25"]["response"], text)

    def test_scores_fingerprints_credentials_and_unrecognized_fields_are_discarded(self):
        record = {"response": "answer", "judge_result": {"correct": True}, "score": 100,
                  "total_score": 100, "dimensions_breakdown": [{"score": 100}],
                  "request_fingerprint": "forged", "judge_fingerprint": "forged",
                  "report_fingerprint": "forged", "api_key": "secret", "citations": [{"correct": True}]}
        normalized, _, metadata = self.load({"25": record}, allow_subset=True)
        self.assertEqual(normalized["25"], {"model": "my-lab/model-v1", "response": "answer"})
        serialized = json.dumps(metadata)
        for value in ("answer", "secret", "forged", "api_key"):
            self.assertNotIn(value, serialized)
        plain, _, plain_metadata = self.load({"25": {"response": "answer"}}, allow_subset=True)
        self.assertEqual(normalized, plain)
        self.assertEqual(metadata["normalized_fingerprint"], plain_metadata["normalized_fingerprint"])

    def test_usage_and_provider_citation_metadata_are_preserved_without_rendering(self):
        metadata = {"usage": {"prompt_tokens": 12, "completion_tokens": 7},
                    "provider_citations": [{"url": "https://example.org", "start_index": 0, "end_index": 6}],
                    "provider_raw_citations": [{"type": "url_citation", "url": "https://example.org"}]}
        normalized, _, _ = self.load({"25": {"response": "Claim [1].", **metadata}}, allow_subset=True)
        self.assertEqual(normalized["25"], {"model": "my-lab/model-v1", "response": "Claim [1].", **metadata})
        normalized, _, _ = self.load({"25": {"response": "A", "usage": None}}, allow_subset=True)
        self.assertIsNone(normalized["25"]["usage"])

    def test_invalid_optional_metadata_types_rejected(self):
        for field, value in (("usage", []), ("usage", 3), ("provider_citations", "url"),
                             ("provider_raw_citations", True)):
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                self.load({"25": {"response": "A", field: value}}, allow_subset=True)

    def test_nonfinite_values_rejected_even_in_discarded_fields(self):
        for value in ("NaN", "Infinity", "-Infinity", "1e999"):
            path = self.base / "outputs.json"
            path.write_text('{"25":{"response":"A","discarded":{"n":' + value + '}}}', encoding="utf-8")
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonfinite"):
                load_results(path, model="model", questions=self.questions, allow_subset=True)

    def test_question_ids_and_arguments_are_validated(self):
        path = self.write(self.records())
        for questions in ([], {}, [None], [{"id": "25"}, {"id": 25}], [{"id": True}]):
            with self.subTest(questions=questions), self.assertRaises(ValueError):
                load_results(path, model="model", questions=questions)
        for model in (None, "", " ", " model ", 4):
            with self.subTest(model=model), self.assertRaises(ValueError):
                load_results(path, model=model, questions=self.questions)
        with self.assertRaisesRegex(ValueError, "boolean"):
            load_results(path, model="model", questions=self.questions, allow_subset="yes")

    def test_import_does_not_write_files_or_mutate_questions(self):
        path = self.write(self.records())
        before_bytes = path.read_bytes()
        before_questions = json.dumps(self.questions)
        before_files = set(self.base.iterdir())
        load_results(path, model="model", questions=self.questions)
        self.assertEqual(path.read_bytes(), before_bytes)
        self.assertEqual(json.dumps(self.questions), before_questions)
        self.assertEqual(set(self.base.iterdir()), before_files)


if __name__ == "__main__":
    unittest.main()
