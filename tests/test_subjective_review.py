"""Offline checks for rubric correctness, durable results, and real SDK shapes."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation_toolkit.subjective_eval import write
from evaluation_toolkit.subjective_eval.runtime import validate_questions
from evaluation_toolkit.subjective_eval.ExpertCriteria import score_by_criteria as expert
from evaluation_toolkit.subjective_eval.ExpertCriteria import stat as expert_stat
from evaluation_toolkit.subjective_eval.scoring import compute_final_score as final
from evaluation_toolkit.DR_clients.GeminiDR import GeminiDeepResearch
from evaluation_toolkit.DR_clients.OpenaiDR import OpenaiDeepResearch
from evaluation_toolkit.DR_clients.QwenDR import QwenDeepResearch, _convert_to_qwen_multimodal
from evaluation_toolkit.DR_clients.TavilyResearch import TavilyDeepResearch


RUBRIC = """### 1. Quality (100 points)
#### (1) Evidence (40 points)
#### (2) Reasoning (60 points)"""


from evaluation_toolkit.subjective_eval.scoring.provenance import (question_fingerprint, task_set_fingerprint, report_fingerprint, report_set_fingerprint)


def dimensions(criteria=RUBRIC, fraction=.8):
    return [{"dimension_id": key, "score": cap * fraction}
            for key, cap in expert.parse_criteria_dimensions(criteria).items()]


class SubjectiveReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_public_rubrics_with_explicit_question_36_correction(self):
        questions = json.loads((ROOT / "data/subjective_questions_public_40.json").read_text())
        for question in questions:
            criteria = question["expert_evaluation_criteria"]
            caps = expert.parse_criteria_dimensions(criteria)
            self.assertEqual(sum(caps.values()), 106 if question["id"] == "36" else 100)
            self.assertEqual(expert.validate_dimension_scores(dimensions(criteria, 1), criteria), 100)
            self.assertEqual(expert.validate_dimension_scores(dimensions(criteria, .5), criteria), 50)

    def test_question_36_preserves_raw_scores_and_marks_correction(self):
        criteria = json.loads((ROOT / "data/subjective_questions_public_40.json").read_text())[35]["expert_evaluation_criteria"]
        scores = dimensions(criteria, 1)
        with patch.object(expert, "call_llm", return_value=json.dumps(scores)) as request:
            result = expert.evaluate_report_with_whole_criteria("report", criteria, "zh")
        self.assertEqual(result["raw_total_score"], 106)
        self.assertEqual(result["total_score"], 100)
        self.assertEqual(result["dimensions_breakdown"], scores)
        self.assertEqual(result["scoring_adjustments"][0]["factor"], 30 / 36)
        self.assertIn("Do not normalize", request.call_args.args[0][0]["content"])
        with self.assertRaises(ValueError):
            expert.parse_criteria_dimensions(criteria + " Modified rubric")
        scores[-1]["score"] = 999
        with self.assertRaises(ValueError):
            expert.validate_dimension_scores(scores, criteria)

    def test_fractional_dimension_scores_are_preserved(self):
        scores = [{"dimension_id": "1.1", "score": 30.25}, {"dimension_id": "1.2", "score": 40.5}]
        with patch.object(expert, "call_llm", return_value=json.dumps(scores)):
            result = expert.evaluate_report_with_whole_criteria("report", RUBRIC, "en")
        self.assertEqual(result["total_score"], 70.75)

    def test_rejects_empty_duplicate_missing_out_of_range_and_nonfinite_scores(self):
        bad = [[], {}, [{"score": 80}], [{"dimension_id": "1.1", "score": 20}],
               [{"dimension_id": "1.1", "score": 20}, {"dimension_id": "1.1", "score": 20}],
               [{"dimension_id": "1.1", "score": 41}, {"dimension_id": "1.2", "score": 20}],
               [{"dimension_id": "1.1", "score": math.nan}, {"dimension_id": "1.2", "score": 20}],
               [{"dimension_id": "1.1", "score": True}, {"dimension_id": "1.2", "score": 20}]]
        for scores in bad:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                expert.validate_dimension_scores(scores, RUBRIC)

    def test_unstructured_rubric_still_checks_score_schema(self):
        self.assertEqual(expert.validate_dimension_scores([{"dimension_id": "1.1", "score": 80.5}], "Custom criteria"), 80.5)
        with self.assertRaises(ValueError):
            expert.validate_dimension_scores([{"dimension_id": "1.1", "score": 101}], "Custom criteria")

    def test_partial_structured_rubric_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            expert.parse_criteria_dimensions("### 1. Quality (100 points)\n#### (1) Evidence (40 points)")

    def write_args(self):
        dataset = self.base / "questions.json"
        dataset.write_text(json.dumps([{"id": "1", "language": "en", "question": "topic"}]))
        return SimpleNamespace(local_dataset=str(dataset), max_samples=None, num_workers=1,
                               model="gpt-4o", output_dir=str(self.base / "reports"))

    def expert_args(self):
        questions = self.base / "questions.json"
        questions.write_text(json.dumps([{"id": "1", "language": "en", "question": "topic", "expert_evaluation_criteria": RUBRIC}]))
        reports = self.base / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "reports_gpt-4o.json").write_text(json.dumps({"1": {"model": "gpt-4o", "response": "report"}}))
        return SimpleNamespace(question_path=str(questions), report_model="gpt-4o", judge="gpt-5.4",
                               report_dir=str(reports), output_dir=str(self.base / "scores"),
                               max_workers=1, max_samples=None)

    def test_report_saved_before_next_result_or_interrupt(self):
        args = self.write_args()
        output = Path(args.output_dir) / "reports_gpt-4o.json"
        def completions(futures):
            yield futures[0]
            self.assertEqual(json.loads(output.read_text())["1"]["response"], "saved report")
            raise KeyboardInterrupt
        with patch.object(write, "attempt_query", return_value=("1", "saved report", None)), patch.object(write, "as_completed", side_effect=completions):
            with self.assertRaises(KeyboardInterrupt):
                write.main(args)
        self.assertTrue(output.is_file())

    def test_score_saved_before_next_result_or_interrupt(self):
        args = self.expert_args()
        output = Path(args.output_dir) / "judge_gpt-5.4/scores_gpt-4o.jsonl"
        def completions(futures):
            yield next(iter(futures))
            self.assertEqual(json.loads(output.read_text())["total_score"], 80)
            raise KeyboardInterrupt
        with patch.object(expert, "call_llm", return_value=json.dumps(dimensions())), patch.object(expert.concurrent.futures, "as_completed", side_effect=completions):
            with self.assertRaises(KeyboardInterrupt):
                expert.main(args)
        self.assertTrue(output.is_file())

    def test_invalid_cached_score_retried_and_replaced(self):
        args = self.expert_args()
        output = Path(args.output_dir) / "judge_gpt-5.4/scores_gpt-4o.jsonl"
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps({"task_id": "1", "model": "gpt-4o", "total_score": 0, "dimensions_breakdown": []}) + "\n")
        with patch.object(expert, "call_llm", return_value=json.dumps(dimensions())) as request:
            self.assertEqual(expert.main(args), 0)
            request.assert_called_once()
            self.assertEqual(expert.main(args), 0)
            request.assert_called_once()
        self.assertEqual(len(output.read_text().splitlines()), 1)
        self.assertEqual(json.loads(output.read_text())["total_score"], 80)

    def test_report_cache_cannot_be_reused_for_another_provider(self):
        args = self.write_args()
        output = Path(args.output_dir) / "reports_gpt-4o.json"
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps({"1": {"model": "another/gpt-4o", "response": "report"}}))
        with self.assertRaisesRegex(ValueError, "different model"):
            write.main(args)

    def test_empty_or_mismatched_report_cannot_be_scored(self):
        args = self.expert_args()
        report = Path(args.report_dir) / "reports_gpt-4o.json"
        report.write_text(json.dumps({"1": {"model": "gpt-4o", "response": "   "}}))
        with patch.object(expert, "call_llm") as request:
            self.assertEqual(expert.main(args), 1)
            request.assert_not_called()

    def test_gemini_real_sdk_interaction_shape(self):
        from google.genai._gaos.types import interactions
        response = interactions.Interaction.model_validate({
            "id": "audit", "status": "completed", "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "SDK-shaped answer"}]}]})
        model = GeminiDeepResearch.__new__(GeminiDeepResearch)
        model.agent_name = "test-agent"
        model.client = SimpleNamespace(interactions=SimpleNamespace(
            create=Mock(return_value=response), get=Mock(return_value=response)))
        self.assertEqual(model.run_research("system", "task")["content"], "SDK-shaped answer")

    def test_gemini_cancelled_and_empty_results_are_failures(self):
        for status in ("cancelled", "requires_action", "completed"):
            model = GeminiDeepResearch.__new__(GeminiDeepResearch)
            model.agent_name = "test-agent"
            response = SimpleNamespace(id="1", status=status, output_text="", steps=[])
            model.client = SimpleNamespace(interactions=SimpleNamespace(create=Mock(return_value=response), get=Mock(return_value=response)))
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                model.run_research("system", "task")

    def test_gemini_images_are_forwarded(self):
        model = GeminiDeepResearch.__new__(GeminiDeepResearch)
        model.agent_name = "test-agent"
        response = SimpleNamespace(id="1", status="completed", output_text="answer")
        create = Mock(return_value=response)
        model.client = SimpleNamespace(interactions=SimpleNamespace(create=create, get=Mock(return_value=response)))
        model.run_research("system", "task", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}])
        self.assertEqual(create.call_args.kwargs["input"][1], {"type": "image", "mime_type": "image/png", "data": "abc"})

    def test_research_polling_has_a_deadline(self):
        for cls in (GeminiDeepResearch, TavilyDeepResearch):
            model = cls.__new__(cls)
            model.timeout_seconds = .01
            model.poll_interval = 0
            model.agent_name = "test-agent"
            response = SimpleNamespace(id="1", status="in_progress")
            model.client = SimpleNamespace(
                interactions=SimpleNamespace(create=Mock(return_value=response), get=Mock(return_value=response)),
                research=Mock(return_value={"request_id": "1"}), get_research=Mock(return_value={"status": "pending"}))
            with self.subTest(provider=cls.__name__), self.assertRaises(TimeoutError):
                model.run_research("system", "task")

    def test_tavily_empty_result_and_missing_request_id_fail(self):
        for initial in ({}, {"request_id": "1"}):
            model = TavilyDeepResearch.__new__(TavilyDeepResearch)
            model.poll_interval = 0
            model.client = SimpleNamespace(research=Mock(return_value=initial), get_research=Mock(return_value={"status": "completed", "content": ""}))
            with self.subTest(initial=initial), self.assertRaises(RuntimeError):
                model.run_research("system", "task")

    def test_tavily_cannot_silently_drop_images(self):
        model = TavilyDeepResearch.__new__(TavilyDeepResearch)
        model.client = SimpleNamespace(research=Mock())
        with self.assertRaises(ValueError):
            model.run_research("system", "task", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}])
        model.client.research.assert_not_called()

    def qwen_chunk(self, phase, content, status="typing", status_code=200):
        from dashscope.api_entities.dashscope_response import GenerationResponse
        return GenerationResponse(status_code=status_code, code="error", message="failed",
                                  output={"message": {"phase": phase, "content": content, "status": status}}, usage=None)

    def test_qwen_only_saves_finished_answer_not_planning(self):
        model = QwenDeepResearch.__new__(QwenDeepResearch)
        model.api_key, model.model_name = "placeholder", "qwen-deep-research"
        first = [self.qwen_chunk("answer", "Clarify?", "finished")]
        second = [self.qwen_chunk("ResearchPlanning", "plan", "finished"),
                  self.qwen_chunk("WebResearch", "internal search", "finished"),
                  self.qwen_chunk("answer", "final "), self.qwen_chunk("answer", "report", "finished"),
                  self.qwen_chunk("KeepAlive", "")]
        with patch("evaluation_toolkit.DR_clients.QwenDR.dashscope.Generation.call", side_effect=[iter(first), iter(second)]) as request:
            self.assertEqual(model.run_research("system", "task")["content"], "final report")
            self.assertTrue(all(call.kwargs["incremental_output"] for call in request.call_args_list))
            for call in request.call_args_list:
                for message in call.kwargs['messages']:
                    self.assertIn(message['role'], ['user', 'assistant'])
                    self.assertIsInstance(message['content'], str)
            self.assertEqual(request.call_args_list[0].kwargs['messages'][0]['content'], 'system\ntask')

    def test_qwen_stream_error_or_truncation_cannot_be_a_report(self):
        model = QwenDeepResearch.__new__(QwenDeepResearch)
        for chunks in ([self.qwen_chunk("answer", "partial")],
                       [self.qwen_chunk("answer", "partial"), self.qwen_chunk("answer", "", status_code=500)]):
            with self.subTest(chunks=chunks), self.assertRaises(RuntimeError):
                model._collect_answer(iter(chunks), model.deadline())

    def test_qwen_text_blocks_are_flattened_and_images_explicitly_rejected(self):
        self.assertEqual(_convert_to_qwen_multimodal([
            {"type": "text", "text": "first"}, {"type": "text", "text": "second"}]), 'first\nsecond')
        with self.assertRaises(ValueError):
            _convert_to_qwen_multimodal([{"type": "image_url", "image_url": {"url": "https://image"}}])
        model = QwenDeepResearch.__new__(QwenDeepResearch)
        with patch('evaluation_toolkit.DR_clients.QwenDR.dashscope.Generation.call') as request:
            with self.assertRaises(ValueError):
                model.run_research('system', 'task', [{"type": "image_url", "image_url": {"url": "https://image"}}])
            request.assert_not_called()

    def test_openai_incomplete_result_cannot_be_a_report(self):
        model = OpenaiDeepResearch.__new__(OpenaiDeepResearch)
        model.model_name = "o3-deep-research"
        model.client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=SimpleNamespace(status="incomplete", output_text="partial"))))
        with self.assertRaises(RuntimeError):
            model.run_research("system", "task")

    def test_openai_background_result_is_polled(self):
        model = OpenaiDeepResearch.__new__(OpenaiDeepResearch)
        model.model_name, model.poll_interval = "o3-deep-research", 0
        create = Mock(return_value=SimpleNamespace(id="1", status="queued"))
        retrieve = Mock(return_value=SimpleNamespace(status="completed", output_text="final", usage=None))
        model.client = SimpleNamespace(responses=SimpleNamespace(create=create, retrieve=retrieve))
        self.assertEqual(model.run_research("system", "task")["content"], "final")
        self.assertTrue(create.call_args.kwargs["background"])
        retrieve.assert_called_once()

    def score_csvs(self):
        expert_csv = self.base / "leaderboard.csv"
        fact_csv = self.base / "fact.csv"
        question = {"id": "1", "language": "en", "question": "topic", "expert_evaluation_criteria": RUBRIC}
        report = {"model": "gpt-4o", "response": "report"}
        metadata = {"Total Tasks": [1], "Evaluated Tasks": [1], "Coverage (%)": [100],
                    "Task Set Fingerprint": [task_set_fingerprint([question])],
                    "Report Set Fingerprint": [report_set_fingerprint({"1": report_fingerprint("1", report)})],
                    "Provenance Verified": [True]}
        pd.DataFrame({"Model": ["gpt-4o"], "Avg Score": [80], **metadata}).to_csv(expert_csv, index=False)
        pd.DataFrame({"Model Name": ["gpt-4o"], "Avg Citation Score (Macro)": [60], "Avg Source Quality Score (S_source)": [40],
                      "Evaluation As Of (UTC)": ["2026-10-16T00:00:00+00:00"], **metadata}).to_csv(fact_csv, index=False)
        return expert_csv, fact_csv

    def test_final_score_multiround_and_scale(self):
        expert_csv, fact_csv = self.score_csvs()
        second = self.base / "second.csv"
        frame = pd.read_csv(expert_csv)
        frame['Model'], frame['Avg Score'] = ' GPT-4O ', 60
        frame.to_csv(second, index=False)
        result = final.compute_final_score([str(expert_csv), str(second)], str(fact_csv), str(self.base / "out.csv"), .8, .1, .1)
        self.assertEqual(result.iloc[0]["S_final"], 66)

    def test_final_score_duplicate_fact_models_fail(self):
        expert_csv, fact_csv = self.score_csvs()
        frame = pd.read_csv(fact_csv)
        pd.concat([frame, frame]).to_csv(fact_csv, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            final.compute_final_score([str(expert_csv)], str(fact_csv), str(self.base / "out.csv"), .8, .1, .1)

    def test_final_score_missing_model_or_nan_fail(self):
        expert_csv, fact_csv = self.score_csvs()
        for value in ("different-model", "gpt-4o"):
            frame = pd.read_csv(fact_csv)
            frame["Model Name"] = value
            if value == "gpt-4o":
                frame["Avg Citation Score (Macro)"] = math.nan
            frame.to_csv(fact_csv, index=False)
            with self.assertRaises(ValueError):
                final.compute_final_score([str(expert_csv)], str(fact_csv), str(self.base / "out.csv"), .8, .1, .1)

    def test_final_language_does_not_fall_back_to_all_languages(self):
        expert_csv, fact_csv = self.score_csvs()
        (self.base / "leaderboard_en.csv").write_text(expert_csv.read_text())
        with self.assertRaises(FileNotFoundError):
            final.run_all_versions([str(self.base)], str(fact_csv), str(self.base / "out"), .8, .1, .1)
        self.assertTrue((self.base / "out/final_leaderboard.csv").is_file())
        self.assertFalse((self.base / "out/final_leaderboard_en.csv").exists())

    def test_final_missing_input_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            final.compute_final_score([str(self.base / "missing.csv")], str(self.base / "fact.csv"), str(self.base / "out.csv"), .8, .1, .1)

    def test_changed_question_invalidates_report_cache(self):
        args = self.write_args()
        with patch.object(write, "attempt_query", return_value=("1", "report", None)) as request:
            self.assertEqual(write.main(args), 0)
            self.assertEqual(write.main(args), 0)
            self.assertEqual(request.call_count, 1)
            dataset = Path(args.local_dataset)
            data = json.loads(dataset.read_text())
            data[0]["question"] = "changed topic"
            dataset.write_text(json.dumps(data))
            self.assertEqual(write.main(args), 0)
            self.assertEqual(request.call_count, 2)

    def test_changed_report_invalidates_expert_cache(self):
        args = self.expert_args()
        with patch.object(expert, "call_llm", return_value=json.dumps(dimensions())) as request:
            self.assertEqual(expert.main(args), 0)
            self.assertEqual(expert.main(args), 0)
            self.assertEqual(request.call_count, 1)
            path = Path(args.report_dir) / "reports_gpt-4o.json"
            data = json.loads(path.read_text())
            data["1"]["response"] = "changed report"
            path.write_text(json.dumps(data))
            self.assertEqual(expert.main(args), 0)
            self.assertEqual(request.call_count, 2)

    def test_expert_missing_reports_fail_and_sample_limit_is_stable(self):
        args = self.expert_args()
        questions = Path(args.question_path)
        data = json.loads(questions.read_text())
        data.append(dict(data[0], id="2"))
        questions.write_text(json.dumps(data))
        with patch.object(expert, "call_llm", return_value=json.dumps(dimensions())) as request:
            self.assertEqual(expert.main(args), 1)
            args.max_samples = 1
            self.assertEqual(expert.main(args), 0)
            self.assertEqual(expert.main(args), 0)
            request.assert_called_once()

    def test_expert_empty_reports_and_invalid_sample_limit_fail(self):
        args = self.expert_args()
        (Path(args.report_dir) / "reports_gpt-4o.json").write_text("{}")
        self.assertEqual(expert.main(args), 1)
        for limit in (0, -1):
            args.max_samples = limit
            with self.assertRaises(ValueError):
                expert.main(args)

    def test_expert_statistics_report_partial_coverage(self):
        args = self.expert_args()
        query = Path(args.question_path)
        data = json.loads(query.read_text())
        data.append(dict(data[0], id="2", language="zh"))
        query.write_text(json.dumps(data))
        scores = self.base / "score-input"
        scores.mkdir()
        report = {"model": "gpt-4o", "response": "report"}
        (scores / "scores.jsonl").write_text(json.dumps({"task_id": "1", "model": "gpt-4o", "total_score": 80,
            "dimensions_breakdown": dimensions(), "question_fingerprint": question_fingerprint(data[0]),
            "report_fingerprint": report_fingerprint("1", report)}) + "\n")
        output = self.base / "partial.csv"
        self.assertEqual(expert_stat.stat(str(scores), str(query), str(output)), 1)
        row = pd.read_csv(output).iloc[0]
        self.assertEqual(row['Coverage (%)'], 50)
        self.assertEqual(row['Total Tasks'], 2)
        self.assertEqual(row['Evaluated Tasks'], 1)
        self.assertEqual(expert_stat.stat(str(scores), str(query), str(output), 'en'), 0)

    def test_final_score_rejects_partial_expert_leaderboard(self):
        expert_csv, fact_csv = self.score_csvs()
        frame = pd.read_csv(expert_csv)
        frame['Coverage (%)'] = 97.5
        frame.to_csv(expert_csv, index=False)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            final.compute_final_score([str(expert_csv)], str(fact_csv), str(self.base / "out.csv"), .8, .1, .1)

    def test_question_metadata_requires_nonempty_unique_ids_and_valid_languages(self):
        for records in ([], {}, [{"id": "", "language": "en", "question": "topic"}],
                        [{"id": None, "language": "en", "question": "topic"}],
                        [{"id": "1", "language": "fr", "question": "topic"}],
                        [{"id": "1", "question": "topic"}],
                        [{"id": "1", "language": "en", "question": ""}],
                        [{"id": "1", "language": "en", "question": "topic"},
                         {"id": 1, "language": "zh", "question": "topic"}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                validate_questions(records)

    def test_empty_questions_fail_in_all_subjective_entry_points(self):
        write_args = self.write_args()
        Path(write_args.local_dataset).write_text('[]')
        with self.assertRaises(ValueError):
            write.main(write_args)
        expert_args = self.expert_args()
        Path(expert_args.question_path).write_text('[]')
        with self.assertRaises(ValueError):
            expert.main(expert_args)
        with self.assertRaises(ValueError):
            expert_stat.stat(str(self.base), expert_args.question_path, str(self.base / 'out.csv'))


if __name__ == "__main__":
    unittest.main()
