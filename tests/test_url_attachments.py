"""URL PDF-page attachments, tested with generated PDFs and mocked HTTP only."""
import base64
import copy
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pymupdf
import requests

from evaluation_toolkit.objective_eval import predict


SOURCE_URL = "https://example.invalid/official-report"


def generated_pdf():
    """Each page has a distinct size and color so wrong-page renders differ."""
    with pymupdf.open() as document:
        for number, color in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 1):
            page = document.new_page(width=100 + number * 20, height=80 + number * 10)
            page.draw_rect(page.rect, color=color, fill=color)
            page.insert_text((10, 30), f"PAGE {number}")
        return document.tobytes()


def source_attachment(**overrides):
    value = {
        "source_url": SOURCE_URL, "mime_type": "application/pdf",
        "pdf_page": 2, "printed_page": 44, "description": "CIB dashboard",
    }
    value.update(overrides)
    return value


class URLAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.pdf = generated_pdf()
        self.response = Mock(content=self.pdf)
        self.http = patch.object(predict.requests, "get", return_value=self.response).start()
        self.addCleanup(patch.stopall)
        patch.object(socket.socket, "connect", side_effect=AssertionError("Live network forbidden")).start()
        self.options = SimpleNamespace(
            local_dataset=None, model="gpt-4o", temperature=0,
            max_completion_tokens=128,
        )
        predict.args = self.options
        self.question = {
            "id": "url-test", "language": "en", "question": "Read the dashboard.",
            "attachment": [source_attachment()],
        }

    def expected_png(self, page):
        with pymupdf.open(stream=self.pdf, filetype="pdf") as document:
            return document.load_page(page - 1).get_pixmap(
                matrix=pymupdf.Matrix(predict.PDF_PAGE_RENDER_SCALE, predict.PDF_PAGE_RENDER_SCALE),
                alpha=False,
            ).tobytes("png")

    def test_default_materializes_only_selected_page_as_png(self):
        real_document = pymupdf.open(stream=self.pdf, filetype="pdf")
        selected_pages = []

        class TrackedDocument:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                real_document.close()

            def __getattr__(self, name):
                return getattr(real_document, name)

            def load_page(self, index):
                selected_pages.append(index)
                return real_document.load_page(index)

        expected = self.expected_png(2)
        with patch.object(predict.pymupdf, "open", return_value=TrackedDocument()):
            blocks, annotation = predict.build_attachment_content(self.question["attachment"])
        self.assertEqual(selected_pages, [1])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "image_url")
        image_url = blocks[0]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(image_url.split(",", 1)[1]), expected)
        self.assertNotEqual(expected, self.expected_png(1))
        self.assertNotEqual(expected, self.expected_png(3))
        self.http.assert_called_once_with(
            SOURCE_URL, headers={"User-Agent": "ICBCBench/1.0"}, timeout=30,
        )
        self.response.raise_for_status.assert_called_once_with()
        self.assertIn(SOURCE_URL, annotation)
        self.assertIn("PDF page 2 (1-based)", annotation)
        self.assertIn("printed page 44", annotation)
        self.assertIn("CIB dashboard", annotation)

    def test_source_url_takes_precedence_over_nonexistent_local_file(self):
        attachment = source_attachment(file_path="attachments/never-bundled.png")
        blocks, _ = predict.build_attachment_content([attachment], "missing/questions.json")
        self.assertEqual(len(blocks), 1)
        self.assertTrue(predict.request_fingerprint(
            dict(self.question, attachment=[attachment]), self.options,
        ))

    def test_text_only_reference_does_not_download_or_render(self):
        with patch.object(predict.pymupdf, "open", side_effect=AssertionError("Unexpected PDF render")):
            blocks, annotation = predict.build_attachment_content(
                self.question["attachment"], materialize_images=False,
            )
            messages = predict.format_message(self.question, for_dr_model=True)
        self.assertEqual(blocks, [])
        self.assertIn(SOURCE_URL, annotation)
        self.assertIsInstance(messages[1]["content"], str)
        self.assertIn(SOURCE_URL, messages[1]["content"])
        self.assertIn("PDF page 2 (1-based)", messages[1]["content"])
        self.http.assert_not_called()

    def test_multimodal_message_keeps_source_annotation_and_image(self):
        content = predict.format_message(self.question)[1]["content"]
        self.assertEqual([block["type"] for block in content], ["text", "image_url"])
        self.assertIn(SOURCE_URL, content[0]["text"])
        self.assertIn("CIB dashboard", content[0]["text"])
        self.assertEqual(base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]),
                         self.expected_png(2))

    def test_text_only_research_routes_source_references_as_text(self):
        for model in ("tavily-research", "qwen-deep-research", "Qwen-deep-research"):
            with self.subTest(model=model):
                self.options.model = model
                dr = Mock()
                dr.run_research.return_value = {"content": "answer", "usage": None}
                with patch.object(predict, "create_dr_client", return_value=dr):
                    result = predict.attempt_question(self.question)
                self.assertEqual(result[1], "answer")
                content = dr.run_research.call_args.kwargs["multimodal_content"]
                self.assertIsInstance(content, str)
                self.assertIn(SOURCE_URL, content)
        self.http.assert_not_called()

    def test_url_and_processing_metadata_invalidate_fingerprints_without_network(self):
        original = predict.request_fingerprint(self.question, self.options)
        for field, value in (
            ("source_url", SOURCE_URL + "-revised"), ("pdf_page", 3),
            ("printed_page", 45), ("description", "Revised dashboard"),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.question)
                changed["attachment"][0][field] = value
                self.assertNotEqual(predict.request_fingerprint(changed, self.options), original)
        with patch.object(predict, "URL_ATTACHMENT_PROCESSING", "future-version"):
            self.assertNotEqual(predict.request_fingerprint(self.question, self.options), original)
        with patch.object(predict, "PDF_PAGE_RENDER_SCALE", 3):
            self.assertNotEqual(predict.request_fingerprint(self.question, self.options), original)
        changed_question = dict(self.question, question="Read a different dashboard.")
        self.assertNotEqual(predict.request_fingerprint(changed_question, self.options), original)
        self.http.assert_not_called()

    def test_download_timeout_and_http_errors_are_not_silently_omitted(self):
        self.http.side_effect = requests.Timeout("download timeout")
        with self.assertRaisesRegex(RuntimeError, "required PDF attachment.*download timeout"):
            predict.build_attachment_content(self.question["attachment"])
        self.http.side_effect = None
        self.response.raise_for_status.side_effect = requests.HTTPError("404")
        with self.assertRaisesRegex(RuntimeError, "required PDF attachment.*404"):
            predict.build_attachment_content(self.question["attachment"])

    def test_non_pdf_or_corrupt_sources_fail(self):
        for data in (b"<html>Login required</html>", b"%PDF-1.7\ncorrupt document"):
            with self.subTest(data=data):
                self.response.content = data
                with self.assertRaisesRegex(ValueError, "PDF"):
                    predict.build_attachment_content(self.question["attachment"])

    def test_page_validation_and_out_of_range_fail_clearly(self):
        for page in (None, 0, -1, True, 1.5, "2"):
            with self.subTest(page=page), self.assertRaisesRegex(ValueError, "positive 1-based integer"):
                predict.build_attachment_content([source_attachment(pdf_page=page)])
        self.http.assert_not_called()
        with self.assertRaisesRegex(ValueError, "page 4 is out of range.*3 pages"):
            predict.build_attachment_content([source_attachment(pdf_page=4)])

    def test_reference_schema_errors_fail_before_download(self):
        for overrides in ({"source_url": ""}, {"source_url": "file:///tmp/report.pdf"},
                          {"mime_type": "image/png"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                predict.build_attachment_content([source_attachment(**overrides)])
        self.http.assert_not_called()

    def test_legacy_local_image_loading_and_content_hashing_remain_available(self):
        with tempfile.TemporaryDirectory(prefix="url-attachment-local-") as directory:
            image = Path(directory) / "local.png"
            image.write_bytes(self.expected_png(1))
            attachment = {"file_path": "local.png", "mime_type": "image/png",
                          "original_filename": "local.png", "source_url": None}
            dataset_path = str(Path(directory) / "questions.json")
            blocks, _ = predict.build_attachment_content([attachment], dataset_path)
            self.assertEqual(base64.b64decode(blocks[0]["image_url"]["url"].split(",", 1)[1]), image.read_bytes())
            options = SimpleNamespace(**{**vars(self.options), "local_dataset": dataset_path})
            question = dict(self.question, attachment=[attachment])
            original = predict.request_fingerprint(question, options)
            image.write_bytes(self.expected_png(3))
            self.assertNotEqual(predict.request_fingerprint(question, options), original)
        self.http.assert_not_called()

    def test_text_only_mode_rejects_required_local_image_without_source(self):
        with tempfile.TemporaryDirectory(prefix="url-attachment-local-") as directory:
            image = Path(directory) / "local.png"
            image.write_bytes(self.expected_png(1))
            with self.assertRaisesRegex(ValueError, "requires a source URL"):
                predict.build_attachment_content(
                    [{"file_path": str(image), "mime_type": "image/png"}], materialize_images=False,
                )


if __name__ == "__main__":
    unittest.main()
