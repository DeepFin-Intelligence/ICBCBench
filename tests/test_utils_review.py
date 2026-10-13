"""Container parsing and local data validation regressions."""
import json
from pathlib import Path
import tempfile
import unittest

from evaluation_toolkit.utils import safe_json_loads, load_local_dataset


class UtilityReviewTests(unittest.TestCase):
    def test_prose_wrapped_single_element_array_preserves_container(self):
        self.assertEqual(safe_json_loads('Scores follow: [{"dimension_id":"1.1","score":10}] Done.'),
                         [{'dimension_id': '1.1', 'score': 10}])

    def test_nested_json_brackets_in_strings_and_trailing_text(self):
        expected = [{'fact': 'value contains ] and }', 'values': [1, 2]}]
        self.assertEqual(safe_json_loads('Result: ' + json.dumps(expected) + ' Additional [note].'), expected)

    def test_code_fences_and_objects(self):
        self.assertEqual(safe_json_loads('```json\n[{"idx": 1}]\n```'), [{'idx': 1}])
        self.assertEqual(safe_json_loads('Answer: {"correct": "yes"} done.'), {'correct': 'yes'})

    def test_truncated_array_does_not_return_a_complete_inner_object(self):
        for text in ['[{"score":80},', '[{"score":80},{"score":',
                     'Result: [{"idx":1,"result":"supported"},']:
            with self.subTest(text=text):
                self.assertIsNone(safe_json_loads(text))

    def test_non_json_bracketed_prose_does_not_hide_the_array(self):
        self.assertEqual(safe_json_loads('[Note] Result: [{"idx":1}] done.'), [{'idx': 1}])

    def load(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.json'
            path.write_text(json.dumps(value))
            return load_local_dataset(path)

    def test_empty_dataset_and_optional_columns(self):
        self.assertEqual(self.load([]), {})
        self.assertEqual(self.load([{'id': '1'}, {'id': '2', 'attachment': []}]),
                         {'id': ['1', '2'], 'attachment': [None, []]})

    def test_report_mapping_is_preserved(self):
        records = {'1': {'model': 'gpt-4o', 'response': 'report'}}
        self.assertEqual(self.load(records), records)

    def test_malformed_local_data_reports_a_useful_error(self):
        for value, message in [([{'id': '1'}, 2], 'rows'),
                               ({'id': ['1'], 'question': []}, 'equal lengths'),
                               ({'id': ['1'], 'question': 'text'}, 'all be lists'),
                               ('invalid', 'array or object')]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                self.load(value)


if __name__ == '__main__':
    unittest.main()
