"""Exercise public bilingual workflows with a non-UTF-8 default text encoding."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(os.name == 'nt', 'This regression uses the POSIX C locale')
class PortableIOTests(unittest.TestCase):
    def run_without_utf8_default(self, script, *arguments):
        env = {key: value for key, value in os.environ.items() if not key.endswith('API_KEY')}
        env.update(LC_ALL='C', PYTHONUTF8='0', PYTHONCOERCECLOCALE='0',
                   PYTHON_DOTENV_DISABLED='1')
        script = "import locale\nassert locale.getpreferredencoding(False).lower() != 'utf-8'\n" + textwrap.dedent(script)
        result = subprocess.run([sys.executable, '-c', script, *map(str, arguments)], cwd=ROOT,
                                env=env, capture_output=True, encoding='utf-8', timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_release_check_dry_run_and_demo_read_bilingual_data(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.run_without_utf8_default('''
                from pathlib import Path
                import socket
                import sys
                from unittest.mock import patch
                from evaluation_toolkit import cli

                base = Path(sys.argv[1])
                with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')):
                    assert cli.main(['check']) == 0
                    assert cli.main(['run', '--model', 'gpt-4o', '--track', 'both', '--limit', '2',
                                     '--as_of', '2026-10-16', '--run_dir', str(base / 'plan'), '--dry_run']) == 0
                    assert not (base / 'plan').exists()
                    assert cli.main(['demo', '--output_dir', str(base / 'demo')]) == 0
            ''', base)
            with (base / 'demo/subjective/final_leaderboard.csv').open(encoding='utf-8-sig') as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row['S_final']), 74)
            self.assertEqual(int(row['Total Tasks']), 2)

    def test_child_stage_writes_utf8_log_under_non_utf8_parent_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_without_utf8_default('''
                from pathlib import Path
                import sys
                from unittest.mock import patch
                from evaluation_toolkit import cli

                base = Path(sys.argv[1]) / 'run'
                options = cli.parser().parse_args(['run', '--model', 'gpt-4o', '--track', 'objective',
                                                   '--run_dir', str(base)])
                steps = [{'name': 'unicode-log', 'command': [sys.executable, '-c',
                                                            'print(chr(0x4e2d) + chr(0x6587))']}]
                with patch.object(cli, 'check_keys'), patch.object(cli, 'build_plan', return_value=({}, {}, steps)):
                    assert cli.run_pipeline(options) == 0
                assert (base / 'logs/unicode-log.log').read_text(encoding='utf-8').strip() == chr(0x4e2d) + chr(0x6587)
            ''', directory)

    def test_resume_reads_utf8_manifest_and_frozen_chinese_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_without_utf8_default('''
                import json
                from pathlib import Path
                import socket
                import sys
                from types import SimpleNamespace
                from unittest.mock import patch
                from evaluation_toolkit import cli

                base = Path(sys.argv[1]) / 'run'
                arguments = ['run', '--model', 'gpt-4o', '--track', 'objective', '--limit', '2',
                             '--run_dir', str(base)]
                with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')), \\
                     patch.object(cli, 'check_keys'), \\
                     patch.object(cli.platform, 'platform', return_value='test-platform'), \\
                     patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as stages:
                    assert cli.main(arguments) == 0
                    assert cli.main(arguments) == 0
                    assert stages.call_count == 6
                subset = json.loads((base / 'inputs/objective.json').read_text(encoding='utf-8'))
                assert [question['language'] for question in subset] == ['en', 'zh']
                manifest = json.loads((base / 'run_manifest.json').read_text(encoding='utf-8'))
                assert manifest['status'] == 'complete'
            ''', directory)


if __name__ == '__main__':
    unittest.main()
