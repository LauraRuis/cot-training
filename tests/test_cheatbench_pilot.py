"""Local contract checks; no GPU, model, network or container execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from contract.cheatbench.pilot import check_runtime, container_command, docker_recipe, grade, parse_reply


class PilotContracts(unittest.TestCase):
    def test_action_only_after_reasoning_and_exact_schema(self):
        chain, action = parse_reply('Inspect code.\n</think>\n{"command": "ls"}')
        self.assertEqual(action, {"command": "ls"})
        for raw in ('{"command": "ls"}', 'x</think>{"command":"ls","done":"ok"}',
                    'x</think>{"command":42}', 'x</think>["ls"]'):
            with self.assertRaises((ValueError, TypeError)):
                parse_reply(raw)

    def test_command_payload_is_only_a_container_argument(self):
        payload = 'echo "$HOME"; touch /tmp/should-not-run'
        argv = container_command(Path('/task.sif'), None, payload)
        self.assertEqual(argv[-4:], ['/task.sif', '/bin/bash', '-c', payload])
        self.assertIn('--pid', argv)
        self.assertIn('home,cwd,hostfs,bind-paths', argv)
        self.assertNotIn('--bind', argv)

    def test_docker_recipe_rejects_untranslated_instructions(self):
        image, setup = docker_recipe('FROM example/image:tag\nWORKDIR /app\nRUN git status\n')
        self.assertEqual(image, 'example/image:tag')
        self.assertIn('git status', setup)
        with self.assertRaises(ValueError):
            docker_recipe('FROM example/image\nCOPY secret /app\nRUN true')

    def test_site_ignoring_pid_flag_stops_runner(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def fake_checked(argv, log, timeout):
                log.write_text('WARNING: Not virtualizing pid namespace\nPID_NAMESPACE=pid:[123]\n')
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('contract.cheatbench.pilot.checked', side_effect=fake_checked), \
                 patch('contract.cheatbench.pilot.os.readlink', return_value='pid:[123]'):
                with self.assertRaisesRegex(RuntimeError, 'PID isolation is unavailable'):
                    check_runtime(Path('/task.sif'), root)

    def test_reward_file_without_test_results_is_not_a_failed_solution(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / 'verifier'
            def fake_execute(argv, log, timeout):
                (out / 'reward.txt').write_text('0')
                return {'returncode': 1, 'timed_out': False}
            with patch('contract.cheatbench.pilot.execute', side_effect=fake_execute):
                result = grade(root / 'task.sif', None, root, out)
            self.assertIsNone(result['success'])
            self.assertEqual(result['status'], 'verifier_error')

    def test_valid_native_failure_is_counted_as_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / 'verifier'
            def fake_execute(argv, log, timeout):
                (out / 'reward.txt').write_text('0')
                (out / 'output.json').write_text(json.dumps({'tests': [{'name': 't', 'status': 'FAILED'}]}))
                return {'returncode': 1, 'timed_out': False}
            with patch('contract.cheatbench.pilot.execute', side_effect=fake_execute):
                result = grade(root / 'task.sif', None, root, out)
            self.assertEqual(result['success'], 0)

    def test_native_parser_with_no_passes_keeps_zero_but_requires_review(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / 'verifier'
            def fake_execute(argv, log, timeout):
                (out / 'reward.txt').write_text('0')
                (out / 'output.json').write_text('{"tests": []}')
                (out / 'run-script-stdout.txt').write_text('FAIL package [build failed]')
                (out / 'run-script-stderr.txt').write_text('compiler diagnostic')
                log.write_text('RESULT: FAILED')
                return {'returncode': 1, 'timed_out': False}
            with patch('contract.cheatbench.pilot.execute', side_effect=fake_execute):
                result = grade(root / 'task.sif', None, root, out)
            self.assertEqual(result['success'], 0)
            self.assertEqual(result['status'], 'empty_test_results_review_required')


if __name__ == '__main__':
    unittest.main()
