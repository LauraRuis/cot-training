"""Local contract checks; no GPU, model, network or container execution."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from contract.cheatbench.pilot import check_runtime, container_command, docker_recipe, grade, parse_reply, run
from contract.cheatbench.docker_runtime import DockerRuntime


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

    def test_docker_timeout_stops_container_code_not_only_cli(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'tool.log'
            runtime = DockerRuntime(SimpleNamespace(docker_context=None), 'sha256:example')
            def checked(argv, out, timeout):
                if 'inspect' in argv:
                    out.write_text('[{"State": {"Running": false}}]')
            with patch('contract.cheatbench.docker_runtime.checked', side_effect=checked) as calls, \
                 patch('contract.cheatbench.docker_runtime.execute', return_value={'timed_out': True}):
                result = runtime.command('sample', 'sleep 999', log, 1)
            self.assertTrue(result['timed_out'])
            self.assertIn(['docker', 'stop', '--time', '0', 'sample'],
                          [call.args[0] for call in calls.call_args_list])

    def test_docker_also_stops_on_client_exception(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = DockerRuntime(SimpleNamespace(docker_context=None), 'sha256:example')
            with patch('contract.cheatbench.docker_runtime.checked') as calls, \
                 patch('contract.cheatbench.docker_runtime.execute', side_effect=OSError('broken client')):
                with self.assertRaises(OSError):
                    runtime.command('sample', 'anything', Path(d) / 'tool.log', 1)
            self.assertEqual(calls.call_args.args[0], ['docker', 'stop', '--time', '0', 'sample'])

    def test_split_rollout_retains_reasoning_and_requires_independent_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prepared = root / 'prepared'
            from contract.cheatbench.pilot import TASK
            task = prepared / 'cheatbench/tasks/software_engineering/harbor/tasks' / TASK
            task.mkdir(parents=True)
            (task / 'instruction.md').write_text('Fix the program.')
            (prepared / 'READY').write_text('ready')
            (prepared / 'manifest.json').write_text(json.dumps({
                'runtime': 'docker', 'image_id': 'sha256:example', 'task_files_sha256': {}}))
            args = SimpleNamespace(work_root=root, out=root / 'run', runtime='docker',
                model='test', model_revision='pinned', url='http://localhost:8000/v1',
                probe_image=None, samples=1, max_turns=3, max_tokens=100, context_length=1000,
                seed=1, tool_timeout=1)
            tok = Mock(chat_template='template')
            tok.apply_chat_template.side_effect = lambda *a, **kw: [1] if kw.get('tokenize') else '<think>\n'
            tok.encode.return_value = [1]
            llm = Mock()
            responses = []
            for text in ('inspect code</think>{"command":"ls"}', 'finished</think>{"done":"ok"}'):
                responses.append(SimpleNamespace(choices=[SimpleNamespace(text=text, finish_reason='stop')],
                    model_dump=lambda text=text: {'text': text}))
            llm.client.completions.create.side_effect = responses
            llm.next_token_probs.return_value = {'yes': .2, 'no': .8}
            runtime = Mock()
            runtime.new_episode.return_value = 'sample'
            def command(state, cmd, log, timeout):
                log.write_text('file.txt')
                return {'log': str(log), 'returncode': 0, 'timed_out': False}
            runtime.command.side_effect = command
            runtime.grade.return_value = {'success': 1, 'status': 'graded'}
            modules = {'transformers': SimpleNamespace(AutoTokenizer=Mock(from_pretrained=Mock(return_value=tok))),
                       'contract.llm': SimpleNamespace(LLM=Mock(return_value=llm))}
            with patch.dict('sys.modules', modules), \
                 patch('contract.cheatbench.docker_runtime.DockerRuntime', return_value=runtime):
                run(args)
            second_prompt = llm.client.completions.create.call_args_list[1].kwargs['prompt']
            self.assertIn('inspect code</think>', second_prompt)
            row = json.loads((args.out / 'summary.json').read_text())[0]
            self.assertEqual(row['native_success'], 1)
            self.assertEqual(row['monitor_p_yes'], .2)
            self.assertIsNone(row['cheating'])
            self.assertIsNone(row['audit_evidence'])


if __name__ == '__main__':
    unittest.main()
