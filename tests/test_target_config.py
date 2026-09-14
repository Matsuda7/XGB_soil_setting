import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from module import pipeline


class TargetConfigTests(unittest.TestCase):
    def test_selection_and_override(self):
        self.assertEqual(pipeline.resolve_targets({}), ('c', 'phi', 'gamma'))
        self.assertEqual(pipeline.resolve_targets({'targets': ['gamma', 'phi', 'gamma']}), ('gamma', 'phi'))
        self.assertEqual(pipeline.resolve_targets({'targets': ['gamma']}, ['phi']), ('phi',))

    def test_invalid_targets(self):
        for value in ([], 'phi', None, ['unknown'], [1], [['phi']]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pipeline.resolve_targets({'targets': value})

    def test_plan_and_cli_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'pipeline.json'
            config.write_text('{"targets": ["gamma"]}')
            for extra, expected in (([], ['gamma']), (['--targets', 'phi'], ['phi'])):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), patch.object(pipeline, 'missing_inputs', return_value=[]), patch.object(pipeline, 'run') as run:
                    self.assertEqual(pipeline.main(['--config', str(config), '--plan', *extra]), 0)
                    run.assert_not_called()
                self.assertEqual(list(json.loads(output.getvalue())['targets']), expected)

    def test_cli_uses_configured_target_for_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'pipeline.json'
            config.write_text('{"targets": ["phi"]}')
            with patch.object(pipeline, 'run', return_value=0) as run:
                self.assertEqual(pipeline.main(['--config', str(config)]), 0)
                run.assert_called_once_with('all', ('phi',), str(config), False)

    def test_invalid_configuration_stops_before_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'pipeline.json'
            for contents in ('{"targets": []}', '{"targets": ["phii"]}', '[]', '{'):
                config.write_text(contents)
                with patch.object(pipeline, 'run') as run, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        pipeline.main(['--config', str(config)])
                    self.assertEqual(error.exception.code, 2)
                    run.assert_not_called()

    def test_run_dispatches_only_configured_target_and_reports_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'pipeline.json'
            config.write_text('{"targets": ["gamma"]}')
            with patch.object(pipeline, 'RESULTS', root), patch.object(pipeline, 'missing_inputs', return_value=[]), patch('module.distributions.distribute', return_value={'status': 'not_implemented'}) as dispatch:
                self.assertEqual(pipeline.run(stage='distribute', config_path=config), 2)
                dispatch.assert_called_once_with('gamma')
            result = json.loads((root / 'run_summary.json').read_text())
            self.assertEqual(result['targets'], ['gamma'])
            self.assertEqual(list(result['stages']['distribute']), ['gamma'])


if __name__ == '__main__':
    unittest.main()
