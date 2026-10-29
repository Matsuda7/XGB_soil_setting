"""CLI-only checks: analysis functions are mocked and never executed."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from module import full_distribution as full


class DistributionCLITests(unittest.TestCase):
    def test_plan_never_initializes_analysis(self):
        with patch.object(full,'initialize',side_effect=AssertionError('analysis started')) as initialize, contextlib.redirect_stdout(io.StringIO()) as output:
            result=full.main(['--comparison-run','unused','--output','unused','--stage','tabpfn','--plan'])
        self.assertEqual(result,0)
        initialize.assert_not_called()
        self.assertIn('"analysis_executed": false',output.getvalue())

    def test_stopped_run_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'STOP_TABPFN').touch()
            with patch.object(full,'initialize') as initialize, contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                full.main(['--comparison-run','unused','--output',tmp,'--stage','tabpfn'])
            initialize.assert_not_called()
            self.assertTrue((Path(tmp)/'STOP_TABPFN').exists())

    def test_explicit_resume_dispatches_only_requested_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            flag=Path(tmp)/'STOP_TABPFN';flag.touch()
            with patch.object(full,'initialize',return_value=({'tabpfn':{}},{})),patch.object(full,'tabpfn') as predict,patch.object(full,'prepare') as prepare:
                result=full.main(['--comparison-run','unused','--output',tmp,'--stage','tabpfn','--resume'])
            self.assertEqual(result,0)
            self.assertFalse(flag.exists())
            predict.assert_called_once()
            prepare.assert_not_called()


if __name__=='__main__':unittest.main()
