import tempfile
import unittest
from pathlib import Path
from module.prediction_output import prepare_chunk_output

class PredictionOutputTests(unittest.TestCase):
    def test_archive_preserves_files_and_new_output_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "chunks"
            out.mkdir()
            name = "N_value_10m_000001.csv.gz"
            (out / name).write_bytes(b"old predictions")
            (out / "partial.tmp").write_bytes(b"partial")
            archive = prepare_chunk_output(out, False, "archive")
            self.assertEqual((archive / name).read_bytes(), b"old predictions")
            self.assertTrue((archive / "partial.tmp").exists())
            self.assertEqual(list(out.iterdir()), [])
            self.assertIsNone(prepare_chunk_output(out, False, "archive"))

    def test_error_and_resume_do_not_move_existing_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "chunks"
            out.mkdir()
            chunk = out / "N_value_10m_000001.csv.gz"
            chunk.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                prepare_chunk_output(out, False, "error")
            self.assertIsNone(prepare_chunk_output(out, True, "archive"))
            self.assertEqual(chunk.read_bytes(), b"original")

    def test_bad_policy_fails_without_creating_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "chunks"
            with self.assertRaises(ValueError):
                prepare_chunk_output(out, False, "typo")
            self.assertFalse(out.exists())
