import os
import tempfile
import unittest

class BaseEvaluatorTest(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory and Evaluator instance."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        """Clean up the temporary directory after tests."""
        self.tmp_dir.cleanup()

    def create_file(self, filename, contents):
        """Helper to write JSON or text files in the temporary directory."""
        path = os.path.join(self.tmp_path, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(contents)
        return path
    