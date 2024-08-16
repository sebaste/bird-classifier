"""Test the err module."""


import contextlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "../classifier/aux")
from classifier.aux.err import *


@contextlib.contextmanager
def noStderr():
    """Context manager to suppress stderr."""

    savedStderr = sys.stderr
    class DevNull(object):
        """Class representing '/dev/null'."""
        def write(self, _):
            """Empty write."""
        def flush(self):
            """Empty flush."""
    sys.stderr = DevNull()
    try:
        yield
    finally:
        sys.stderr = savedStderr

class ErrTestCases(unittest.TestCase):
    """Test suite for the err module."""

    @staticmethod
    @patch("sys.stderr.write")
    def test_uses_stderr_write(mock_write):
        """Test that err() calls sys.stderr.write()."""
        err("test")
        mock_write.assert_called()

    @staticmethod
    @patch("sys.exit")
    @patch("sys.stderr.write")
    def test_uses_stderr_and_exits_with_default_exit_code(mock_write, mock_exit):
        """Test that err_exit() calls sys.stderr.write() and that it returns a default exit code."""
        err_exit("test")
        mock_write.assert_called()
        mock_exit.assert_called_with(1)

    @staticmethod
    @patch("sys.exit")
    def test_exits_with_correct_exit_code(mock_exit):
        """Test that err_exit() returns a given exit code."""
        with noStderr():
            err_exit("test", exit_code=42)
            mock_exit.assert_called_with(42)

if __name__ == "__main__":
    unittest.main()
