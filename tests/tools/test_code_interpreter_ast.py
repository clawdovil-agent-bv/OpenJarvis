"""code_interpreter: AST validation blocks the old substring bypasses."""

from __future__ import annotations

import pytest

from openjarvis.tools.code_interpreter import UnsafeCodeError, _validate_ast


class TestCodeInterpreterValidation:
    @pytest.mark.parametrize(
        "code",
        [
            "import os",
            "import subprocess as s",
            "from os import system",
            "getattr(__builtins__, 'system')",
            "eval ('1+1')",  # a space defeated the old substring check
            "().__class__.__base__.__subclasses__()",
            "open('/etc/passwd')",
            "__import__('os')",
        ],
    )
    def test_dangerous_code_blocked(self, code):
        with pytest.raises((UnsafeCodeError, SyntaxError)):
            _validate_ast(code)

    @pytest.mark.parametrize(
        "code",
        [
            "print(sum(range(10)))",
            "import math\nprint(math.sqrt(2))",
            "import json\nprint(json.dumps({'a': 1}))",
            "xs = [i * 2 for i in range(5)]\nprint(xs)",
        ],
    )
    def test_safe_code_allowed(self, code):
        _validate_ast(code)  # must not raise
