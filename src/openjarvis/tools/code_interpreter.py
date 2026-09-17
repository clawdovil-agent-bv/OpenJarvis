"""Code interpreter tool — Python execution with AST validation + hardening.

Security model (defense in depth):

1. **AST allow/deny validation** (this module) rejects code *before* it runs:
   imports of dangerous modules, dunder-attribute walks, and calls to
   ``eval``/``exec``/``compile``/``__import__``/``open``/``getattr`` &c. This
   replaces the old substring blocklist, which was trivially bypassed (e.g.
   ``getattr(__builtins__, 'sys'+'tem')`` or a simple space: ``eval ('...')``).
2. **Isolated interpreter** — the child runs with ``-I -B -S`` (isolated mode,
   no ``.pyc``, no ``site``), a sanitized environment, and POSIX resource
   limits (CPU + address space + no new files) applied in a ``preexec_fn``.
3. **Outer sandbox** — on a server this tool should run inside the Docker
   sandbox (see ``code_interpreter_docker`` / ``deploy/docker/Dockerfile.sandbox``)
   or be disabled entirely. AST validation is a filter, not a jail: the Docker
   boundary is the real containment for untrusted code.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

# Modules whose import is refused outright — they grant filesystem, process,
# network, or interpreter-internal access that defeats the interpreter's intent.
_BLOCKED_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "ctypes",
        "signal",
        "importlib",
        "builtins",
        "pty",
        "fcntl",
        "multiprocessing",
        "threading",
        "asyncio",
        "resource",
        "mmap",
        "gc",
        "inspect",
        "code",
        "codeop",
        "pdb",
        "cProfile",
        "pickle",
        "shelve",
        "marshal",
        "webbrowser",
        "http",
        "urllib",
        "ftplib",
        "telnetlib",
        "smtplib",
        "requests",
        "httpx",
        "pathlib",
        "glob",
        "tempfile",
    }
)

# Names that must never be referenced or called (escape / IO primitives).
_BLOCKED_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
        "help",
    }
)


class UnsafeCodeError(ValueError):
    """Raised when submitted code fails AST validation."""


def _validate_ast(code: str) -> None:
    """Reject code that could escape the interpreter or perform IO.

    Raises :class:`UnsafeCodeError` (or ``SyntaxError``) on anything unsafe.
    Blocking is by *structure*, not by string matching, so obfuscation such as
    ``getattr(x, 'sys'+'tem')`` or spacing tricks cannot slip through.
    """
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_IMPORTS:
                    raise UnsafeCodeError(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _BLOCKED_IMPORTS:
                raise UnsafeCodeError(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Attribute):
            # Dunder attribute access enables __class__ / __subclasses__ /
            # __globals__ escape chains — refuse all of it.
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise UnsafeCodeError(f"dunder attribute access '{node.attr}' blocked")
        elif isinstance(node, ast.Name):
            if node.id in _BLOCKED_NAMES:
                raise UnsafeCodeError(f"use of '{node.id}' is not allowed")
            if node.id.startswith("__") and node.id.endswith("__"):
                raise UnsafeCodeError(f"dunder name '{node.id}' is not allowed")


def _child_limits() -> None:  # pragma: no cover - POSIX-only, runs in child
    """Apply resource limits in the forked child before exec (POSIX only)."""
    try:
        import resource

        # 10s CPU, 512 MB address space, no new files written.
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        _mem = 512 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (_mem, _mem))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        os.setsid()
    except Exception:
        # Never let hardening failure crash the child before exec.
        pass


@ToolRegistry.register("code_interpreter")
class CodeInterpreterTool(BaseTool):
    """Execute Python code after AST validation, in a hardened subprocess."""

    tool_id = "code_interpreter"

    def __init__(self, timeout: int = 30, max_output: int = 10000):
        self._timeout = timeout
        self._max_output = max_output

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_interpreter",
            description=(
                "Execute Python code and return the output."
                " Code is AST-validated and runs in a hardened, isolated"
                " subprocess (no imports of os/sys/subprocess/network, no"
                " file IO, no eval/exec)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                },
                "required": ["code"],
            },
            category="code",
            required_capabilities=["code:execute"],
            metadata={"structured_allow_object_text": True},
        )

    def execute(self, **params: Any) -> ToolResult:
        code = params.get("code", "")
        if not code:
            return ToolResult(
                tool_name="code_interpreter",
                content="No code provided.",
                success=False,
            )

        # Security check — reject before running, by AST structure.
        try:
            _validate_ast(code)
        except SyntaxError as exc:
            return ToolResult(
                tool_name="code_interpreter",
                content=f"SyntaxError: {exc}",
                success=False,
            )
        except UnsafeCodeError as exc:
            return ToolResult(
                tool_name="code_interpreter",
                content=f"Blocked by code validation: {exc}",
                success=False,
            )

        # Sanitized environment — drop inherited secrets/tokens.
        safe_env = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        # ``preexec_fn`` only exists / is safe on POSIX.
        preexec = _child_limits if os.name == "posix" else None

        try:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-S", "-c", code],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=safe_env,
                cwd=os.environ.get("OPENJARVIS_CODE_CWD") or None,
                preexec_fn=preexec,  # noqa: PLW1509 - intentional child hardening
            )
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            if len(output) > self._max_output:
                output = output[: self._max_output] + "\n... (output truncated)"
            return ToolResult(
                tool_name="code_interpreter",
                content=output or "(no output)",
                success=result.returncode == 0,
                metadata={"returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name="code_interpreter",
                content=f"Execution timed out after {self._timeout} seconds.",
                success=False,
            )
        except Exception as exc:
            return ToolResult(
                tool_name="code_interpreter",
                content=f"Execution error: {exc}",
                success=False,
            )


__all__ = ["CodeInterpreterTool", "UnsafeCodeError"]
