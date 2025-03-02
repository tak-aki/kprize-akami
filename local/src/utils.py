import io
import json
import logging
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import Version, parse


@contextmanager
def suppress_logging_below(level):
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


class PythonImplementation(Enum):
    """Enumeration of Python implementations."""

    CPYTHON = "cpython"
    PYPY = "pypy"


@dataclass
class PythonVersion:
    """Represents a Python version with its implementation and availability.

    Attributes:
        implementation (PythonImplementation): The Python implementation (e.g., CPython, PyPy)
        version (str): The full version string (e.g., '3.10.12')
        is_installed (bool): Whether the version is installed on the system
        path (str | Path | None): The path to the Python executable
        is_freethreaded (bool): Whether the Python version is free-threaded
            Free-threaded Python allows multiple threads to run simultaneously.
    """

    implementation: PythonImplementation
    version: str
    is_installed: bool
    path: str | Path | None = None
    is_freethreaded: bool = False

    @property
    def base_version(self) -> str:
        """Returns the base version (e.g., '3.10' from '3.10.12')."""
        return ".".join(self.version.split(".")[:2])

    @property
    def parsed_version(self) -> Version:
        """Returns a packaging.version.Version object for comparison."""
        return parse(self.version)


@dataclass
class VersionConstraints:
    """Represents Python version constraints with metadata.

    Attributes:
        min_version (str | None): Minimum version constraint
        max_version (str | None): Maximum version constraint
        specifier_set (SpecifierSet): Set of version specifiers
        source_file (str): The file where the constraints were extracted from
        confidence (float): Confidence level of the constraints (0.0 to 1.0)
    """

    min_version: str | None
    max_version: str | None
    specifier_set: SpecifierSet
    source_file: str
    confidence: float  # 0.0 to 1.0

    @classmethod
    def from_specifier_string(cls, spec_str: str, source_file: str, confidence: float = 1.0) -> "VersionConstraints":
        """Create VersionConstraints from a version specifier string.

        Args:
            spec_str (str): The version specifier string
            source_file (str): The file where the constraints were extracted from
            confidence (float): Confidence level of the constraints (0.0 to 1.0)

        Returns:
            VersionConstraints: The parsed version constraints
        """
        # (1) Parse the version specifier string
        spec_set = SpecifierSet(spec_str)

        # (2) Extract min and max versions from specifiers
        min_ver = None
        max_ver = None
        # (2a) Iterate over the specifiers
        for spec in spec_set:
            # (2b) Skip non-version specifiers
            ver_str = spec.version
            # (2c) Check for minimum and maximum versions
            if spec.operator in (">=", ">"):
                # (2d) Update min_version if needed
                if min_ver is None or parse(ver_str) > parse(min_ver):
                    min_ver = ver_str
            # (2e) Check for maximum version
            elif spec.operator in ("<=", "<"):
                # (2f) Update max_version if needed
                if max_ver is None or parse(ver_str) < parse(max_ver):
                    max_ver = ver_str
        # (3) Create and return the VersionConstraints object
        return cls(
            min_version=min_ver,
            max_version=max_ver,
            specifier_set=spec_set,
            source_file=source_file,
            confidence=confidence,
        )


@dataclass
class CommandResult:
    """Holds information about a subprocess command result.

    Args:
        command (str): The command that was executed.
        returncode (int): The return code of the command.
        stdout (str): The standard output of the command.
        stderr (str): The standard error of the command.
    """

    command: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Indicates if returncode == 0."""
        return self.returncode == 0

    def __str__(self) -> str:
        """Informal string representation, used for user-facing display."""
        if self.success:
            return "CommandResult(command={!r}, returncode={!r}, success={!r}, stdout={!r})".format(
                self.command, self.returncode, self.success, self.stdout
            )
        else:
            return "CommandResult(command={!r}, returncode={!r}, success={!r}, stdout={!r}, stderr={!r})".format(
                self.command, self.returncode, self.success, self.stdout, self.stderr
            )

    def __repr__(self) -> str:
        """Official string representation, used for debugging."""
        return str(self)

    def raise_for_status(self) -> None:
        """Raise a subprocess.CalledProcessError if the command failed.

        Raises:
            subprocess.CalledProcessError: If the command's return code is non-zero.
        """
        if not self.success:
            raise subprocess.CalledProcessError(
                returncode=self.returncode, cmd=self.command, output=self.stdout, stderr=self.stderr
            )


@dataclass
class EnvironmentConfig:
    """Configuration for UV environment setup.

    Args:
        python_version (str, optional):
            Python version to use (e.g. "3.10")
        base_dir (str, optional):
            Base directory for environments to be stored.
            NOTE: We default to the /kaggle/tmp directory as it
                  has access to a larger storage volume.
        pytest_options (str, optional):
            Additional pytest options to pass.
    """

    python_version: str = "3.10"
    base_dir: str = "/kaggle/tmp"
    pytest_options: str = ""


@dataclass
class TestResult:
    """Results from running tests in the environment.

    Args:
        success (bool): Whether tests passed
        output (str): Test output
        error (str): Error output if any
        duration (float): Test duration in seconds
    """

    success: bool
    output: str
    error: str
    duration: float


@dataclass
class SWEBenchInstance:
    """Represents a single instance from the SWE-Bench dataset.

    Attributes:
        repo (str): Repository URL or identifier (e.g. "owner/repo")
        instance_id (str): Unique identifier for this instance
        base_commit (str): The commit hash where the bug exists
        patch (str): The code changes that fix or modify the bug
        test_patch (str): The test changes associated with the fix
        problem_statement (str): Description of the bug/issue
        hints_text (str, optional): Additional hints or context about the bug
        created_at (datetime): When the instance was created
        version (str): Version identifier for this instance
        fail_to_pass (list[str]): Whether this instance should go from failing to passing
        pass_to_pass (list[str]): Whether this instance should maintain passing status
        environment_setup_commit (str): Commit hash used for environment setup
    """

    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str | None
    created_at: datetime
    version: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    environment_setup_commit: str

    @classmethod
    def from_df_row(cls, row: Any) -> "SWEBenchInstance":
        """Create an instance from a pandas DataFrame row.

        Args:
            row: A single row (pd.Series or dict) from the SWE-Bench-Lite dataset.

        Returns:
            A SWEBenchInstance object populated with row data.
        """
        return cls(
            repo=row["repo"],
            instance_id=str(row["instance_id"]),
            base_commit=row["base_commit"],
            patch=row["patch"],
            test_patch=row["test_patch"],
            problem_statement=row["problem_statement"],
            hints_text=row["hints_text"],
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
            version=row["version"],
            fail_to_pass=json.loads(row["FAIL_TO_PASS"]),
            pass_to_pass=json.loads(row["PASS_TO_PASS"]),
            environment_setup_commit=row["environment_setup_commit"],
        )

    @property
    def github_repo_url(self) -> str:
        """Constructs the full GitHub URL for the repo.

        Returns:
            str: The GitHub repo URL.
        """
        return os.path.join("https://github.com", self.repo)

    @property
    def github_pull_url(self) -> str:
        """Constructs the full GitHub URL for the PR that fixes the issue.

        Returns:
            str: The GitHub URL for the PR.
        """
        pull_number = self.instance_id.rsplit("-", 1)[-1]
        return os.path.join(self.github_repo_url, "pull", pull_number)

    @property
    def repo_at_base_commit_url(self) -> str:
        """Constructs the full GitHub URL for the base commit (where the bug exists).

        Returns:
            str: The GitHub URL for the base commit.
        """
        return os.path.join(self.github_repo_url, "tree", self.base_commit)

    @property
    def repo_at_environment_setup_commit_url(self) -> str:
        """Constructs the full GitHub URL for the environment setup commit.

        Returns:
            str: The GitHub URL for the environment setup commit.
        """
        return os.path.join(self.github_repo_url, "tree", self.environment_setup_commit)


def extract_updated_tests(diff_text: str) -> list[str]:
    """Extracts pytest-compatible test identifiers from a git diff.

    This function identifies modified test files and their respective test functions
    using regex, returning them in the pytest node ID format: `path/to/test_file.py::test_function_name`.
    If a test file is modified but no specific test functions are added/modified,
    returns just the file path to run all tests in that file.

    Args:
        diff_text (str): The git diff output containing file changes and modifications.

    Returns:
        list[str]: A list of pytest test identifiers in the format `file_path::test_function_name`
                   or just `file_path` for modified test files.
    """
    # Patterns for identifying test files and functions
    test_file_pattern = re.compile(r"^[+]{3} b/(.+?test.*?\.py)", re.MULTILINE)
    test_func_pattern = re.compile(r"^\+\s*def (test_[a-zA-Z0-9_]+)", re.MULTILINE)

    # Pattern for parameterized tests (they might have different signature)
    param_test_pattern = re.compile(
        r"^\+.*?@pytest\.mark\.parametrize.*?\n.*?\n*?^\+\s*def (test_[a-zA-Z0-9_]+)", re.MULTILINE | re.DOTALL
    )

    test_identifiers = set()

    # Split the diff into file chunks
    diff_files = diff_text.split("diff --git ")

    for diff_chunk in diff_files[1:]:  # Skip the first empty chunk
        # Extract the test file path
        file_matches = test_file_pattern.findall(diff_chunk)
        if not file_matches:
            continue

        file_path = file_matches[0]

        # Skip renamed files without content changes
        if "similarity index 100%" in diff_chunk:
            continue

        # Extract all test functions (both regular and parameterized)
        test_funcs = set(test_func_pattern.findall(diff_chunk))
        param_funcs = set(param_test_pattern.findall(diff_chunk))

        # Combine all found test functions
        all_funcs = test_funcs.union(param_funcs)

        if all_funcs:
            # If we found specific test functions, add them with the file path
            for func_name in all_funcs:
                test_identifiers.add(f"{file_path}::{func_name}")
        else:
            # If the test file was modified but no specific test functions were found,
            # add the file path to run all tests in that file
            test_identifiers.add(file_path)

    return sorted(list(test_identifiers))


class DualOutput:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = io.StringIO()

    def write(self, message):
        self.original_stdout.write(message)
        self.buffer.write(message)

    def flush(self):
        self.original_stdout.flush()
        self.buffer.flush()


@contextmanager
def capture_stdout():
    original_stdout = sys.stdout
    dual_output = DualOutput(original_stdout)
    sys.stdout = dual_output

    try:
        yield dual_output.buffer
    finally:
        sys.stdout = original_stdout


def save_json(data: Any, path: str | Path) -> None:
    """Save data to a JSON file.

    Args:
        data: The data to save.
        path (str | Path): The path to the JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
