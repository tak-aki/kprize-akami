import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import tomli
from packaging.requirements import Requirement
from packaging.version import parse

from .utils import CommandResult, PythonImplementation, PythonVersion, VersionConstraints


class UVManager:
    """Manages a UV virtual environment with persistent shell session support.

    This class provides functionality to create, manage, and interact with UV virtual
    environments. It supports both context manager and direct usage patterns, maintains
    a persistent shell session, and provides methods for package installation and
    command execution. This class was designed to be used within a Kaggle notebook
    to allow the creation of small environments for recreating swe-bench
    github issue environments.

    Attributes:
        venv_path (Path): Absolute path to the virtual environment
        python_version (str): Python version being used (e.g. "3.10")
        env_ready (bool): Whether the environment is ready for use

    Example:
        >>> # Using as context manager
        >>> with UVManager("./my_venv", python_version="3.10") as uv:
        ...     result = uv.pip_install("requests")
        ...     assert result.success

        >>> # Direct usage
        >>> uv = UVManager("./other_venv")
        >>> uv.initialize()
        >>> result = uv.send("python --version")
        >>> print(result.stdout)
        Python 3.10.x
        >>> uv.cleanup()
    """

    def __init__(self, venv_path: str | Path, python_version: str = "3.10", env_vars: dict[str, str] | None = None):
        """Initialize the UV environment manager.

        Args:
            venv_path (str | Path): Path where virtual environment should be created
            python_version (str): Python version to use (e.g. "3.10")
            env_vars (dict[str, str] | None): Additional environment variables to set in the shell

        Note:
            This doesn't create the environment immediately. Call initialize()
            or use as context manager to create and activate the environment.
        """
        self.venv_path = Path(venv_path).absolute()
        self.python_version = python_version
        self._shell = None
        self._logger = logging.getLogger(self.__class__.__name__)
        self.env_ready = False

        # Base environment variables
        self._env_vars = {
            "UV_LINK_MODE": "copy",  # Prevent hardlink warnings
            "VIRTUAL_ENV": str(self.venv_path),
            "PATH": f"{self.venv_path}/bin:{os.environ.get('PATH', '')}",
        }
        if env_vars:
            self._env_vars.update(env_vars)

    def __enter__(self) -> "UVManager":
        """Initialize environment when used as context manager."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Cleanup when exiting context manager."""
        self.cleanup()

    def _verify_environment(self) -> bool:
        """Verify that the UV environment is properly set up and functional.

        Returns:
            bool: True if environment is ready, False otherwise
        """
        try:
            # Check that $VIRTUAL_ENV matches self.venv_path
            venv_check = self.send("echo $VIRTUAL_ENV", bypass_env_check=True)
            if not venv_check.stdout.strip() == str(self.venv_path):
                self._logger.warning(
                    "VIRTUAL_ENV does not match the expected path. "
                    f"Expected: {self.venv_path}, Got: {venv_check.stdout}"
                )
                return False

            # Verify Python is accessible and correct version substring
            py_check = self.send("python --version", bypass_env_check=True)
            if not py_check.success:
                self._logger.warning("Running 'python --version' failed.")
                return False
            # Check if the declared python_version (e.g. '3.10') is in the output
            if self.python_version not in py_check.stdout:
                self._logger.warning(
                    f"Python version mismatch. Expected string '{self.python_version}' in '{py_check.stdout.strip()}'"
                )
                return False

            # Try importing a basic module and check sys.prefix
            import_check = self.send('python -c "import sys; print(sys.prefix)"', bypass_env_check=True)
            if not import_check.success:
                self._logger.warning("Failed to import and print sys.prefix.")
                return False
            if str(self.venv_path) not in import_check.stdout:
                self._logger.warning(f"sys.prefix does not match the expected path: {import_check.stdout}")
                return False

            return True

        except Exception as e:
            self._logger.warning(f"Environment verification failed: {e}")
            return False

    def _initialize_shell(self) -> None:
        """Initialize a persistent shell session with UV environment setup.

        Raises:
            RuntimeError: If shell initialization fails
        """
        if self._shell is not None:
            return

        self._shell = subprocess.Popen(
            ["bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **self._env_vars},
        )

        # Setup environment variables
        for name, value in self._env_vars.items():
            self._run_in_shell(f'export {name}="{value}"')

        # Activate using UV
        stdout, stderr = self._run_in_shell(f'eval "$(uv venv {self.venv_path})"')
        if stderr:
            self._logger.warning(f"UV environment activation warning: {stderr}")

    def _run_in_shell(self, command: str) -> tuple[str, str]:
        """Execute a command in the persistent shell session.

        Args:
            command (str): Shell command to execute

        Returns:
            Tuple of (stdout, stderr) from command execution
        """
        if not self._shell:
            self._initialize_shell()

        terminator = f"__CMD_DONE_{id(command)}__"
        full_command = f"{command}; echo {terminator}; echo {terminator} >&2"

        self._shell.stdin.write(full_command + "\n")
        self._shell.stdin.flush()

        def read_until_terminator(pipe) -> str:
            output = []
            while True:
                line = pipe.readline()
                if not line or terminator in line:
                    break
                output.append(line)
            return "".join(output).rstrip()

        stdout = read_until_terminator(self._shell.stdout)
        stderr = read_until_terminator(self._shell.stderr)

        return stdout, stderr

    def initialize(self) -> None:
        """Create and initialize the UV virtual environment.

        This method:
            (1) Creates the virtual environment directory
            (2) Sets up the UV environment
            (3) Initializes the shell session
            (4) Verifies the environment is working

        Raises:
            RuntimeError: If environment creation or verification fails
        """
        self._logger.info(f"Creating UV environment at {self.venv_path}")

        try:
            # Create the virtual environment directory
            self.venv_path.mkdir(parents=True, exist_ok=True)

            # Create the venv using uv
            subprocess.run(
                ["uv", "venv", "--python", self.python_version, str(self.venv_path)],
                capture_output=True,
                text=True,
                check=True,
            )

            # Initialize shell and verify
            self._initialize_shell()
            if not self._verify_environment():
                raise RuntimeError("Environment verification failed.")
            self.env_ready = True

            if not self.env_ready:
                raise RuntimeError("Environment verification failed after waiting")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create UV environment: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Error setting up UV environment: {str(e)}")

    def send(self, command: str, cwd: Path | str | None = None, bypass_env_check: bool = False) -> CommandResult:
        """Sends any arbitrary command to be executed in the UV environment.

        Args:
            command (str): The command to execute
            cwd (Path | str | None): Working directory for the command
            bypass_env_check (bool, optional): Whether to bypass the env setup check
                Only really used to allow initial setup commands to pass into the env.

        Returns:
            CommandResult containing command output and status

        Example:
            >>> uv = UVManager("./my_venv")
            >>> uv.initialize()
            >>> result = uv.send("python --version")
            >>> assert result.success
            >>> print(result.stdout)
            Python 3.10.x
        """
        # (1) We cannot run commands until the env is setup
        if not self.env_ready and not bypass_env_check:
            raise RuntimeError("Environment not ready. Call initialize() first.")

        # (2) Change current directory if cwd is passed
        if cwd:
            self._run_in_shell(f"cd {cwd}")

        # (3) Run the command and get the outputs, errors and return code
        stdout, stderr = self._run_in_shell(command)
        retcode_out, _ = self._run_in_shell("echo $?")

        # (4) Change back to original directory if cwd is passed
        if cwd:
            self._run_in_shell("cd -")

        try:
            returncode = int(retcode_out.strip())
        except ValueError:
            returncode = -1

        return CommandResult(command=command, returncode=returncode, stdout=stdout, stderr=stderr)

    def run(self, script: str | Path, args: list[str] | None = None) -> CommandResult:
        """Alias to uv_run

        Args:
            script (str | Path): Path to the Python script to run
            args (list[str] | None): List of arguments to pass to the script

        Returns:
            CommandResult containing script output and status

        Example:
            >>> uv = UVManager("./my_venv")
            >>> uv.initialize()
            >>> result = uv.run("script.py", ["--arg1", "value1"])
        """
        return self.uv_run(script, args)  # Forwarding arguments properly

    def uv_run(self, script: str | Path, args: list[str] | None = None) -> CommandResult:
        """Run a Python script using UV's run command.

        Args:
            script (str | Path): Path to the Python script to run
            args (list[str] | None): List of arguments to pass to the script

        Returns:
            CommandResult containing script output and status

        Example:
            >>> uv = UVManager("./my_venv")
            >>> uv.initialize()
            >>> result = uv.uv_run("script.py", ["--arg1", "value1"])
        """
        cmd = ["uv", "run"]
        if args:
            cmd.extend(args)
        cmd.append(str(script))

        return self.send(" ".join(cmd))

    def pip_install(
        self, package: str, editable: bool = False, cwd: Path | str | None = None, verbose: bool = True
    ) -> CommandResult:
        """Install a package using UV's pip interface.

        Args:
            package (str): Package specification (name, path, or requirements file)
            editable (bool): If True, install in editable mode (-e flag)
            verbose (bool): If True, print installation progress

        Returns:
            CommandResult containing installation output and status

        Example:
            >>> uv = UVManager("./my_venv")
            >>> uv.initialize()
            >>> result = uv.pip_install("requests")
            >>> assert result.success
        """
        if not self.env_ready:
            raise RuntimeError("Environment not ready. Call initialize() first.")

        # If package is a string, split it safely, or handle it as a list
        if isinstance(package, str):
            package_list = shlex.split(package)
        else:
            package_list = package

        cmd = ["uv", "pip", "install"]
        if editable:
            cmd.append("-e")
        cmd.extend(package_list)

        result = self.send(" ".join(cmd), cwd=cwd)

        if verbose:
            if result.stdout:
                self._logger.info(result.stdout)
            if result.stderr:
                if "Installed" in result.stderr or "Resolved" in result.stderr or "Using Python" in result.stderr:
                    self._logger.info(result.stderr)
                else:
                    self._logger.error(result.stderr)

        return result

    def cleanup(self) -> None:
        """Clean up resources and terminate the shell session.

        This should be called when done using the environment if not using
        the context manager.
        """
        if self._shell:
            self._shell.terminate()
            self._shell = None
        self.env_ready = False


class PythonVersionDetector:
    """Detects appropriate Python versions for a project using modern parsing approaches.

    Attributes:
        python_versions (list[PythonVersion]):
            Available Python versions from the Universal Python Installer
        _logger (logging.Logger): Internal logger instance
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self.python_versions = self._get_available_python_versions()

    def _parse_uv_python_list(self, output: str) -> list[PythonVersion]:
        """Parse the output of 'uv python list' command.

        Args:
            output (str): The output string from 'uv python list'.

        Returns:
            list[PythonVersion]:
                A list of PythonVersion objects.
        """
        # (1) Initialize the list of versions
        versions = []

        # (2) Parse the output line by line
        for line in output.splitlines():
            # (3a) Skip empty lines
            if not line.strip():
                continue

            # (3b) Split the line into parts
            parts = line.split(maxsplit=1)

            # (3c) Skip if no parts
            if not parts:
                continue

            # (3d) Extract version info and availability
            version_info = parts[0]
            availability = parts[1] if len(parts) > 1 else "<download available>"

            # (3e) Match the version info
            match = re.match(
                r"(cpython|pypy)-(\d+\.\d+\.\d+(?:a\d+)?)"
                r"(\+freethreaded)?-linux-x86_64-gnu",
                version_info,
            )

            # (3f) Skip if no match
            if not match:
                continue

            # (3g) Extract the implementation, version, and freethreaded info
            impl, version, freethreaded = match.groups()
            implementation = PythonImplementation.CPYTHON if impl == "cpython" else PythonImplementation.PYPY

            # (3h) Check if the version is installed
            is_installed = not availability.startswith("<download")
            path = availability if is_installed else None

            # (3i) Append the version to the list if it's installed and available
            if path and " -> " in path:
                path = path.split(" -> ")[0].strip()

            # (3j) Append the version to the list
            versions.append(
                PythonVersion(
                    implementation=implementation,
                    version=version,
                    is_installed=is_installed,
                    path=path,
                    is_freethreaded=bool(freethreaded),
                )
            )

        # (4) Return the list of versions
        return versions

    def _get_available_python_versions(self) -> list[PythonVersion]:
        """Get list of available Python versions from UV.

        Returns:
            list[PythonVersion]:
                A list of PythonVersion objects.
        """
        # (1) Run the 'uv python list' command
        try:
            result = subprocess.run(["uv", "python", "list"], capture_output=True, text=True, check=True)
            # (2) Parse the output
            return self._parse_uv_python_list(result.stdout)
        # (3) Handle any errors
        except subprocess.CalledProcessError as e:
            self._logger.warning(f"Failed to list Python versions: {e.stderr}")
            return []

    def parse_pyproject_toml(self, path: Path) -> VersionConstraints | None:
        """Parse Python version constraints from pyproject.toml using proper TOML parser.

        Args:
            path (Path): The path to the pyproject.toml file.

        Returns:
            VersionConstraints | None:
                The parsed version constraints or None if not found.
        """
        # (1) Attempt to parse the pyproject.toml file
        try:
            # (2a) Load the TOML data
            data = tomli.loads(path.read_text())

            # (2b) Check project.requires-python (PEP 621)
            if "project" in data and "requires-python" in data["project"]:
                return VersionConstraints.from_specifier_string(
                    data["project"]["requires-python"], "pyproject.toml", confidence=1.0
                )

            # (2c) Check poetry dependencies (looks for "tool.poetry.dependencies.python")
            if "tool" in data and "poetry" in data:
                if "dependencies" in data["tool"]["poetry"]:
                    if "python" in data["tool"]["poetry"]["dependencies"]:
                        return VersionConstraints.from_specifier_string(
                            data["tool"]["poetry"]["dependencies"]["python"], "pyproject.toml (poetry)", confidence=0.9
                        )

            # (2d) Check PDM dependencies (looks for "tool.pdm.python")
            if "tool" in data and "pdm" in data:
                if "python" in data["tool"]["pdm"]:
                    return VersionConstraints.from_specifier_string(
                        data["tool"]["pdm"]["python"], "pyproject.toml (pdm)", confidence=0.9
                    )
        # (3) Handle any errors
        except Exception as e:
            self._logger.warning(f"Error parsing pyproject.toml: {e}")

        # (4) Return None if no constraints found
        return None

    def parse_setup_py(self, path: Path) -> VersionConstraints | None:
        """Parse Python version constraints from setup.py file.

        Uses multiple strategies to detect version constraints:
            1. Analyzes Python version classifiers
            2. Checks for explicit python_requires in setup arguments
            3. Infers version range from classifier information

        Args:
            path (Path): Path to the setup.py file to analyze

        Returns:
            VersionConstraints | None: Parsed version constraints if found, None otherwise
        """
        # (1) Read and parse the setup.py file
        try:
            content = path.read_text()

            # (2a) Look for Python version classifiers with single quotes
            classifiers = re.findall(r"'Programming Language :: Python :: (\d+\.\d+)'", content)

            # (2b) If none found, try double quotes
            if not classifiers:
                classifiers = re.findall(r'"Programming Language :: Python :: (\d+\.\d+)"', content)

            # (3) Process classifier versions if found
            if classifiers:
                # (3a) Filter to only Python 3 versions
                versions = [v for v in classifiers if not v.startswith("2")]

                if versions:
                    # (3b) Get min and max supported versions
                    min_ver = min(versions, key=lambda x: parse(x))
                    max_ver = max(versions, key=lambda x: parse(x))

                    # (3c) Create constraint up to next major version
                    major, minor = map(int, max_ver.split("."))
                    next_minor = f"{major}.{minor + 1}"
                    spec = f">={min_ver},<{next_minor}"

                    return VersionConstraints.from_specifier_string(spec, "setup.py (classifiers)", confidence=0.7)

            # (4) Check for explicit python_requires
            requires_match = re.search(r'python_requires\s*=\s*[\'"]([^\'"]+)[\'"]', content)
            if requires_match:
                return VersionConstraints.from_specifier_string(
                    requires_match.group(1), "setup.py (python_requires)", confidence=0.9
                )

        # (5) Handle any parsing errors
        except Exception as e:
            self._logger.warning(f"Error parsing setup.py: {e}")

        return None

    def parse_setup_cfg(self, path: Path) -> VersionConstraints | None:
        """Parse Python version constraints from setup.cfg.

        Args:
            path (Path): The path to the setup.cfg file.

        Returns:
            VersionConstraints | None:
                The parsed version constraints or None if not found.
        """
        # (1) Attempt to parse the setup.cfg file
        try:
            # (2a) Load the setup.cfg data
            from configparser import ConfigParser

            config = ConfigParser()
            config.read(path)

            # (2b) Check for "options.python_requires" in the "metadata" section (PEP 518)
            if "options" in config:
                if "python_requires" in config["options"]:
                    return VersionConstraints.from_specifier_string(
                        config["options"]["python_requires"], "setup.cfg", confidence=0.8
                    )
        # (3) Handle any errors
        except Exception as e:
            self._logger.warning(f"Error parsing setup.cfg: {e}")

        # (4) Return None if no constraints found
        return None

    def parse_requirements_txt(self, path: Path) -> VersionConstraints | None:
        """Parse Python version constraints from requirements.txt using packaging.requirements.

        Args:
            path (Path): The path to the requirements.txt file.

        Returns:
            VersionConstraints | None:
                The parsed version constraints or None if not found.
        """
        # (1) Attempt to parse the requirements.txt file
        try:
            # (2a) Read the requirements.txt file
            content = path.read_text()
            constraints = []

            # (2b) Parse each line in the file (ignoring comments and empty lines)
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # (2c) Attempt to parse the requirement (ignoring any errors and looking for python_version)
                try:
                    req = Requirement(line)
                    if "python_version" in str(req.marker):
                        # Extract the version constraint from the marker
                        marker_str = str(req.marker)
                        version_part = re.search(r'python_version([^"\']+)["\']([^"\']+)["\']', marker_str)
                        if version_part:
                            op, ver = version_part.groups()
                            constraints.append(f"{op}{ver}")
                except Exception:
                    continue

            # (2d) Return the constraints if found
            if constraints:
                return VersionConstraints.from_specifier_string(
                    ",".join(constraints), "requirements.txt", confidence=0.7
                )

        # (3) Handle any errors
        except Exception as e:
            self._logger.warning(f"Error parsing requirements.txt: {e}")

        # (4) Return None if no constraints found
        return None

    def get_project_constraints(self, repo_path: Path | str) -> VersionConstraints | None:
        """Get Python version constraints from all project configuration files.

        Checks multiple config files in order of reliability:
            1. pyproject.toml (PEP 621)
            2. setup.cfg
            3. setup.py
            4. requirements.txt

        Args:
            repo_path (Path | str): Path to the repository root

        Returns:
            VersionConstraints | None: Version constraints if found, None otherwise
        """
        # (1) Ensure path is a Path object
        repo_path = Path(repo_path)

        # (2) Define config files to check in priority order
        config_files = [
            (repo_path / "pyproject.toml", self.parse_pyproject_toml),
            (repo_path / "setup.cfg", self.parse_setup_cfg),
            (repo_path / "setup.py", self.parse_setup_py),
            (repo_path / "requirements.txt", self.parse_requirements_txt),
        ]

        # (3) Try each config file in order
        for file_path, parser in config_files:
            if file_path.exists():
                if constraints := parser(file_path):
                    self._logger.info(
                        f"Found constraints in {file_path.name}: "
                        f"{constraints.specifier_set} "
                        f"(confidence: {constraints.confidence})"
                    )
                    return constraints

        # (4) No constraints found in any file
        return None

    def select_python_version(
        self,
        repo_path: Path | str,
        python_fallback_version: str = "3.10",
        newest_allowed_is_fallback: bool = True,
    ) -> PythonVersion:
        """Select the most appropriate Python version for a project.

        Uses the following selection process:
            1. Gets project version constraints from config files
            2. Filters available CPython versions to stable releases
            3. Filters versions to those matching constraints
            4. Selects newest compatible version
            5. Falls back to specified version if no match
            6. If specified version is newer than fallback optionally set to fallback

        Args:
            repo_path (Path | str): Path to the repository root
            python_fallback_version (str): Version to use if no compatible version found

        Returns:
            PythonVersion: Selected Python version object
        """
        # (1) Get project constraints
        constraints = self.get_project_constraints(repo_path)

        # (2) Filter to stable CPython versions
        cpython_versions = [
            v
            for v in self.python_versions
            if v.implementation == PythonImplementation.CPYTHON and not re.search(r"a|b|rc", v.version)
        ]

        # (3) Apply version constraints if found
        if constraints:
            try:
                # (3a) Filter to versions matching constraints
                compatible_versions = []
                for version in cpython_versions:
                    try:
                        if version.parsed_version in constraints.specifier_set:
                            compatible_versions.append(version)
                            self._logger.debug(f"Version {version.version} matches constraints")
                        else:
                            self._logger.debug(f"Version {version.version} excluded by constraints")
                    except Exception as e:
                        self._logger.warning(f"Error checking version {version.version}: {e}")
            except Exception as e:
                self._logger.warning(f"Error applying constraints: {e}")
                compatible_versions = cpython_versions
        else:
            compatible_versions = cpython_versions

        # (4) Sort compatible versions newest first
        sorted_versions = sorted(compatible_versions, key=lambda v: v.parsed_version, reverse=True)

        # (5) Return newest compatible version or fallback
        if not sorted_versions:
            self._logger.warning(f"No compatible versions found, using fallback {python_fallback_version}")
            return PythonVersion(
                implementation=PythonImplementation.CPYTHON,
                version=f"{python_fallback_version}.0",
                is_installed=False,
                path=None,
            )

        # (6) Return the selected newest version that meets requirements
        selected = sorted_versions[0]
        if selected.base_version > python_fallback_version and newest_allowed_is_fallback:
            self._logger.info(f"Selected version {python_fallback_version}.0 because actual selected version is newer.")
            return PythonVersion(
                implementation=PythonImplementation.CPYTHON,
                version=f"{python_fallback_version}.0",
                is_installed=False,
                path=None,
            )
        self._logger.info(f"Selected version {selected.version} from {len(sorted_versions)} compatible versions")
        return selected
