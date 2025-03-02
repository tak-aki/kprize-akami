import logging
import shutil
import subprocess
import uuid
from pathlib import Path

from .environment import PythonVersionDetector, UVManager
from .utils import CommandResult, SWEBenchInstance


class GitHubRepo:
    """Handles operations with a GitHub repository (cloning, checkout, patching).

    Attributes:
        instance_id (str): A unique string identifier for the instance (aka GitHub issue).
        repo_name (str): The name of Github repository
        org_name (str): The name of owner of the Github repository
        repo_url (str): The full GitHub repository URL (e.g., "https://github.com/user/repo.git")
        env_setup_commit_hash (str): The commit hash used for environment setup.
        base_commit_hash (str | None): The commit hash for the relevant issue.
        root_dir (Path): Path to the root directory where all temporary directories will be.
        root_repo_path (Path): Path to this repo's specific temporary directory
        _logger (logging.Logger): Internal logger instance
    """

    instance_id: str
    repo_name: str
    org_name: str
    repo_url: str
    env_setup_commit_hash: str
    base_commit_hash: str
    root_dir: Path
    root_repo_path: Path
    _logger: logging.Logger

    def __init__(
        self,
        instance_id: str,
        repo_name: str,
        org_name: str,
        repo_url: str,
        env_setup_commit_hash: str,
        base_commit_hash: str | None = None,
        root_dir: str | Path = "/kaggle/tmp",
    ) -> None:
        """Initializes the GitHubRepo object.

        Args:
            instance_id (str):
                 A unique string identifier for the instance (aka GitHub issue).
            repo_name (str):
                The name of Github repo (repo)
            org_name (str):
                The name of owner of the Github repo (repo)
            repo_url (str):
                The full GitHub repository URL (e.g., "https://github.com/user/repo.git")
            env_setup_commit_hash (str):
                The commit hash used for environment setup.
            base_commit_hash (str, optional):
                The commit hash for the relevant issue.
            root_dir (str | Path, optional):
                The root directory where all temporary directories will be stored.
        """
        # (1) Set attributes related to commit hashes
        self.env_setup_commit_hash = env_setup_commit_hash
        self.base_commit_hash = base_commit_hash or env_setup_commit_hash

        # (2) Set attributes related to the repository name and URL
        self.instance_id = instance_id
        self.repo_name = repo_name
        self.org_name = org_name
        self.repo_url = repo_url

        # (3) Set attributes related to the root directory and paths
        self.root_dir = Path(root_dir)
        self.root_repo_path = self.root_dir

        # (4) Initialize the logger
        self._logger = logging.getLogger(self.__class__.__name__)

    def find_free_directory(self, base_dir: str | Path | None = None) -> Path:
        """Finds a free directory in the base directory, creating one if necessary.

        Args:
            base_dir (str | Path, optional):
                Base directory to search within.

        Returns:
            Path:
                The path object pointing to the newly created directory.

        Raises:
            RuntimeError: If no free directory could be found after multiple attempts.
        """
        base_path = Path(base_dir or self.root_dir)
        for i in range(100):
            dir_name = f"temp_env_{i}"
            temp_dir = base_path / dir_name
            if not temp_dir.exists():
                temp_dir.mkdir(parents=True, exist_ok=True)
                return temp_dir
        raise RuntimeError("Could not find a free temporary directory")

    def clone_and_checkout(self, checkout_commit_hash: str | None = None) -> Path:
        """Clones the repository and checks out the specified commit.

        Args:
            checkout_commit_hash (str, optional):
                The commit hash that we will checkout.
                If not provided we will use the commit hash for the environment setup.

        Returns:
            The local filesystem path to the cloned repo.

        Raises:
            RuntimeError: If clone or checkout fails.
        """
        try:
            # (1) Clone the repository
            self.clone_repo()

            # (2) Checkout the specified commit
            self.checkout_commit(checkout_commit_hash or self.env_setup_commit_hash)

        # (3) Handle any errors
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Git clone/checkout error: {e.stderr}")
            raise RuntimeError(f"Git clone/checkout error: {e.stderr}") from e

        # (4) Return the path to the cloned repository
        return self.root_repo_path

    def clone_repo(self, force_reclone: bool = True) -> "GitHubRepo":
        """Clones the repository into the specified path.

        Args:
            force_reclone (bool, optional):
                Whether to force re-cloning the repository if it already exists.

        Returns:
            GitHubRepo: The instance itself (allows method chaining).
        """
        # (1) If repo (validated by .git) already exists, do nothing
        if (self.root_repo_path / ".git").exists():
            # (1a) If force_reclone is False, log a warning and return
            if not force_reclone:
                self._logger.warning(f"Repository path {self.root_repo_path} already exists. Skipping clone.")
                return self
            # (1b) If force_reclone is True, remove the existing directory and proceed.
            self._logger.warning(f"Repository path {self.root_repo_path} already exists. Recloning...")
            shutil.rmtree(self.root_repo_path)
            self.root_repo_path.mkdir(parents=True, exist_ok=True)

        # (2) Clone a single branch of the repository
        self._logger.info(f"Cloning {self.repo_url} to {self.root_repo_path}...")
        try:
            # Clone with --no-single-branch to get all branches
            subprocess.run(
                ["git", "clone", "--no-single-branch", self.repo_url, str(self.root_repo_path)],
                capture_output=True,
                text=True,
                check=True,
            )

            # Fetch all tags and remote branches
            subprocess.run(
                ["git", "fetch", "--all", "--tags", "--prune"],
                cwd=self.root_repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

        # (3) Handle any errors
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Git clone error: {e.stderr}")
            raise RuntimeError(f"Git clone error: {e.stderr}") from e

        # (4) Enable method chaining
        return self

    def checkout_commit(self, commit_hash: str) -> "GitHubRepo":
        """Checks out the specified commit hash in the cloned repository.

        Args:
            commit_hash (str): The commit hash to checkout.

        Returns:
            GitHubRepo: The instance itself (allows method chaining).
        """
        # (1) Check if the repository path exists
        if not (self.root_repo_path / ".git").exists():
            raise RuntimeError(f"Repository path {self.root_repo_path} does not exist. Clone the repo first.")

        # (2) Checkout the specified commit
        try:
            subprocess.run(
                ["git", "checkout", commit_hash], cwd=self.root_repo_path, capture_output=True, text=True, check=True
            )
            self._logger.info(f"Checked out commit {commit_hash}.")

        # (3) Handle any errors
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Git checkout error: {e.stderr}")
            raise RuntimeError(f"Git checkout error: {e.stderr}") from e

        # (4) Enable method chaining
        return self

    def apply_patch(self, patch_content: str) -> "GitHubRepo":
        """Applies a patch to the cloned repository.

        Args:
            patch_content (str):
                The diff/patch content as a string.

        Raises:
            RuntimeError: If patch application fails.
        """
        # (1) Write the patch content to a file in the temporary directory
        patch_file = self.root_repo_path / "local_changes.patch"
        patch_file.write_text(patch_content)

        # (2) Apply the patch
        self._logger.info(f"Applying patch at {patch_file}...")
        try:
            subprocess.run(
                ["git", "apply", patch_file.name], cwd=self.root_repo_path, capture_output=True, text=True, check=True
            )
            self._logger.info("Patch applied successfully.")
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Patch application error: {e.stderr}")
            raise RuntimeError(f"Patch application error: {e.stderr}") from e

        # Enable method chaining
        return self

    @classmethod
    def from_swebench_instance(cls, instance: SWEBenchInstance, root_dir: str | Path = "/kaggle/tmp") -> "GitHubRepo":
        """Creates a GitHubRepo instance from a SWEBenchInstance object.

        Args:
            instance (SWEBenchInstance):
                The SWE-Bench instance to create the GitHubRepo object from.

        Returns:
            GitHubRepo:
                A GitHubRepo object initialized with the instance's repo URL and commit hash.
        """
        org_name, repo_name = instance.repo.split("/")
        return GitHubRepo(
            instance_id=instance.instance_id,
            repo_name=repo_name.strip(),
            org_name=org_name.strip(),
            repo_url=instance.github_repo_url,
            base_commit_hash=instance.base_commit,
            env_setup_commit_hash=instance.environment_setup_commit,
            root_dir=root_dir,
        )

    def cleanup(self) -> None:
        """Cleans up the cloned repository directory."""
        self._logger.info(f"Cleaning up temporary repository directory {self.root_repo_path}...")
        shutil.rmtree(self.root_repo_path, ignore_errors=True)
        self._logger.info("Temporary repository directory cleaned up.")

    def __repr__(self) -> str:
        """String representation of the GitHubRepo object."""
        return (
            f"GitHubRepo("
            f"rood_dir={self.root_dir}, "
            f"root_repo_path={self.root_repo_path}, "
            f"repo_name={self.repo_name}, "
            f"org_name={self.org_name}, "
            f"repo_url={self.repo_url}, "
            f"base_commit_hash={self.base_commit_hash}"
            f"env_setup_commit_hash={self.env_setup_commit_hash})"
        )


class RepoUVManager(UVManager):
    """A specialized UVManager for tackling SWE-problems.

    This class can:
        - clone a GitHub repo
        - check out commits
        - detect Python versions
        - apply patches
        - install dependencies from the local cloned directory

    Inherits:
        UVManager: The base environment manager with persistent shell session.

    Attributes:
        github_repo (GitHubRepo): A reference to the GitHubRepo object managing our cloned repo.
        python_version_detector (PythonVersionDetector): The object responsible for identifying the python version to use.
        repo_path (Path | None): The local filesystem path where the repo is cloned.
        venv_name (str): The name of the virtual environment (usually unique and generated).
        venv_path (Path): The full path to the virtual environment including the name.

    Extended Features:
        - Automatic detection of Python version availability (via `uv python list`).
        - Support for installing dependencies from requirements.txt, pyproject.toml, or setup.py.
        - Ability to run pytest easily via `run_pytest`.
        - Optional patch application through the GitHubRepo reference.
    """

    def __init__(
        self,
        venv_dir_path: str | Path,
        github_repo: GitHubRepo,
        auto_detect_python: bool = True,
        fallback_python_version: str = "3.10",
        venv_name: str | None = None,
        env_vars: dict[str, str] | None = None,
        do_clone_and_checkout: bool = True,
    ):
        """Initialize the RepoUVManager.

        Args:
            venv_dir_path (str | Path):
                Path where the UV virtual environment should be created.
            github_repo (GitHubRepo):
                Manages the cloned repository (commit checkout, patching, etc.).
            auto_detect_python (bool, optional):
                Whether to auto-detect a suitable Python version from project constraints.
            fallback_python_version (str, optional):
                Used if no constraints are found or no matching version is available in 'uv python list'.
            venv_name (str, optional):
                The name of the virtual environment to create in venv_dir_path.
                If None, generate a default with generate_venv_name().
            env_vars (dict[str, str] | None, optional):
                Additional environment variables to set in the shell.
            do_clone_and_checkout (bool, optional):
                Whether to clone and checkout the env setup commit during init.
        """
        self._logger = logging.getLogger(self.__class__.__name__)

        # Initialize the github repo (for python versions and other stuff)
        self.github_repo = github_repo
        self.repo_path = github_repo.root_repo_path
        self.clone_and_checkout_repo()

        self.venv_name = venv_name or self.generate_venv_name()
        self.venv_path = Path(venv_dir_path) / self.venv_name

        # Initialize python detector and get the selected python version
        self.python_version_detector = PythonVersionDetector()
        self.selected_python = self.python_version_detector.select_python_version(
            repo_path=self.repo_path, python_fallback_version=fallback_python_version
        )

        super().__init__(venv_path=self.venv_path, python_version=self.selected_python.version, env_vars=env_vars)

    def generate_venv_name(self, instance_id: str | None = None, commit_hash: str | None = None) -> str:
        """Generates a unique environment name based on the repository, instance ID, and commit hash.

        Any arguments not provided are inferred from self.github_repo, where:
            - instance_id = self.github_repo.instance_id (if defined in GitHubRepo)
            - commit_hash = self.github_repo.env_setup_commit_hash

        Args:
            instance_id (str, optional):
                The repository identifier string, e.g. 'repo__user-repo-issue'.
            commit_hash (str, optional):
                The commit hash to reference.

        Returns:
            str: A unique environment name with a random UUID suffix.
        """
        instance_id = instance_id or getattr(self.github_repo, "instance_id", "repo")
        commit_hash = commit_hash or getattr(self.github_repo, "env_setup_commit_hash", "HEAD")

        # In case the repo includes slashes
        sanitized_instance = instance_id.replace("/", "_")
        return f"{sanitized_instance}-{commit_hash}-{uuid.uuid4().hex[:8]}"

    def clone_and_checkout_repo(self, commit_hash: str | None = None) -> Path:
        """Clone the GitHub repo (if not already) and check out the given commit hash.

        Args:
            commit_hash (str | None):
                The commit hash to check out. If None, uses the repo's default environment hash.

        Returns:
            Path: The local path to the cloned repository.
        """
        repo_path = self.github_repo.clone_and_checkout(checkout_commit_hash=commit_hash)
        self._logger.info(f"Repo cloned at {repo_path}")
        self.repo_path = repo_path  # This should already be set but we update just in case...
        return repo_path

    def install_repo_dependencies(self) -> CommandResult | None:
        """Install dependencies from the cloned repo by checking:
            (1) pyproject.toml build-system requirements
            (2) requirements.txt
            (3) pyproject.toml [project] table (install local project)
            (4) setup.py (install local project)
            (5) otherwise, do nothing

        We install in editable mode by default, so local code changes are reflected.

        Returns:
            CommandResult | None:
                The result of the pip install command, or None if no install was done.
        """
        if not self.env_ready:
            raise RuntimeError("Environment not ready. Call initialize() first.")

        if not self.repo_path:
            self._logger.warning("No repo is attached or cloned yet. Skipping dependency install.")
            return None

        result: CommandResult | None = None

        # (0) Upgrade pip and setup tools and whatnot
        self.pip_install("--upgrade pip", cwd=self.repo_path, verbose=True)
        self.pip_install("--upgrade setuptools wheel", cwd=self.repo_path, verbose=True)

        # (1) Build-System Requires: install them if present in pyproject.toml
        pyproj = self.repo_path / "pyproject.toml"
        if pyproj.exists():
            build_system_requires = self._get_build_system_requires(pyproj)
            if build_system_requires:
                self._logger.info(f"Installing build-system requirements: {build_system_requires}")
                build_cmd = " ".join(build_system_requires)
                build_result = self.pip_install(build_cmd, editable=True, cwd=self.repo_path, verbose=True)
                if not build_result.success:
                    self._logger.error("Failed to install build-system requirements!")
                    return build_result  # Early return if needed
            else:
                self._logger.debug("No build-system.requires found in pyproject.toml")

        # (2) requirements.txt and requirements-dev.txt
        req_file = self.repo_path / "requirements.txt"
        req_dev_file = self.repo_path / "requirements-dev.txt"
        if req_file.exists():
            self._logger.info("Installing dependencies from requirements.txt...")
            result = self.pip_install(["-r", str(req_file)], editable=True, cwd=self.repo_path)
            if req_dev_file.exists():
                self._logger.info("Installing dev dependencies from requirements-dev.txt...")
                dev_result = self.pip_install(["-r", str(req_dev_file)], editable=True, cwd=self.repo_path)
                if not dev_result.success:
                    self._logger.error("Failed to install dev dependencies!")
                    return dev_result
            return result

        # (3) pyproject.toml [project]
        if pyproj.exists():
            content = pyproj.read_text()
            if "[project]" in content:
                self._logger.info(
                    "Detected [project] table in pyproject.toml; installing with uv pip install . [editable]"
                )
                result = self.pip_install(".", editable=True, cwd=self.repo_path)
                return result
            else:
                # Rename pyproject.toml -> pyproject.toml.bak (effectively 'hiding' it)
                pyproj_backup = pyproj.with_name("pyproject.toml.bak")
                pyproj.rename(pyproj_backup)
                self._logger.info(
                    "pyproject.toml found but no [project] table (hiding via rename). Will check setup.py next."
                )

        # (4) Fallback to setup.py
        setup_py = self.repo_path / "setup.py"
        if setup_py.exists():
            self._logger.info("Installing local code via setup.py with uv pip install . (editable)")
            result = self.pip_install(".", editable=True, cwd=self.repo_path)
            return result

        # (5) No recognized files
        self._logger.info(
            "No recognized dependency file found (requirements.txt, pyproject.toml, or setup.py). Skipping installation."
        )
        return result

    def _get_build_system_requires(self, pyproject_path: Path) -> list[str]:
        """Extract build-system dependencies from pyproject.toml."""
        try:
            import tomllib  # Python 3.11+; use 'tomli' for older versions
        except ImportError:
            import tomli as tomllib

        requirements = []
        toml_data = tomllib.loads(pyproject_path.read_text())

        build_system = toml_data.get("build-system", {})
        if not build_system:
            return requirements  # No build-system table found

        requires_list = build_system.get("requires", [])
        for item in requires_list:
            requirements.append(f'"{item}"')  # Quote items to avoid shell parsing issues

        # Optionally, infer extras based on build-backend
        build_backend = build_system.get("build-backend")
        if build_backend == "setuptools.build_meta" and not any("wheel" in r for r in requirements):
            requirements.append('"wheel"')
        elif build_backend == "hatchling.build" and not any("editables" in r for r in requirements):
            requirements.append('"editables"')

        return requirements

    def run_pytest(self, test_path: str | None = None, extra_args: list[str] | None = None) -> CommandResult:
        """Convenience method to run pytest in the environment.

        Args:
            test_path (str | None):
                Specific test path or module to run (e.g. "tests/test_file.py::test_func").
                If None, runs all tests in the current repo_path.
            extra_args (list[str] | None):
                Additional command-line arguments (e.g. ["-v", "--pdb"]).

        Returns:
            CommandResult: Contains stdout, stderr, and return code.
        """
        if not self.env_ready:
            raise RuntimeError("Environment not ready. Call initialize() first.")
        if not self.repo_path:
            raise RuntimeError("No repo_path available; cannot run tests in an un-cloned repo.")

        cmd_tokens = ["pytest"]
        if test_path:
            cmd_tokens.append(test_path)
        if extra_args:
            cmd_tokens.extend(extra_args)

        # Build the final command string
        cmd_str = " ".join(cmd_tokens)
        self._logger.info(f"Running pytest: {cmd_str} (cwd={self.repo_path})")
        return self.send(cmd_str, cwd=self.repo_path)

    def apply_patch(self, patch_content: str) -> None:
        """Applies a patch to the cloned repository by delegating to GitHubRepo.

        Args:
            patch_content (str): The diff/patch content as a string.

        Raises:
            RuntimeError: If no GitHubRepo is set or patch application fails.
        """
        self._logger.info("Applying patch content via GitHubRepo...")
        self.github_repo.apply_patch(patch_content)

    def get_git_patch(self) -> str | None:
        """Generate a git patch from the current changes in the repo."""
        try:
            result = self.send("git diff", cwd=self.repo_path)
            if not result.success or not result.stdout.strip():
                return None

            # Check if there's at least one line starting with '+'
            lines = result.stdout.splitlines()
            if not any(line.startswith("+") for line in lines):
                return None

            # Otherwise return the stdout
            return result.stdout

        # Catch any errors and log and return None
        except Exception as e:
            self._logger.error(f"Failed to generate git patch: {e}")
            return None

    def remove_github_actions(self) -> None:
        """Remove the .github directory in the cloned repo, if it exists.

        This is done to prevent any unwanted GitHub Actions from interfering locally.
        """
        if not self.repo_path:
            self._logger.warning("No repo path is set, cannot remove .github folder.")
            return

        github_dir = self.repo_path / ".github"
        if github_dir.exists():
            import shutil

            self._logger.info(f"Removing GitHub Actions files at {github_dir}")
            shutil.rmtree(github_dir)
        else:
            self._logger.info("No .github directory found. Skipping removal.")

    def cleanup_repo(self) -> None:
        """Cleans up the cloned repository directory from disk (if desired).

        This is separate from environment cleanup, which ends the shell session.
        """
        if not self.repo_path:
            self._logger.warning("No repo path is set. Nothing to remove.")
            return

        import shutil

        if self.repo_path.exists():
            self._logger.info(f"Removing cloned repository at {self.repo_path}...")
            shutil.rmtree(self.repo_path, ignore_errors=True)
            self._logger.info("Repository folder removed.")

    def cleanup(self, remove_venv: bool = True) -> None:
        """Override cleanup() so we can remove the environment directory too, plus the repo.

        This method:
            - Terminates the persistent shell session from UVManager
            - Resets env_ready to False
            - Removes the cloned repo from disk
            - (Optionally, remove self.venv_path if you want the venv folder gone too)

        Args:
            remove_venv (bool): Whether to remove the environment directory
        """
        super().cleanup()  # Kills the shell, sets env_ready=False
        self.cleanup_repo()

        if remove_venv:
            if self.venv_path.exists():
                self._logger.info(f"Removing environment directory at {self.venv_path}...")
                shutil.rmtree(self.venv_path, ignore_errors=True)
                self._logger.info("Environment directory removed.")
            else:
                self._logger.info(f"No environment directory found at {self.venv_path}...")
