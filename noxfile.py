"""pyfuzzylite: a fuzzy logic control library in Python.

This file is part of pyfuzzylite.

Repository: https://github.com/fuzzylite/pyfuzzylite/

License: FuzzyLite License

Copyright: FuzzyLite by Juan Rada-Vilela. All rights reserved.
"""

import nox

nox.options.sessions = ["check", "freeze", "install", "lint", "test"]

PYTHON_FILES = ["fuzzylite/", "tests/", "noxfile.py"]
MARKDOWN_FILES = ["README.md", "THANKS.md", "docs/"]


@nox.session(python=False)
def check(session: nox.Session) -> None:
    """Check the `pyproject.toml` is valid."""
    session.run(*"poetry check".split(), external=True)


@nox.session(python=False)
def format(session: nox.Session) -> None:
    """Run code formatting."""
    formatters = {
        # "ruff": format_ruff,
        "black": format_black,
    }
    posargs = list(session.posargs)
    if not posargs:
        posargs = list(formatters.keys())
    for formatter in posargs:
        formatters[formatter](session)


@nox.session(python=False)
def format_black(session: nox.Session) -> None:
    """Run code formatting with black."""
    files = PYTHON_FILES
    session.run(*"black --version".split(), external=True)
    session.run("black", *files, external=True)


@nox.session(python=False)
def format_ruff(session: nox.Session) -> None:
    """Run code formatting with ruff."""
    files = PYTHON_FILES
    session.run(*"ruff --version".split(), external=True)
    session.run(*"ruff format".split(), *files, external=True)


@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """Run static code analysis and checks format is correct."""
    linters = {
        # "ruff": lint_ruff,
        "black": lint_black,
        "pyright": lint_pyright,
        "mypy": lint_mypy,
        "markdown": lint_markdown,
    }
    # Case 1: posargs is empty, run all linters
    # Case 2: posargs is not empty, run only the specified linters
    posargs = list(session.posargs)
    if not posargs:
        posargs = list(linters.keys())
    for linter in posargs:
        linters[linter](session)


@nox.session(python=False)
def lint_black(session: nox.Session) -> None:
    """Run black linter."""
    files = PYTHON_FILES
    session.run(*"black --version".split(), external=True)
    session.run(*"black --check".split(), *files, external=True)


@nox.session(python=False)
def lint_ruff(session: nox.Session) -> None:
    """Run ruff linter."""
    files = PYTHON_FILES
    session.run(*"ruff --version".split(), external=True)
    session.run(*"ruff check".split(), *files, external=True)


@nox.session(python=False)
def lint_pyright(session: nox.Session) -> None:
    """Run pyright linter."""
    session.run(*"pyright --version".split(), external=True)
    session.run("pyright", external=True)


@nox.session(python=False)
def lint_mypy(session: nox.Session) -> None:
    """Run mypy linter."""
    files = PYTHON_FILES
    session.run(*"mypy --version".split(), external=True)
    session.run("mypy", *files, external=True)


@nox.session(python=False)
def lint_markdown(session: nox.Session) -> None:
    """Run markdown linter."""
    files = MARKDOWN_FILES
    session.run(*f"pymarkdown scan {' '.join(files)}".split(), external=True)


@nox.session(python=False)
def install(session: nox.Session) -> None:
    """Install the project using poetry."""
    session.run(*"poetry install -v --no-interaction".split(), external=True)


@nox.session(python=False)
def install_upgrade(session: nox.Session) -> None:
    """Install the project using poetry and upgraded dependencies."""
    session.run(*"poetry lock -v".split(), external=True)
    session.run(*"poetry install -v --no-interaction".split(), external=True)


@nox.session(python=False)
def freeze(session: nox.Session) -> None:
    """Print all the versions of dependencies."""
    session.run(*"poetry show".split(), external=True)


@nox.session(python=False)
def test(session: nox.Session) -> None:
    """Run the tests in the project."""
    session.run(*"coverage run -m pytest".split(), external=True)
    session.run(*"coverage report".split(), external=True)


@nox.session(python=False)
def benchmark(session: nox.Session) -> None:
    """Run the benchmarks."""
    if "codspeed" in session.posargs:
        session.run(*"pytest tests/test_benchmark_codspeed.py".split(), external=True)
    else:
        session.run(*"pytest tests/test_benchmark_pytest.py".split(), external=True)


@nox.session(python=False)
def docs(session: nox.Session) -> None:
    """Build the documentation and deploy if passed `publish` in arguments."""
    if "publish" in session.posargs:
        session.run(*"mkdocs gh-deploy --strict --clean --force".split(), external=True)
    else:
        session.run(*"mkdocs build --strict --clean".split(), external=True)


@nox.session
def publish(session: nox.Session) -> None:
    """Build the distributable and upload it to pypi."""
    repository = "testpypi" if "test" in session.posargs else "pypi"
    session.run(*"rm -rf dist/".split(), external=True)
    session.run(*"poetry build".split(), external=True)
    session.run(*"twine check --strict dist/*".split(), external=True)
    session.run(
        *f"twine upload --repository {repository} dist/* --config-file .pypirc --verbose".split(),
        external=True,
    )
