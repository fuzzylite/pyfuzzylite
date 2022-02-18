from pathlib import Path
from typing import List

import nox


@nox.session(python=False)
def check(session: nox.Session) -> None:
    """checks the `pyproject.toml` is valid"""
    session.run("poetry", "check", external=True)


@nox.session(python=False)
def install(session: nox.Session) -> None:
    """installs the project using poetry"""
    # lock all dependencies to the latest available compatible versions (faster than update)
    session.run("poetry", "lock", "-v", external=True)
    # install our project
    session.run("poetry", "install", "-v", external=True)


@nox.session(python=False)
def freeze(session: nox.Session) -> None:
    """prints all the versions of libraries"""
    session.run("poetry", "export", "--without-hashes", external=True)


@nox.session(python=False)
def test(session: nox.Session) -> None:
    """runs the tests in the project"""
    session.run(
        "coverage",
        "run",
        "-m",
        "pytest",
        "--cov=fuzzylite/",
        "tests/",
        external=True,
    )
    session.run(
        "coverage",
        "report",
        "-m",
        external=True,
    )

@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """runs static code analysis and checks format is correct"""
    session.run("mypy", "--version", external=True)
    session.run(
        "mypy", "fuzzylite/", "tests/", "--strict", external=True, success_codes=[0]
    )
    files = ["fuzzylite/", "tests/", "noxfile.py"]
    session.run("black", "--check", *files, external=True)
    session.run(
        "nbqa",
        "black",
        "--check",
        "-tpy36",
        *black_notebook_folders(),
        external=True,
    )


@nox.session(python=False)
def format(session: nox.Session) -> None:
    """runs code formatting"""
    files = ["fuzzylite/", "tests/", "noxfile.py"]
    session.run(
        "autoflake",
        "-r",
        "--in-place",
        "--remove-all-unused-imports",
        "--remove-unused-variables",
        "fuzzylite/",
        external=True,
    )
    session.run("isort", *files, external=True)
    session.run("black", *files, external=True)
    session.run(
        "nbqa",
        "black",
        "-tpy36",
        *black_notebook_folders(),
        "--nbqa-mutate",
        external=True,
    )


def black_notebook_folders() -> List[str]:
    """
    retrieves the list of notebook folders to lint (or format)
    """
    # include
    notebooks = [
        str(folder)
        for folder in Path("/tests/notebooks").glob("*.ipynb")
        if folder.is_dir()
    ]
    return notebooks


@nox.session(python=False)
def prepublish(session: nox.Session) -> None:
    import fuzzylite as fl
    import toml

    file = Path("pyproject.toml")
    pyproject = toml.load(str(file))

    pyproject["tool"]["poetry"]["name"] = fl.lib.name
    pyproject["tool"]["poetry"]["version"] = fl.lib.version
    pyproject["tool"]["poetry"]["description"] = fl.lib.description
    pyproject["tool"]["poetry"]["authors"] = [f"{fl.lib.author} <{fl.lib.author_email}>"]
    pyproject["tool"]["poetry"]["maintainers"] = [f"{fl.lib.author} <{fl.lib.author_email}>"]

    file.write_text(toml.dumps(pyproject))
