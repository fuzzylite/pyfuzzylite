from typing import List

import nox


@nox.session(python=False)
def check(session: nox.Session) -> None:
    """Check the `pyproject.toml` is valid."""
    session.run("poetry", "check", external=True)


@nox.session(python=False)
def format(session: nox.Session) -> None:
    """Run code formatting."""
    files = ["fuzzylite/", "tests/", "noxfile.py"]
    notebooks: List[str] = []

    # source code
    session.run("black", *files, external=True)
    session.run("ruff", "--fix", *files, external=True)

    # notebooks
    if notebooks:
        session.run(
            "nbqa",
            "black",
            *notebooks,
            external=True,
        )
        session.run(
            "nbqa",
            "ruff",
            "--ignore=E402",  # Module level import not at top of file
            "--fix",
            *notebooks,
            external=True,
        )


@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """Run static code analysis and checks format is correct."""
    files = ["fuzzylite/", "tests/", "noxfile.py"]
    notebooks: List[str] = []

    # source code
    session.run("black", "--check", *files, external=True)
    session.run("ruff", "check", *files, external=True)

    # mypy
    session.run("mypy", "--version", external=True)
    session.run(
        "mypy",
        *files,
        external=True,
        success_codes=[0],
    )
    # notebooks
    if notebooks:
        session.run("nbqa", "black", "--check", *notebooks, external=True)
        session.run(
            "nbqa",
            "ruff",
            "--ignore=E402",  # Module level import not at top of file
            *notebooks,
            external=True,
        )

        # session.run(
        #     "nbqa",
        #     "mypy",
        #     *notebooks,
        #     external=True,
        # )


@nox.session(python=False)
def install(session: nox.Session) -> None:
    """Install the project using poetry."""
    session.run(
        "poetry",
        "install",
        "-v",
        "--no-interaction",
        # "-E",
        # "numpy",
        external=True,
    )


@nox.session(python=False)
def install_upgrade(session: nox.Session) -> None:
    """Install the project using poetry and upgraded dependencies."""
    session.run("poetry", "lock", "-v", external=True)
    session.run(
        "poetry",
        "install",
        "-v",
        "--no-interaction",
        external=True,
    )


@nox.session(python=False)
def freeze(session: nox.Session) -> None:
    """Print all the versions of dependencies."""
    session.run("poetry", "export", "--without-hashes", external=True)


@nox.session(python=False)
def test(session: nox.Session) -> None:
    """Run the tests in the project."""
    session.run(
        "coverage",
        "run",
        "-m",
        "pytest",
        external=True,
    )
    session.run(
        "coverage",
        "report",
        external=True,
    )


@nox.session(python=False)
def prepublish(_: nox.Session) -> None:
    """Prepares to publish the distributable."""
    pass
    # import toml

    # import fuzzylite as fl

    # file = Path("pyproject.toml")
    # pyproject = toml.load(str(file))

    # pyproject["tool"]["poetry"]["name"] = fl.lib.name
    # pyproject["tool"]["poetry"]["version"] = fl.lib.version
    # pyproject["tool"]["poetry"]["description"] = fl.lib.description
    # pyproject["tool"]["poetry"]["authors"] = [
    #     f"{fl.lib.author} <{fl.lib.author_email}>"
    # ]
    # pyproject["tool"]["poetry"]["maintainers"] = [
    #     f"{fl.lib.author} <{fl.lib.author_email}>"
    # ]

    # file.write_text(toml.dumps(pyproject))
