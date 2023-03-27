import nox


@nox.session(python=False)
def check(session: nox.Session) -> None:
    """Check the `pyproject.toml` is valid."""
    session.run("poetry", "check", external=True)


@nox.session(python=False)
def format(session: nox.Session) -> None:
    """Run code formatting."""
    session.run("black", "--version", external=True)
    session.run("ruff", "--version", external=True)

    files = ["fuzzylite/", "tests/", "noxfile.py"]

    session.run("black", *files, external=True)
    session.run("ruff", "--fix", *files, external=True)


@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """Run static code analysis and checks format is correct."""
    session.run("black", "--version", external=True)
    session.run("ruff", "--version", external=True)
    session.run("mypy", "--version", external=True)

    files = ["fuzzylite/", "tests/", "noxfile.py"]

    session.run("black", "--check", *files, external=True)
    session.run("ruff", "check", *files, external=True)
    session.run(
        "mypy",
        *files,
        external=True,
    )


@nox.session(python=False)
def install(session: nox.Session) -> None:
    """Install the project using poetry."""
    session.run(
        "poetry",
        "install",
        "-v",
        "--no-interaction",
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
    session.run("pip", "freeze", external=True)


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


@nox.session
def test_publish(session: nox.Session) -> None:
    """Build the distributable and upload it to testpypi."""
    session.run("rm", "-rf", "dist/", external=True)
    session.run("poetry", "build", external=True)
    session.run("twine", "check", "--strict", "dist/*", external=True)
    session.run(
        "twine",
        "upload",
        "--repository",
        "testpypi",
        "dist/*",
        "--config-file",
        ".pypirc",
        "--verbose",
        external=True,
    )


@nox.session
def publish(session: nox.Session) -> None:
    """Build the distributable and upload it to pypi."""
    session.run("rm", "-rf", "dist/", external=True)
    session.run("poetry", "build", external=True)
    session.run("twine", "check", "--strict", "dist/*", external=True)
    session.run(
        "twine",
        "upload",
        "--repository",
        "pypi",
        "dist/*",
        "--config-file",
        ".pypirc",
        "--verbose",
        external=True,
    )
