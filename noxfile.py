"""pyfuzzylite (TM), a fuzzy logic control library in Python.

Copyright (C) 2010-2023 FuzzyLite Limited. All rights reserved.
Author: Juan Rada-Vilela, Ph.D. <jcrada@fuzzylite.com>.

This file is part of pyfuzzylite.

pyfuzzylite is free software: you can redistribute it and/or modify it under
the terms of the FuzzyLite License included with the software.

You should have received a copy of the FuzzyLite License along with
pyfuzzylite. If not, see <https://github.com/fuzzylite/pyfuzzylite/>.

pyfuzzylite is a trademark of FuzzyLite Limited
fuzzylite is a registered trademark of FuzzyLite Limited.
"""

import nox

nox.options.sessions = ["check", "freeze", "install", "lint", "test"]


@nox.session(python=False)
def check(session: nox.Session) -> None:
    """Check the `pyproject.toml` is valid."""
    session.run(*"poetry check".split(), external=True)


@nox.session(python=False)
def format(session: nox.Session) -> None:
    """Run code formatting."""
    files = ["fuzzylite/", "tests/", "noxfile.py"]

    session.run(*"black --version".split(), external=True)
    session.run("black", *files, external=True)

    session.run(*"ruff --version".split(), external=True)
    session.run(*"ruff --fix".split(), *files, external=True)


@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """Run static code analysis and checks format is correct."""
    files = ["fuzzylite/", "tests/", "noxfile.py"]

    session.run(*"black --version".split(), external=True)
    session.run(*"black --check".split(), *files, external=True)

    session.run(*"ruff check".split(), *files, external=True)
    session.run(*"ruff --version".split(), external=True)

    session.run(*"pyright --version".split(), external=True)
    session.run("pyright", external=True)

    session.run(*"mypy --version".split(), external=True)
    session.run("mypy", *files, external=True)


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
    session.run(*"pip freeze".split(), external=True)


@nox.session(python=False)
def test(session: nox.Session) -> None:
    """Run the tests in the project."""
    session.run(*"coverage run -m pytest".split(), external=True)
    session.run(*"coverage report".split(), external=True)


@nox.session
def test_publish(session: nox.Session) -> None:
    """Build the distributable and upload it to testpypi."""
    session.run(*"rm -rf dist/".split(), external=True)
    session.run(*"poetry build".split(), external=True)
    session.run(*"twine check --strict dist/*".split(), external=True)
    session.run(
        *"twine upload --repository testpypi dist/* --config-file .pypirc --verbose".split(),
        external=True,
    )


@nox.session
def publish(session: nox.Session) -> None:
    """Build the distributable and upload it to pypi."""
    session.run(*"rm -rf dist/".split(), external=True)
    session.run(*"poetry build".split(), external=True)
    session.run(*"twine check --strict dist/*".split(), external=True)
    session.run(
        *"twine upload --repository pypi dist/* --config-file .pypirc --verbose".split(),
        external=True,
    )
