"""
 pyfuzzylite (TM), a fuzzy logic control library in Python.
 Copyright (C) 2010-2017 FuzzyLite Limited. All rights reserved.
 Author: Juan Rada-Vilela, Ph.D. <jcrada@fuzzylite.com>

 This file is part of pyfuzzylite.

 pyfuzzylite is free software: you can redistribute it and/or modify it under
 the terms of the FuzzyLite License included with the software.

 You should have received a copy of the FuzzyLite License along with
 pyfuzzylite. If not, see <http://www.fuzzylite.com/license/>.

 pyfuzzylite is a trademark of FuzzyLite Limited
 fuzzylite is a registered trademark of FuzzyLite Limited.
"""
from typing import List  # noqa: I202

import setuptools


class PyTest(setuptools.Command):
    user_options: List[str] = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        import unittest
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover('tests', pattern='test_*.py')
        result = unittest.TextTestRunner().run(test_suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)


def setup_package() -> None:
    import fuzzylite as fl

    setuptools.setup(
        name=fl.lib.name,
        version=fl.lib.version,
        description=fl.lib.description,
        long_description=fl.lib.summary,
        long_description_content_type='text/markdown',
        keywords=['fuzzy logic control', 'soft computing', 'artificial intelligence'],
        url=fl.lib.website,
        download_url='https://github.com/fuzzylite/pyfuzzylite.git',
        project_urls={
            'Home': fl.lib.website,
            'Documentation': 'https://www.fuzzylite.com/documentation',
            'Bug Tracker': 'https://github.com/fuzzylite/pyfuzzylite/issues',
            'Source Code': 'https://github.com/fuzzylite/pyfuzzylite',
        },
        author=fl.lib.author,
        author_email=fl.lib.author_email,
        maintainer=fl.lib.author,
        maintainer_email=fl.lib.author_email,
        license=fl.lib.license,
        packages=setuptools.find_packages(),
        # package_dir={'fuzzylite': '.'},
        # entry_points={
        #     'console_scripts': ['fuzzylite=fuzzylite:console']
        # },
        platforms=['OS Independent'],
        provides=[fl.lib.name],
        python_requires='>=3.6',
        # setup_requires=['pytest-runner'],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU Affero General Public License v3',
            'License :: Other/Proprietary License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Software Development :: Libraries',
        ],
        cmdclass={'test': PyTest},
        # zip_safe=True
    )


if __name__ == '__main__':
    setup_package()
