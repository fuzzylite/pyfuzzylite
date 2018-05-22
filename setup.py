from distutils.core import Command, setup

import fuzzylite

with open('README.md') as file:
    long_description = file.read()


class PyTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import unittest
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover('tests', pattern='test_*.py')
        result = unittest.TextTestRunner().run(test_suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)


library = fuzzylite.Library()
setup(
    name=library.name,
    version=library.version,
    description=library.description,
    long_description=long_description,
    keywords='fuzzy logic control',
    url='https://github.com/fuzzylite/pyfuzzylite',
    author=library.author,
    author_email=library.author_email,
    maintainer=library.author,
    maintainer_email=library.author_email,
    license=library.license,
    packages=['fuzzylite'],
    package_dir={'fuzzylite': '.'},
    platforms=['OS Independent'],
    provides='pyfuzzylite',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
    ],
    cmdclass={'test': PyTest},
)
