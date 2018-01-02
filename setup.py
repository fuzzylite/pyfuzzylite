from distutils.core import Command, setup

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
        raise SystemExit(result)


setup(
    name='pyfuzzylite',
    version='7.0',
    description='a fuzzy logic control in Python',
    long_description=long_description,
    keywords='fuzzy logic control',
    url='https://github.com/fuzzylite/pyfuzzylite',
    author='Juan Rada-Vilela, PhD',
    author_email='jcrada@fuzzylite.com',
    maintainer='Juan Rada-Vilela, PhD',
    maintainer_email='jcrada@fuzzylite.com',
    license='GNU General Public License 3',
    packages=['fuzzylite'],
    package_dir={'fuzzylite': '.'},
    platforms=['OS Independent'],
    provides='pyfuzzylite',
    classifiers=[
        'Development Status :: 1 - Planning',
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
