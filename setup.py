from distutils.core import setup

with open('README.md') as file:
    long_description = file.read()

setup(
    name='fuzzylite',
    version='7.0',
    description='a library for fuzzy logic control',
    long_description=long_description,
    url='https://fuzzylite.com/python',
    author='Juan Rada-Vilela, PhD',
    author_email='jcrada@fuzzylite.com',
    license='FuzzyLite License',
    packages=['fuzzylite'],
    package_dir={'fuzzylite': '.'}
)
