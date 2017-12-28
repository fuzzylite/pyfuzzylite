from distutils.core import setup

setup(
    name='fuzzylite',
    version='7.0',
    packages=['fuzzylite'],
    package_dir={'': 'src'},
    url='https://fuzzylite.com/python',
    license='FuzzyLite License',
    author='Juan Rada-Vilela, PhD',
    author_email='jcrada@fuzzylite.com',
    description=
    '''pyfuzzylite(TM) is a free and open-source fuzzy logic control library programmed in Python. The goal of 
pyfuzzylite is to easily design and efficiently operate fuzzy logic controllers following an object-oriented model 
without relying on external libraries.  

pyfuzzylite is the Python equivalent of the fuzzylite(R) library.

pyfuzzylite is a trademark of FuzzyLite Limited.
fuzzylite is a registered trademark of FuzzyLite Limited.'''
)
