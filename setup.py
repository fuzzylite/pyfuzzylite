from distutils.core import setup

with open('README.md') as file:
    long_description = file.read()

setup(
    name='pyfuzzylite',
    version='7.0',
    description='a fuzzy logic control in Python',
    long_description=long_description,
    keywords='fuzzy logic control',
    url='https://github.com/fuzzylite/pyfuzzylite',
    author='Juan Rada-Vilela, PhD',
    author_email='jcrada@fuzzylite.com',
    license='FuzzyLite License',
    packages=['fuzzylite'],
    package_dir={'fuzzylite': '.'},
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
    python_requires='>=3.6',
)
