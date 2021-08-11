#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The setup script."""

from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

# requirements = [
#     'Click>=6.0', 'numpy', 'scipy', 'matplotlib','nbsphinx',
# ]

requirements = []

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
]

setup(
    author="Luis Miguel Sanchez Brea",
    author_email='optbrea@ucm.es',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Optical Diffraction and Interference (scalar and vectorial)",
    entry_points={
        'console_scripts': [
            'diffractio=diffractio.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    # long_description=readme + '\n\n' + history,
    long_description=readme,
    include_package_data=True,
    keywords=[
        'diffractio', 'optics', 'diffraction', 'interference',
        'Rayleigh-Sommerfeld', 'Beam Propagation Method', 'BPM', 'WPM'
    ],
    name='diffractio',
    packages=find_packages(include=['diffractio']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://bitbucket.org/optbrea/diffractio/src/master/',
    version='0.0.13',
    zip_safe=False,
)
