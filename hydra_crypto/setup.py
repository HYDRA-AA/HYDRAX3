#!/usr/bin/env python3
"""
HYDRA Cryptography Package Setup
"""

from setuptools import setup, find_packages

# Read the README file for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hydra-crypto',
    version='1.0.0',
    description='HYDRA encryption algorithm with JIT acceleration and reliable decryption',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='HYDRA Cryptography Team',
    author_email='hydra@example.com',
    url='https://github.com/hydra-crypto/hydra',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'numba>=0.53.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'flake8>=3.9.0',
            'mypy>=0.812',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Security :: Cryptography',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'hydra-encrypt=hydra_crypto.cli:main',
        ],
    },
)
