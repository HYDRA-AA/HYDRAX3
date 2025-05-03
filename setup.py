#!/usr/bin/env python3
"""
Setup script for HYDRA encryption package.
"""

from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hydra-encryption",
    version="0.1.0",
    author="HYDRA Encryption Project Contributors",
    author_email="example@email.com",  # Replace with actual contact
    description="A novel encryption algorithm for high security against classical and quantum threats",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hydra-encryption",  # Replace with actual URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Security :: Cryptography",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        # No external dependencies for core functionality
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.5b2",
            "isort>=5.8",
            "flake8>=3.9",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    keywords="encryption, cryptography, security, quantum-resistant, cipher",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/hydra-encryption/issues",
        "Documentation": "https://github.com/yourusername/hydra-encryption/wiki",
        "Source Code": "https://github.com/yourusername/hydra-encryption",
    },
    entry_points={
        "console_scripts": [
            "hydra=src.cli:main",
        ],
    },
)
