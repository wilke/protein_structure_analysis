"""Setup script for protein_compare package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read() if __import__('os').path.exists("README.md") else ""

setup(
    name="protein_compare",
    version="0.2.0",
    author="Protein Compare",
    description="Batch comparison tool for predicted protein structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "biopython>=1.81",
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "tmtools>=0.1",
        "pandas>=2.0",
        "click>=8.1",
        "joblib>=1.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "protein_compare=protein_compare.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
