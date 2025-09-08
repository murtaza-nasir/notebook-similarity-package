"""
Setup script for Notebook Similarity Detector package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="notebook-similarity-detector",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for detecting similar or identical submissions in Jupyter notebooks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/notebook-similarity-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Education",
        "Topic :: Software Development :: Plagiarism Detection",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    entry_points={
        "console_scripts": [
            "notebook-similarity=notebook_similarity.cli:main",
        ],
    },
    include_package_data=True,
)