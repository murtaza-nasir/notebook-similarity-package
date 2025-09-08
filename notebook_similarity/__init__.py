"""
Notebook Similarity Detector Package
A tool for detecting similar or identical submissions in Jupyter notebooks.
"""

from .detector import NotebookSimilarityDetector
from .analyzer import analyze_directory

__version__ = "1.0.0"
__all__ = ["NotebookSimilarityDetector", "analyze_directory"]