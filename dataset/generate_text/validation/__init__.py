"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Validation module for generated text corpora.
"""

from .validate_corpora import main as validate_corpora
from .metrics import CorpusMetrics
from .error_detectors import ErrorDetector
from .report_generator import ReportGenerator

__all__ = [
    "validate_corpora",
    "CorpusMetrics",
    "ErrorDetector",
    "ReportGenerator",
]
