"""Statistical tests for comparing distributions of real and synthetic columns."""

from .chi_square_test import ChiSquareTest
from .kolmogorov_smirnov_test import KolmogorovSmirnovTest

__all__ = [
    "ChiSquareTest",
    "KolmogorovSmirnovTest",
]
