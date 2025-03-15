"""Type definitions for syntherela.

This module contains type definitions used throughout the syntherela package.
"""

from typing import Dict
import pandas as pd

# Type alias for a dictionary mapping table names to pandas DataFrames
Tables = Dict[str, pd.DataFrame]
