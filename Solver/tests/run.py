import pandas as pd
from pathlib import Path

# Import the function from the lib
from lib import compare_results


compare_results(
        file_a="Solver/results/test(21-03-2026_11:44).csv",
        file_b="Solver/results/benchmark3.csv",
        value_col="2025"
    )
