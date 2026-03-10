import pandas as pd
from pathlib import Path

# Import the function from the lib
from lib import compare_results


compare_results(
        file_a="Solver/results/test(06-03-2026_17:19).csv",
        file_b="Solver/results/benchmark_1.csv",
        value_col="2020"
    )
