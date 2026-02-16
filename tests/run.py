import pandas as pd
from pathlib import Path

# Import the function from the lib
from lib import compare_results


compare_results(
        file_a="results/benchmark_1.csv",
        file_b="results/benchmark_0.csv",
        value_col="2020"
    )
