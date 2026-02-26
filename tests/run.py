import pandas as pd
from pathlib import Path

# Import the function from the lib
from lib import compare_results


compare_results(
        file_a="results/test(26-02-2026_16:02).csv",
        file_b="results/benchmark_1.csv",
        value_col="2020"
    )
