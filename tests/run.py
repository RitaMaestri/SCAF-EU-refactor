import pandas as pd
from pathlib import Path

# Import the function from the lib
from lib import compare_results


compare_results(
        file_a="/home/rita/Documents/Tesi/Projects/SCAF-EU-refactor/results/test(26-02-2026_15:32).csv",
        file_b="results/benchmark_1.csv",
        value_col="2020"
    )
