import pandas as pd
from pathlib import Path

# Import the function from the lib
from lib import compare_results


compare_results(
        file_a="/home/rita/Documents/Tesi/Projects/SCAF-EU-refactor/results/2015-2020REMIND-7sectors(23-02-2026_18:50).csv",
        file_b="results/benchmark_1.csv",
        value_col="2020"
    )
