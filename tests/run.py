import pandas as pd
from pathlib import Path

# Import the function from the lib
from lib import compare_results


compare_results(
        file_a="results/johansen2015-2020exoKnextREMIND-7sectors(04-02-2026_18:19).csv",
        file_b="results/results_for_comparison.csv",
        value_col="2020"
    )
