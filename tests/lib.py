import pandas as pd
import numpy as np
from collections import defaultdict

def compare_results(
    file_a,
    file_b,
    value_col="2020",
    rtol=1e-6,
    atol=1e-12
):
    """
    Compare two CSV files based on row labels and a given value column.

    Assumptions:
    - The first column in each CSV contains the row labels (no column name).
    - The column `value_col` (e.g. "2020") exists in both files.
    - Labels may appear multiple times.
    - The order of appearance of labels matters.
    - The two CSV files may have a different number of rows.
    - Only labels appearing in BOTH files are compared.

    Behaviour:
    - For each label common to both files:
        * the number of occurrences must be the same
        * values in column `value_col` are compared in order
    - If any mismatch is found, a ValueError is raised.
    """

    # Read CSV files
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    # The first column (unnamed) is the label column
    label_col_a = df_a.columns[0]
    label_col_b = df_b.columns[0]

    # Check that the value column exists
    if value_col not in df_a.columns or value_col not in df_b.columns:
        raise ValueError(f"Column '{value_col}' not found in one of the files")

    # Build a dictionary: label -> list of values (order preserved)
    def build_label_dict(df, label_col):
        label_dict = defaultdict(list)
        for label, value in zip(df[label_col], df[value_col]):
            label_dict[label].append(value)
        return label_dict

    dict_a = build_label_dict(df_a, label_col_a)
    dict_b = build_label_dict(df_b, label_col_b)

    # Labels that appear in both files
    common_labels = set(dict_a.keys()) & set(dict_b.keys())

    if not common_labels:
        raise ValueError("No common labels found between the two files")

    # Compare values for each common label
    for label in common_labels:
        values_a = dict_a[label]
        values_b = dict_b[label]

        # Same number of occurrences required
        if len(values_a) != len(values_b):
            raise ValueError(
                f"Different number of occurrences for label '{label}': "
                f"{len(values_a)} vs {len(values_b)}"
            )

        # Compare values occurrence by occurrence (order matters)
        for i, (va, vb) in enumerate(zip(values_a, values_b)):
            if not np.isclose(va, vb, rtol=rtol, atol=atol):
                raise ValueError(
                    f"Mismatch for label '{label}', occurrence {i}:\n"
                    f"  file A ({value_col}): {va}\n"
                    f"  file B ({value_col}): {vb}"
                )

    print("OK: all common labels match.")
