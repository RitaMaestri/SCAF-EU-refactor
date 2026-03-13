import csv
import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections.abc import Iterable
import json
from pathlib import Path


def EXIOBASE_flow_extractor(
    dataframe,
    row_indexes,
    col_indexes,
    new_row_indexes=None,
    new_col_indexes=None,
):


    def _check_tuples(indexes, arg_name):
        if isinstance(indexes, (str, bytes)) or not isinstance(indexes, Iterable):
            raise TypeError(f"{arg_name} must be an iterable of 2-item tuples")

        normalized = []
        for idx in indexes:
            if not isinstance(idx, tuple):
                raise TypeError(
                    f"Each element in {arg_name} must be a tuple, got {type(idx).__name__}"
                )
            if len(idx) != 2:
                raise ValueError(
                    f"Each tuple in {arg_name} must have length 2, got {len(idx)}"
                )
            normalized.append(idx)

        return normalized

    row_labels = _check_tuples(row_indexes, "row_indexes")
    col_labels = _check_tuples(col_indexes, "col_indexes")

    submatrix = dataframe.loc[row_labels, col_labels]

    if new_row_indexes is not None:
        new_row_indexes = list(new_row_indexes)
        if len(new_row_indexes) != len(row_labels):
            raise ValueError(
                "new_row_indexes length must match the number of selected rows"
            )

    if new_col_indexes is not None:
        new_col_indexes = list(new_col_indexes)
        if len(new_col_indexes) != len(col_labels):
            raise ValueError(
                "new_col_indexes length must match the number of selected columns"
            )

    if len(row_labels) == 1 and len(col_labels) >= 1:
        one_dim = submatrix.iloc[0, :]
        if new_col_indexes is not None:
            one_dim.index = new_col_indexes
        if new_row_indexes is not None:
            one_dim.name = new_row_indexes[0]
        return one_dim

    if len(col_labels) == 1 and len(row_labels) >= 1:
        one_dim = submatrix.iloc[:, 0]
        if new_row_indexes is not None:
            one_dim.index = new_row_indexes
        if new_col_indexes is not None:
            one_dim.name = new_col_indexes[0]
        return one_dim

    if new_row_indexes is not None:
        submatrix.index = new_row_indexes
    if new_col_indexes is not None:
        submatrix.columns = new_col_indexes

    return submatrix


def REMIND_time_series_extractor(dataframe, variables_names, years, region):

    variables_list = [v for v in variables_names if pd.notna(v)]
    index_columns = ["Model", "Scenario", "Region", "Variable", "Unit"]
    requested_years = [str(int(y)) for y in years]

    year_columns = [col for col in dataframe.columns if str(col) in requested_years]

    return dataframe[
        (dataframe["Region"] == region)
        & (dataframe["Variable"].isin(variables_list))
    ][index_columns + year_columns].reset_index(drop=True)


def replace_unnamed_multiindex_labels(dataframe):

    def _clean_multiindex(multiindex):
        return pd.MultiIndex.from_tuples(
            tuple(
                "" if isinstance(level, str) and level.startswith("Unnamed:") else level
                for level in key
            )
            for key in multiindex
        )

    cleaned = dataframe.copy()

    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = _clean_multiindex(cleaned.columns)
    if isinstance(cleaned.index, pd.MultiIndex):
        cleaned.index = _clean_multiindex(cleaned.index)

    return cleaned