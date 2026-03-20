
import pandas as pd
import numpy as np
from collections.abc import Iterable

def EXIOBASE_flow_extractor(
    dataframe,
    row_indexes,
    col_indexes,
    new_row_indexes=None,
    new_col_indexes=None,
):
    """Extract a sub-matrix from an EXIOBASE MultiIndex DataFrame.

    Selects rows and columns identified by (category, subcategory) tuples from a
    DataFrame with a 2-level MultiIndex on both axes. When only a single row or
    column is selected the result is squeezed to a 1-D Series. The caller may
    optionally supply replacement index/column labels for the extracted slice.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Source DataFrame with 2-tuple MultiIndex on both rows and columns.
    row_indexes : iterable of (str, str)
        Sequence of (category, subcategory) tuples identifying the rows to extract.
    col_indexes : iterable of (str, str)
        Sequence of (category, subcategory) tuples identifying the columns to extract.
    new_row_indexes : iterable, optional
        Replacement labels for the extracted rows. Must match the length of
        ``row_indexes``.
    new_col_indexes : iterable, optional
        Replacement labels for the extracted columns. Must match the length of
        ``col_indexes``.

    Returns
    -------
    pd.Series or pd.DataFrame
        A Series when exactly one row or one column is selected; a DataFrame
        otherwise.

    Raises
    ------
    TypeError
        If ``row_indexes`` or ``col_indexes`` is not an iterable of 2-tuples.
    ValueError
        If a tuple does not have length 2, or if a replacement index sequence
        has the wrong length.
    """

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

    submatrix = dataframe.loc[row_labels, col_labels].astype(float)

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


def replace_unnamed_multiindex_labels(dataframe):
    """Replace auto-generated ``'Unnamed: …'`` level labels in a MultiIndex DataFrame.

    Pandas sometimes assigns ``'Unnamed: <n>'`` strings when reading files whose
    MultiIndex levels lack explicit names. This function replaces every such
    label with an empty string on both the row and column axes.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame whose MultiIndex labels should be cleaned.

    Returns
    -------
    pd.DataFrame
        A copy of ``dataframe`` with ``'Unnamed: …'`` labels replaced by ``''``.
        Non-MultiIndex axes are left unchanged.
    """

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
