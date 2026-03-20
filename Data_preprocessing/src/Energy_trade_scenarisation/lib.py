import pandas as pd
import numpy as np
from collections.abc import Iterable

def convert_remind_price_to_exiobase(price_2017_usd_per_gj, mapping_path):
    """Convert REMIND prices (2017 USD/GJ) to EXIOBASE units (2020 M EUR/EJ)."""
    conversion_values = pd.read_csv(mapping_path, index_col=0)["value"]

    usd_2017_to_2020 = float(conversion_values.loc["US$ 2017-2020 inflation"])
    usd_to_eur_2020 = float(conversion_values.loc["US$ 2020 - EUR 2020"])
    eur_per_gj_to_m_eur_per_ej = float(conversion_values.loc["EUR/GJ - M EUR/EJ"])

    total_conversion_factor = (
        usd_2017_to_2020 * usd_to_eur_2020 * eur_per_gj_to_m_eur_per_ej
    )

    return float(price_2017_usd_per_gj) * total_conversion_factor


def vol_prices_timeseries_to_df(
    import_volume_ts,
    export_volume_ts,
    import_price_ts,
    export_price_ts,
    volume_unit,
    price_unit,
    import_price_conversion_factor,
    export_price_conversion_factor,
):
    """Combine import/export volume and price time series into a labelled DataFrame.

    Stacks four time series (import volume, export volume, import price, export
    price) into a single DataFrame with IAMC-style label columns prepended.
    Prices are rescaled by the provided conversion factors before stacking.

    Parameters
    ----------
    import_volume_ts : pd.Series
        Import energy volume time series indexed by year.
    export_volume_ts : pd.Series
        Export energy volume time series indexed by year.
    import_price_ts : pd.Series
        Import energy price time series indexed by year (before conversion).
    export_price_ts : pd.Series
        Export energy price time series indexed by year (before conversion).
    volume_unit : str
        Unit string for the volume rows (e.g. ``'EJ/yr'``).
    price_unit : str
        Unit string for the price rows (e.g. ``'M EUR/EJ'``).
    import_price_conversion_factor : float
        Multiplicative factor applied to ``import_price_ts``.
    export_price_conversion_factor : float
        Multiplicative factor applied to ``export_price_ts``.

    Returns
    -------
    pd.DataFrame
        DataFrame with label columns (Model, Scenario, Region, Variable, Sector,
        Unit) followed by one numeric column per year.
    """
    years = list(import_volume_ts.index)

    labels_df = pd.DataFrame(
        {   
            "Model": ["REMIND-EXIOBASE"] * 4,
            "Scenario": ["SSP2-NPi2025"] * 4,
            "Region": ["EUR"] * 4,
            "Variable": [
                "Import|Energy",
                "Export|Energy",
                "Import|Energy Price",
                "Export|Energy Price",
            ],
            "Sector": ["ENERGY"] * 4,
            "Unit": [volume_unit, volume_unit, price_unit, price_unit],
        }
    )

    values_df = pd.DataFrame(
        np.vstack(
            [
                import_volume_ts.to_numpy(),
                export_volume_ts.to_numpy(),
                (import_price_ts * import_price_conversion_factor).to_numpy(),
                (export_price_ts * export_price_conversion_factor).to_numpy(),
            ]
        ),
        columns=years,
    )

    return pd.concat([labels_df, values_df], axis=1)


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
    """Filter a REMIND results DataFrame to the requested variables, years, and region.

    Parameters
    ----------
    dataframe : pd.DataFrame
        REMIND output table with at least the columns Model, Scenario, Region,
        Variable, Unit, and one column per year.
    variables_names : iterable
        Variable names to keep. ``NaN`` entries are silently ignored.
    years : iterable of int or float
        Years to include. Values are cast to ``int`` for column matching.
    region : str
        REMIND region code to filter on (e.g. ``'EUR'``).

    Returns
    -------
    pd.DataFrame
        Subset of ``dataframe`` containing only the matching rows and the
        requested year columns, with the index reset.
    """

    variables_list = [v for v in variables_names if pd.notna(v)]
    index_columns = ["Model", "Scenario", "Region", "Variable", "Unit"]
    requested_years = [str(int(y)) for y in years]

    year_columns = [col for col in dataframe.columns if str(col) in requested_years]

    return dataframe[
        (dataframe["Region"] == region)
        & (dataframe["Variable"].isin(variables_list))
    ][index_columns + year_columns].reset_index(drop=True)


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