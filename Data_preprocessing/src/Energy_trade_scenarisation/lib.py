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