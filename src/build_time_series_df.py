import numpy as np 
import pandas as pd
from Variable_class import Variable
from Variables_specs import VARIABLES_SPECS



exogenous_data = "REMIND_exogenous_data_sectors"

growth_ratios_df = pd.read_csv(
    "data/"+exogenous_data+".csv") 

years = np.array([eval(i) for i in growth_ratios_df.columns[3:]])
year_cols = [str(y) for y in years]



def build_timeseries_df(VARIABLES_SPECS, year_cols):
    """Build the time-series template DataFrame from variable specifications.

    The resulting DataFrame contains one row per scalar element, vector entry,
    or matrix cell defined in ``VARIABLES_SPECS``. The first year column is
    filled with calibration values, while following years are initialized to NaN.

    Args:
        VARIABLES_SPECS: Dictionary of Variable objects keyed by variable name.
        year_cols: Ordered list of year column names as strings.

    Returns:
        A template DataFrame with columns:
        ``variable_name``, ``row_label``, ``col_label``, ``status``, and years.
    """
    rows = []

    for var_name, var in VARIABLES_SPECS.items():
        if var.dimension == "scalar":
            is_exo = bool(var.exo_mask[0])
            status = "exogenous" if is_exo else "endogenous"
            row = {
                "variable_name": var_name,
                "row_label": "",
                "col_label": "",
                "status": status,
                year_cols[0]: var.calibration_value,
            }
            for y in year_cols[1:]:
                row[y] = np.nan
            rows.append(row)

        elif var.dimension == "vector":
            for i, label in enumerate(var.idx_labels):
                is_exo = bool(var.exo_mask[i])
                status = "exogenous" if is_exo else "endogenous"
                row = {
                    "variable_name": var_name,
                    "row_label": label,
                    "col_label": "",
                    "status": status,
                    year_cols[0]: var.calibration_value[i],
                }
                for y in year_cols[1:]:
                    row[y] = np.nan
                rows.append(row)

        elif var.dimension == "matrix":
            rows_labels = var.idx_labels[0]
            cols_labels = var.idx_labels[1]
            for i, row_label in enumerate(rows_labels):
                for j, col_label in enumerate(cols_labels):
                    is_exo = bool(var.exo_mask[i, j])
                    status = "exogenous" if is_exo else "endogenous"
                    row = {
                        "variable_name": var_name,
                        "row_label": row_label,
                        "col_label": col_label,
                        "status": status,
                        year_cols[0]: var.calibration_value[i, j],
                    }
                    for y in year_cols[1:]:
                        row[y] = np.nan
                    rows.append(row)

    cols = ["variable_name", "row_label", "col_label", "status"] + year_cols
    return pd.DataFrame(rows, columns=cols)





def fill_timeseries(timeseries_df_template, growth_ratios_df, year_cols, VARIABLES_SPECS):
    """Fill a time-series template for exogenous rows using growth factors.

    The function applies three passes:
    1) For variables with ``is_t_minus_one`` set, copy year[0] into year[1].
    2) For rows explicitly listed in ``growth_ratios_df``, apply growth factors.
    3) For remaining exogenous rows not listed, hold values constant over years.

    Endogenous rows are never modified.

    Args:
        timeseries_df_template: DataFrame produced by ``build_timeseries_df``.
        growth_ratios_df: DataFrame with growth factors by variable/index.
        year_cols: Ordered list of year column names as strings.
        VARIABLES_SPECS: Dictionary of Variable objects.

    Returns:
        The filled DataFrame.
    """
    df = timeseries_df_template.copy()

    # Identify variables that use t-1 linkage (bool True or source variable name).
    t_minus_one_vars = {var.name for var in VARIABLES_SPECS.values() if var.is_t_minus_one}

    # Pass 0: for t-1 variables, set year[1] = year[0] and keep following years untouched.
    for var_name in t_minus_one_vars:
        mask = df["variable_name"] == var_name
        df.loc[mask, year_cols[1]] = df.loc[mask, year_cols[0]].values

    # Normalise index columns in growth_ratios_df: NaN → ""
    gr = growth_ratios_df.copy()
    gr["row_label"] = gr["row_label"].fillna("")
    gr["col_label"] = gr["col_label"].fillna("")

    # Pass 1: apply growth factors to exogenous rows explicitly present in growth_ratios_df.
    gr_keys = set()
    for _, gr_row in gr.iterrows():
        var_name = gr_row["variable_name"]
        row_label = gr_row["row_label"]
        col_label = gr_row["col_label"]
        gr_keys.add((var_name, row_label, col_label))

        if var_name in t_minus_one_vars:
            continue

        mask = (
            (df["variable_name"] == var_name) &
            (df["row_label"] == row_label) &
            (df["col_label"] == col_label)
        )

        if not mask.any():
            raise ValueError(
                f"Row ('{var_name}', '{row_label}', '{col_label}') from growth_ratios_df "
                f"not found in timeseries_df_template"
            )

        if "endogenous" in df.loc[mask, "status"].values:
            raise ValueError(
                f"Row ('{var_name}', '{row_label}', '{col_label}') is endogenous in "
                f"timeseries_df_template but appears in growth_ratios_df"
            )

        cal_val = df.loc[mask, year_cols[0]].values[0]

        for y in year_cols:
            df.loc[mask, y] = cal_val * gr_row[y]

    # Pass 2: exogenous rows absent from growth_ratios_df are held constant.
    for idx, row in df.iterrows():
        if row["status"] == "endogenous":
            continue
        if row["variable_name"] in t_minus_one_vars:
            continue
        key = (row["variable_name"], row["row_label"], row["col_label"])
        if key not in gr_keys:
            cal_val = row[year_cols[0]]
            for y in year_cols[1:]:
                df.at[idx, y] = cal_val
    return df



def build_and_fill_timeseries_df(VARIABLES_SPECS, growth_ratios_df, year_cols): 
    """Convenience function to build and fill the time-series DataFrame."""
    template = build_timeseries_df(VARIABLES_SPECS, year_cols)
    filled = fill_timeseries(template, growth_ratios_df, year_cols, VARIABLES_SPECS)
    return filled



def timeseries_df_to_dict(timeseries_df, year, status):
    """Convert one year slice of the DataFrame into a variable dictionary.

    Rows are first filtered by status. Then each variable name is mapped to:
    - a scalar if it appears once,
    - a 1D NumPy array if it appears multiple times.

    Args:
        timeseries_df: Time-series DataFrame.
        year: Year column to extract.
        status: Row status to filter (for example, ``endogenous`` or ``exogenous``).

    Returns:
        Dictionary mapping variable names to scalar or 1D array values.
    """
    filtered = timeseries_df[timeseries_df["status"] == status]
    dict = {}
    for var_name, group in filtered.groupby("variable_name", sort=False):
        values = group[year].values
        dict[var_name] = values[0] if len(values) == 1 else values

    return dict




def var_dict_to_timeseries_df(var_dict, timeseries_df, year, VARIABLES_SPECS, year_cols):
    """Write endogenous values from a dictionary into one year of the DataFrame.

    The function updates the selected year for endogenous rows using ``var_dict``.
    If the selected year is not the last year, it also propagates t-1 relationships:
    each lagged variable at ``year+1`` receives the value of its linked variable at
    ``year``.

    Args:
        var_dict: Dictionary in the format returned by ``timeseries_df_to_dict``.
        timeseries_df: Input time-series DataFrame.
        year: Year column to write.
        VARIABLES_SPECS: Dictionary of Variable objects.
        year_cols: Ordered list of year column names as strings.

    Returns:
        A copy of the DataFrame with updated values.
    """
    df = timeseries_df.copy()

    # Fill endogenous variables from var_dict
    for var_name, values in var_dict.items():
        mask = (df["variable_name"] == var_name) & (df["status"] == "endogenous")
        indices = df.index[mask]
        if np.isscalar(values):
            df.loc[indices, year] = values
        else:
            for idx, val in zip(indices, values):
                df.at[idx, year] = val

    # Fill lagged variables: var(year+1) gets linked_var(year).
    if year != year_cols[-1]:
        next_year = year_cols[year_cols.index(year) + 1]
        lagged_vars = {
            var.name: var.is_t_minus_one
            for var in VARIABLES_SPECS.values()
            if var.is_t_minus_one is not False
        }
        for lagged_var_name, previous_period_var_name in lagged_vars.items():
            previous_period_mask = df["variable_name"] == previous_period_var_name
            lagged_mask = df["variable_name"] == lagged_var_name
            df.loc[lagged_mask, next_year] = df.loc[previous_period_mask, year].values

    return df



timeseries_df_template = build_timeseries_df(VARIABLES_SPECS, year_cols)

timeseries_df = fill_timeseries(timeseries_df_template, growth_ratios_df, year_cols, VARIABLES_SPECS)

dict=timeseries_df_to_dict(timeseries_df, year_cols[0], "endogenous")

updated_df = var_dict_to_timeseries_df(dict, timeseries_df, year_cols[1], VARIABLES_SPECS, year_cols)
