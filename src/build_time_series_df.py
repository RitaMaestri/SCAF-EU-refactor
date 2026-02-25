import numpy as np 
import pandas as pd
from Variable_class import Variable
from Variables_specs import VARIABLES_SPECS



exogenous_data = "REMIND_exogenous_data_sectors"

growth_ratios_df = pd.read_csv(
    "data/"+exogenous_data+".csv") 

years = np.array([eval(i) for i in growth_ratios_df.columns[4:]])
year_cols = [str(y) for y in years]

def build_timeseries_df(VARIABLES_SPECS, year_cols):
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


def fill_timeseries(timeseries_df_template, growth_ratios_df, year_cols):
    df = timeseries_df_template

    # Normalise index columns in growth_ratios_df: NaN → ""
    gr = growth_ratios_df.copy()
    gr["Sector_1"] = gr["Sector_1"].fillna("")
    gr["Sector_2"] = gr["Sector_2"].fillna("")

    # Growth ratio year columns (skip the first, which is the base year = 1)
    gr_year_int = list(growth_ratios_df.columns[4:])

    # --- Pass 1: fill rows that appear in growth_ratios_df ---
    gr_keys = set()
    for _, gr_row in gr.iterrows():
        var_name = gr_row["variable"]
        row_label = gr_row["Sector_1"]
        col_label = gr_row["Sector_2"]
        gr_keys.add((var_name, row_label, col_label))

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
        for y_str, y_int in zip(year_cols[1:], gr_year_int[1:]):
            df.loc[mask, y_str] = cal_val * gr_row[y_int]

    # --- Pass 2: exogenous rows absent from growth_ratios_df → hold constant ---
    for idx, row in df.iterrows():
        if row["status"] == "endogenous":
            continue
        key = (row["variable_name"], row["row_label"], row["col_label"])
        if key not in gr_keys:
            cal_val = row[year_cols[0]]
            for y in year_cols[1:]:
                df.at[idx, y] = cal_val

timeseries_df_template = build_timeseries_df(VARIABLES_SPECS, year_cols)
fill_timeseries(timeseries_df_template, growth_ratios_df, year_cols)