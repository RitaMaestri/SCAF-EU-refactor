import pandas as pd


def _year_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if str(c).isdigit()]


def compute_total_energy_supply(df: pd.DataFrame, region: str) -> pd.Series:
    year_cols = _year_columns(df)
    mask = df["Region"] == region
    return df.loc[mask, year_cols].sum()


def extract_trade_volume(df: pd.DataFrame, region: str, variable: str) -> pd.Series:
    year_cols = _year_columns(df)
    mask = (df["Region"] == region) & (df["Variable"] == variable)
    rows = df.loc[mask, year_cols]
    if rows.empty:
        raise ValueError(f"No row found for Region={region!r}, Variable={variable!r}")
    return rows.iloc[0]


def load_remind(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    # Drop trailing empty column produced by trailing semicolon in header
    return df.loc[:, df.columns.notna()]


def create_technical_coefficients_template(
    mapping_df: pd.DataFrame,
    remind_df: pd.DataFrame,
    year_cols: list,
    sectors_df: pd.DataFrame,
    region: str = "EUR",
) -> pd.DataFrame:
    eur_remind = remind_df[remind_df["Region"] == region]
    model = eur_remind["Model"].iloc[0]
    scenario = eur_remind["Scenario"].iloc[0]

    energy_consumers = sectors_df["SCAF sector"].unique().tolist()

    rows = []
    for _, map_row in mapping_df.iterrows():
        energy_use = map_row["energy_use"]
        unit = map_row["unit"]

        unit_str = "" if pd.isna(unit) else str(unit).strip()
        unit_value = f"EJ/{unit_str}" if unit_str else "EJ"

        for energy_consumer in energy_consumers:
            meta = {
                "Model": model,
                "Scenario": scenario,
                "Region": region,
                "Variable": "Technical coefficients",
                "Energy consumers": energy_consumer,
                "Energy uses": energy_use,
                "Unit": unit_value,
            }
            year_values = {y: float("nan") for y in year_cols}
            rows.append({**meta, **year_values})

    return pd.DataFrame(rows)


def fill_technical_coefficients(
    template_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    remind_df: pd.DataFrame,
    energy_volumes_df: pd.DataFrame,
    year_cols: list,
    region: str = "EUR",
) -> pd.DataFrame:
    result = template_df.copy()

    for _, map_row in mapping_df.iterrows():
        energy_use = map_row["energy_use"]
        remind_output = map_row["output_volume_REMIND"]

        if pd.isna(remind_output) or str(remind_output).strip() == "":
            continue

        remind_mask = (
            (remind_df["Region"] == region)
            & (remind_df["Variable"] == remind_output)
        )
        remind_rows = remind_df.loc[remind_mask, year_cols]
        if remind_rows.empty:
            raise ValueError(
                f"No REMIND row for Region={region!r}, Variable={remind_output!r}"
            )
        denominator_ts = remind_rows.iloc[0]

        volumes_mask = (
            (energy_volumes_df["Region"] == region)
            & (energy_volumes_df["energy_use"] == energy_use)
        )
        volumes_rows = energy_volumes_df.loc[volumes_mask, year_cols]
        if volumes_rows.empty:
            raise ValueError(
                f"No energy volume row for Region={region!r}, energy_use={energy_use!r}"
            )
        volume_ts = volumes_rows.iloc[0]

        row_mask = result["Energy uses"] == energy_use
        result.loc[row_mask, year_cols] = (volume_ts / denominator_ts).values

    return result


def compute_ind_technical_coefficients(
    ind_mapping_df: pd.DataFrame,
    remind_df: pd.DataFrame,
    year_cols: list,
    region: str = "EUR",
) -> tuple:
    """Compute IND technical coefficients as IND_energy_consumption / IND_output.

    Returns (coeff_ts, unit_str) where unit_str follows the EJ/{output_unit} convention.
    """
    output_var = ind_mapping_df["IND_output"].iloc[0]
    consumption_var = ind_mapping_df["IND_energy_consumption"].iloc[0]

    unit = ind_mapping_df["output_unit"].iloc[0]
    unit_str = "" if pd.isna(unit) else str(unit).strip()
    unit_value = f"EJ/{unit_str}" if unit_str else "EJ"

    output_ts = extract_trade_volume(remind_df, region=region, variable=output_var)
    consumption_ts = extract_trade_volume(remind_df, region=region, variable=consumption_var)

    coeff_ts = (consumption_ts / output_ts).reindex(year_cols)
    return coeff_ts, unit_value


def build_remind_activity_outputs(
    mapping_df: pd.DataFrame,
    ind_mapping_df: pd.DataFrame,
    remind_df: pd.DataFrame,
    year_cols: list,
    energy_domestic_output: pd.Series,
    region: str = "EUR",
) -> pd.DataFrame:
    """Build one row per energy_use with the corresponding REMIND output variable
    time series (denominator of technical coefficients) in REMIND-like format.

    Columns: Model, Scenario, Region, Variable, energy_use, Unit, [year_cols]
    """
    eur_remind = remind_df[remind_df["Region"] == region]
    reference_row = eur_remind.iloc[0]

    ind_output_var = ind_mapping_df["IND_output"].iloc[0].strip()

    rows = []
    for _, map_row in mapping_df.iterrows():
        energy_use = map_row["energy_use"]
        out_var = map_row["output_volume_REMIND"]

        if energy_use == "PE":
            row = {
                "Model": reference_row["Model"],
                "Scenario": reference_row["Scenario"],
                "Region": region,
                "Variable": "Final Energy|Output",
                "energy_use": energy_use,
                "Unit": "EJ",
            }
            for y in year_cols:
                row[y] = energy_domestic_output[y] if y in energy_domestic_output.index else float("nan")
            rows.append(row)
            continue

        if energy_use == "IND":
            variable = ind_output_var
        elif pd.isna(out_var) or str(out_var).strip() == "":
            continue
        else:
            variable = str(out_var).strip()

        remind_mask = (eur_remind["Variable"] == variable)
        remind_rows = eur_remind.loc[remind_mask]
        if remind_rows.empty:
            raise ValueError(
                f"No REMIND row for Region={region!r}, Variable={variable!r}"
            )
        remind_row = remind_rows.iloc[0]

        row = {
            "Model": remind_row["Model"],
            "Scenario": remind_row["Scenario"],
            "Region": remind_row["Region"],
            "Variable": variable,
            "energy_use": energy_use,
            "Unit": remind_row["Unit"],
        }
        for y in year_cols:
            row[y] = remind_row[y] if y in remind_row.index else float("nan")
        rows.append(row)

    return pd.DataFrame(rows, columns=["Model", "Scenario", "Region", "Variable", "energy_use", "Unit"] + year_cols)


def compute_technical_coefficients(
    mapping_df: pd.DataFrame,
    remind_df: pd.DataFrame,
    energy_volumes_df: pd.DataFrame,
    year_cols: list,
    sectors_df: pd.DataFrame,
    region: str = "EUR",
) -> pd.DataFrame:
    template = create_technical_coefficients_template(
        mapping_df=mapping_df,
        remind_df=remind_df,
        year_cols=year_cols,
        sectors_df=sectors_df,
        region=region,
    )
    return fill_technical_coefficients(
        template_df=template,
        mapping_df=mapping_df,
        remind_df=remind_df,
        energy_volumes_df=energy_volumes_df,
        year_cols=year_cols,
        region=region,
    )
