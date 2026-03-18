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


def compute_technical_coefficients(
    mapping_df: pd.DataFrame,
    remind_df: pd.DataFrame,
    domestic_df: pd.DataFrame,
    region: str = "EUR",
) -> pd.DataFrame:
    year_cols = _year_columns(domestic_df)

    eur_remind = remind_df[remind_df["Region"] == region]
    model = eur_remind["Model"].iloc[0]
    scenario = eur_remind["Scenario"].iloc[0]

    rows = []
    for _, map_row in mapping_df.iterrows():
        energy_use = map_row["energy_use"]
        remind_variable = map_row["output_volume_REMIND"]
        unit = map_row["unit"]

        unit_str = "" if pd.isna(unit) else str(unit).strip()
        meta = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": energy_use,
            "Unit": f"EJ/{unit_str}" if unit_str else "EJ",
        }

        if pd.isna(remind_variable) or str(remind_variable).strip() == "":
            year_values = {y: float("nan") for y in year_cols}
        else:
            remind_mask = (
                (remind_df["Region"] == region)
                & (remind_df["Variable"] == remind_variable)
            )
            remind_rows = remind_df.loc[remind_mask, year_cols]
            if remind_rows.empty:
                raise ValueError(
                    f"No REMIND row for Region={region!r}, Variable={remind_variable!r}"
                )
            numerator_ts = remind_rows.iloc[0]

            domestic_mask = (
                (domestic_df["Region"] == region)
                & (domestic_df["energy_use"] == energy_use)
            )
            domestic_rows = domestic_df.loc[domestic_mask, year_cols]
            if domestic_rows.empty:
                raise ValueError(
                    f"No domestic row for Region={region!r}, energy_use={energy_use!r}"
                )
            domestic_ts = domestic_rows.iloc[0]

            year_values = (numerator_ts / domestic_ts).to_dict()

        rows.append({**meta, **year_values})

    return pd.DataFrame(rows)
