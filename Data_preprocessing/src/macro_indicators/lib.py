import pandas as pd


MACRO_VARIABLES = ["GDP|PPP", "Investments", "Capital Stock|Non-ESM"]


def filter_REMIND(df: pd.DataFrame, calibration_year: int) -> pd.DataFrame:
    """Extract EUR macro-indicator rows from a raw REMIND DataFrame.

    Keeps only rows where:
    - Region == "EUR"
    - Variable is one of MACRO_VARIABLES

    Year columns before *calibration_year* are dropped.
    The trailing NaN column produced by the trailing ';' in .mif files is also removed.
    """
    # Drop trailing NaN-named column produced by the trailing ';' in .mif files
    df = df.loc[:, df.columns.notna()]

    # Filter region and variables of interest
    mask = (df["Region"] == "EUR") & (df["Variable"].isin(MACRO_VARIABLES))
    df = df[mask].copy()

    # Drop year columns before calibration_year
    year_cols_to_drop = [
        c for c in df.columns if str(c).isdigit() and int(c) < calibration_year
    ]
    df = df.drop(columns=year_cols_to_drop)

    return df
