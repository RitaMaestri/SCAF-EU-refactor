from __future__ import annotations

import pandas as pd
from pathlib import Path


def reformat_elasticities(
    src_path: Path,
    region: str = "EUR",
    sector_order: list[str] | None = None,
) -> pd.DataFrame:
    """Extract one region row from an aggregated elasticities CSV and reshape
    it to the ``commodity,elasticity`` format expected by the Solver.

    If *sector_order* is provided the rows are returned in that order.
    """
    df = pd.read_csv(src_path, index_col=0)
    row = df.loc[region]
    if sector_order is not None:
        row = row.reindex(sector_order)
    return row.rename_axis("commodity").reset_index(name="elasticity")


def validate_template_mapping_consistency(
    template_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> None:
    """Raise ValueError if template and mapping reference different variable sets."""
    template_vars = set(template_df["variable_name"].unique())
    mapping_vars = set(mapping_df["variable_solver"].unique())
    errors = []
    missing_in_mapping = template_vars - mapping_vars
    if missing_in_mapping:
        errors.append(f"  Variables in template with no mapping: {sorted(missing_in_mapping)}")
    missing_in_template = mapping_vars - template_vars
    if missing_in_template:
        errors.append(f"  Variables in mapping not present in template: {sorted(missing_in_template)}")
    if errors:
        raise ValueError(
            "Inconsistencies between growth_factors_template and mapping_file:\n" + "\n".join(errors)
        )


def collect_year_columns(
    calibration_root: Path,
    mapping_df: pd.DataFrame,
    end_year: int | None = None,
) -> list[str]:
    """Return sorted list of year column names (as strings) present in the
    source files referenced by the mapping, optionally capped at end_year."""
    year_cols: set[str] = set()
    for file_name in mapping_df["file_name"].unique():
        path = calibration_root / file_name
        if not path.exists():
            print(f"WARNING: source file not found, skipping for year discovery: {path}")
            continue
        df = pd.read_csv(path, nrows=0)  # header only
        for col in df.columns:
            if str(col).isdigit():
                if end_year is None or int(col) <= end_year:
                    year_cols.add(str(col))
    return sorted(year_cols, key=int)


def build_growth_factors(
    template_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    calibration_root: Path,
    end_year: int | None = None,
) -> pd.DataFrame:
    """For each row in template_df, look up the corresponding time series via
    mapping_df, normalise to the first year (growth factor), and return the
    assembled DataFrame."""
    year_cols = collect_year_columns(calibration_root, mapping_df, end_year)
    if not year_cols:
        raise ValueError("No year columns found in source files.")

    # Cache loaded source DataFrames to avoid redundant I/O
    cache: dict[str, pd.DataFrame] = {}

    rows = []
    for _, tmpl_row in template_df.iterrows():
        var_name = tmpl_row["variable_name"]
        row_label = tmpl_row["row_label"] if pd.notna(tmpl_row["row_label"]) else ""
        col_label = tmpl_row["col_label"] if pd.notna(tmpl_row["col_label"]) else ""

        # --- look up mapping ---
        match = mapping_df[mapping_df["variable_solver"] == var_name]
        if match.empty:
            print(f"WARNING: '{var_name}' not found in mapping — skipping.")
            continue
        m = match.iloc[0]
        file_name = m["file_name"]
        var_preproc = m["variable_preprocessing"]
        row_label_col = m["row_label"] if pd.notna(m["row_label"]) and str(m["row_label"]).strip() else ""
        col_label_col = m["col_label"] if pd.notna(m["col_label"]) and str(m["col_label"]).strip() else ""

        # --- load source file (cached) ---
        if file_name not in cache:
            path = calibration_root / file_name
            if not path.exists():
                print(f"WARNING: source file not found: {path} — skipping '{var_name}'.")
                cache[file_name] = None
            else:
                cache[file_name] = pd.read_csv(path)
        src = cache[file_name]
        if src is None:
            continue

        # --- filter to the right row ---
        filtered = src[src["Variable"] == var_preproc].copy()
        if "Region" in filtered.columns:
            filtered = filtered[filtered["Region"] == "EUR"]
        if row_label_col and row_label:
            filtered = filtered[filtered[row_label_col] == row_label]
        if col_label_col and col_label:
            filtered = filtered[filtered[col_label_col] == col_label]

        if filtered.empty:
            print(
                f"WARNING: no match for ({var_name}, row='{row_label}', col='{col_label}') "
                f"in {file_name} — filling with empty values."
            )
            rows.append({"variable_name": var_name, "row_label": row_label, "col_label": col_label,
                         **{yr: "" for yr in year_cols}})
            continue
        if len(filtered) > 1:
            print(
                f"WARNING: multiple matches for ({var_name}, row='{row_label}', col='{col_label}') "
                f"in {file_name} — using first row."
            )
        src_row = filtered.iloc[0]

        # --- extract year values ---
        values = {}
        for yr in year_cols:
            values[yr] = src_row[yr] if yr in src_row.index else float("nan")

        # --- normalise to base year (growth factor) ---
        base_year = year_cols[0]
        base_value = values[base_year]
        if pd.isna(base_value) or base_value == 0:
            print(
                f"WARNING: base-year value is {'NaN' if pd.isna(base_value) else '0'} "
                f"for ({var_name}, row='{row_label}', col='{col_label}') — skipping."
            )
            continue
        normalised = {yr: v / base_value for yr, v in values.items()}

        rows.append({"variable_name": var_name, "row_label": row_label, "col_label": col_label, **normalised})

    return pd.DataFrame(rows, columns=["variable_name", "row_label", "col_label"] + year_cols)


def build_hybridization(calibration_root: Path) -> pd.DataFrame:
    """Load hybridization_df and append energy_trade_projection rows.

    Schema alignment:
    - ``Sector`` in energy_trade_projection → ``Energy consumers``
    - ``Energy uses`` (absent in energy_trade_projection) is filled with empty strings.
    """
    hybridization_df = pd.read_csv(calibration_root / "Hybridization/hybridization_df.csv")
    energy_trade_df = pd.read_csv(calibration_root / "Energy_trade/energy_trade_projection.csv")
    energy_trade_df = energy_trade_df.rename(columns={"Sector": "Energy consumers"})
    energy_trade_df["Energy uses"] = ""
    energy_trade_df = energy_trade_df.reindex(columns=hybridization_df.columns)
    return pd.concat([hybridization_df, energy_trade_df], ignore_index=True)


def fill_row_with_ones(
    df: pd.DataFrame,
    variable_name: str,
    row_label: str,
    col_label: str,
) -> pd.DataFrame:
    """Fill the year columns of the row identified by the 3 labels with ones."""
    year_cols = [c for c in df.columns if str(c).isdigit()]
    mask = (
        (df["variable_name"] == variable_name) &
        (df["row_label"] == row_label) &
        (df["col_label"] == col_label)
    )
    df.loc[mask, year_cols] = 1
    return df
