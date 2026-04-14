import pandas as pd
import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.path_loader import load_config

module_dir = Path(__file__).resolve().parent
repo_root = module_dir.parents[2]
config = load_config(module_dir)

out_path = os.path.join(config["calibration_output_root"], config["out_folder"])

# --- Load energy trade values cache ---
trade_values_path = repo_root / config["energy_trade_values_cache"]
trade_df = pd.read_csv(trade_values_path)

year_cols = [c for c in trade_df.columns if str(c).isdigit()]

import_row = trade_df[trade_df["Variable"] == "aggregated_energy_import_values"].iloc[0]
export_row = trade_df[trade_df["Variable"] == "aggregated_energy_export_values"].iloc[0]

import_values_ts = import_row[year_cols].astype(float)
export_values_ts = export_row[year_cols].astype(float)

# Extract Model/Scenario from cache for output metadata
model = import_row["Model"]
scenario = import_row["Scenario"]

# --- Load REMIND values by energy use, sum EUR non-PE rows ---
remind_values_path = repo_root / config["remind_values_by_energy_use"]
remind_values_df = pd.read_csv(remind_values_path)

eur_non_pe_mask = (
    (remind_values_df["Region"] == "EUR") &
    (remind_values_df["energy_use"] != "PE")
)
remind_eur_non_pe_sum = (
    remind_values_df.loc[eur_non_pe_mask, year_cols]
    .astype(float)
    .sum(axis=0)
)  # pd.Series indexed by the same year_cols strings

remind_eur_all_sum = (
    remind_values_df.loc[remind_values_df["Region"] == "EUR", year_cols]
    .astype(float)
    .sum(axis=0)
)  # includes PE row

remind_eur_pe_ts = (
    remind_values_df.loc[
        (remind_values_df["Region"] == "EUR") & (remind_values_df["energy_use"] == "PE"),
        year_cols,
    ]
    .astype(float)
    .iloc[0]
)

# --- Compute KLM_expenditures ---
KLM_expenditures_ts = remind_eur_non_pe_sum + import_values_ts - export_values_ts

ratio_ts = (
    (remind_eur_non_pe_sum - export_values_ts + import_values_ts) /
    (remind_eur_all_sum - export_values_ts + import_values_ts)
)

# --- Write output ---
Path(out_path).mkdir(parents=True, exist_ok=True)

KLM_expenditures_df = pd.concat(
    [
        pd.DataFrame(
            {
                "Model": [model, model, model],
                "Scenario": [scenario, scenario, scenario],
                "Region": ["EUR", "EUR", "EUR"],
                "Variable": [
                    "KLM_expenditures",
                    "KLM_expenditures_over_output",
                    "primary_energy_expenditures",
                ],
                "Unit": ["bn US$2017", "dimensionless", "bn US$2017"],
            }
        ),
        pd.DataFrame(
            [
                KLM_expenditures_ts.to_numpy(),
                ratio_ts.to_numpy(),
                remind_eur_pe_ts.to_numpy(),
            ],
            columns=year_cols,
        ),
    ],
    axis=1,
)
KLM_expenditures_df.to_csv(os.path.join(out_path, "KLM_expenditures.csv"), index=False)
