import sys
import pandas as pd
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.path_loader import load_config
from lib import (
    compute_total_energy_supply,
    extract_trade_volume,
    load_remind,
    compute_technical_coefficients,
    compute_ind_technical_coefficients,
)

this_folder = Path(__file__).resolve().parent
repo_root = this_folder.parents[2]
config = load_config(this_folder)

energy_domestic_volumes_path = repo_root / config["remind_volumes_by_energy_use"]
energy_trade_volumes_path    = repo_root / config["energy_trade_volumes"]
remind_path                  = repo_root / config["raw_data_root"] / config["remind_file"]
mapping_path                 = repo_root / config["mapping"]
ind_mapping_path             = this_folder / "mappings" / "technical_coefficients_IND.csv"
sectors_mapping_path         = repo_root / config["map_sectors"]
out_path                     = repo_root / config["calibration_output_root"] / config["out_path"]

# --- Load data ---
energy_domestic_df = pd.read_csv(energy_domestic_volumes_path)
energy_trade_df    = pd.read_csv(energy_trade_volumes_path)
remind_df          = load_remind(remind_path)
mapping_df         = pd.read_csv(mapping_path)
ind_mapping_df     = pd.read_csv(ind_mapping_path, skipinitialspace=True)
sectors_df         = pd.read_csv(sectors_mapping_path)

################################################
### Compute standard technical coefficients ####
################################################
year_cols = [c for c in energy_domestic_df.columns if str(c).isdigit()]

# --- Technical coefficients ---
technical_coefficients_df = compute_technical_coefficients(
    mapping_df=mapping_df,
    remind_df=remind_df,
    energy_volumes_df=energy_domestic_df,
    year_cols= year_cols,
    sectors_df=sectors_df,
)
technical_coefficients_df


##############################################
### Compute Energy Technical Coefficients ####
##############################################

# --- Total EUR energy supply (sum over all energy_use rows) ---
total_energy_supply_ts = compute_total_energy_supply(energy_domestic_df, region="EUR")

# --- Import / export volume time series for EUR ---
energy_imports_ts = extract_trade_volume(energy_trade_df, region="EUR", variable="Import|Energy")
energy_exports_ts = extract_trade_volume(energy_trade_df, region="EUR", variable="Export|Energy")

# --- Domestic output: produced energy = supply - imports + exports ---
energy_domestic_output = total_energy_supply_ts - energy_imports_ts + energy_exports_ts

# --- Fill PE row: PE volumes (EUR) / energy_domestic_output ---
pe_volumes_ts = energy_domestic_df.loc[
    (energy_domestic_df["Region"] == "EUR") &
    (energy_domestic_df["energy_use"] == "PE"),
    year_cols,
].iloc[0]

pe_coeff_ts = pe_volumes_ts / energy_domestic_output

pe_mask = technical_coefficients_df["Energy uses"] == "PE"
n_pe = pe_mask.sum()
technical_coefficients_df.loc[pe_mask, year_cols] = [pe_coeff_ts.tolist()] * n_pe
technical_coefficients_df.loc[pe_mask, "Unit"] = "-"

##############################################
### Compute Industry Technical Coefficients ###
##############################################

ind_coeff_ts, ind_unit = compute_ind_technical_coefficients(
    ind_mapping_df=ind_mapping_df,
    remind_df=remind_df,
    year_cols=year_cols,
)

ind_mask = technical_coefficients_df["Energy uses"] == "IND"
n_ind = ind_mask.sum()
technical_coefficients_df.loc[ind_mask, year_cols] = [ind_coeff_ts.tolist()] * n_ind
technical_coefficients_df.loc[ind_mask, "Unit"] = ind_unit

out_path.parent.mkdir(parents=True, exist_ok=True)

technical_coefficients_df.to_csv(str(out_path), index=False)