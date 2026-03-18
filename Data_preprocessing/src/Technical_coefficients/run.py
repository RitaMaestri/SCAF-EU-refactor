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
)

this_folder = Path(__file__).resolve().parent
repo_root = this_folder.parents[2]
config = load_config(this_folder)

energy_domestic_volumes_path = repo_root / config["energy_domestic_volumes"]
energy_trade_volumes_path    = repo_root / config["energy_trade_volumes"]
remind_path                  = repo_root / config["raw_data_root"] / config["remind_file"]
mapping_path                 = repo_root / config["mapping"]

# --- Load data ---
energy_domestic_df = pd.read_csv(energy_domestic_volumes_path)
energy_trade_df    = pd.read_csv(energy_trade_volumes_path)
remind_df          = load_remind(remind_path)
mapping_df         = pd.read_csv(mapping_path)

# --- Total EUR energy supply (sum over all energy_use rows) ---
total_energy_supply_ts = compute_total_energy_supply(energy_domestic_df, region="EUR")

# --- Import / export volume time series for EUR ---
energy_imports_ts = extract_trade_volume(energy_trade_df, region="EUR", variable="Import|Energy")
energy_exports_ts = extract_trade_volume(energy_trade_df, region="EUR", variable="Export|Energy")

# --- Domestic output: produced energy = supply - imports + exports ---
energy_domestic_output = total_energy_supply_ts - energy_imports_ts + energy_exports_ts

# --- Technical coefficients ---
technical_coefficients_df = compute_technical_coefficients(
    mapping_df=mapping_df,
    remind_df=remind_df,
    domestic_df=energy_domestic_df,
)

