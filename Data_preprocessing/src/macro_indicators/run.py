import pandas as pd
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.path_loader import load_config
from lib import filter_REMIND

module_dir = Path(__file__).resolve().parent
repo_root = module_dir.parents[2]
config = load_config(module_dir)

calibration_year = int(config["calibration_year"])
remind_path = repo_root / config["raw_data_root"] / config["remind_file"]
out_dir = repo_root / config["out_dir"]

# --- Load and filter REMIND ---
REMIND_raw = pd.read_csv(remind_path, sep=";")
REMIND_filtered = filter_REMIND(REMIND_raw, calibration_year)

year_cols = [c for c in REMIND_filtered.columns if str(c).isdigit()]

gdp_row = REMIND_filtered[REMIND_filtered["Variable"] == "GDP|PPP"]
inv_row = REMIND_filtered[REMIND_filtered["Variable"] == "Investments"]
kn_row  = REMIND_filtered[REMIND_filtered["Variable"] == "Capital Stock|Non-ESM"]

# --- Compute Investment share = Investments / GDP|PPP (year by year) ---
gdp_values = gdp_row[year_cols].iloc[0]
inv_values = inv_row[year_cols].iloc[0]
inv_share_values = (inv_values / gdp_values).to_dict()

ref = gdp_row.iloc[0]
inv_share_row = {
    "Model":    ref["Model"],
    "Scenario": ref["Scenario"],
    "Region":   ref["Region"],
    "Variable": "Investment share",
    "Unit":     "-",
    **inv_share_values,
}

# --- Assemble output: GDP|PPP, Capital Stock|Non-ESM, Investment share ---
output_df = pd.concat(
    [gdp_row, kn_row, pd.DataFrame([inv_share_row])],
    ignore_index=True,
)

# --- Save ---
out_dir.mkdir(parents=True, exist_ok=True)
output_df.to_csv(out_dir / "macro_indicators.csv", index=False)

