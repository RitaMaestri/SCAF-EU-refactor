import pandas as pd
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.path_loader import load_config

module_dir = Path(__file__).resolve().parent.parent
config = load_config(module_dir)


IEA_input_path = Path(config["raw_data_root"]) / config["iea_input_file"]
output_folder = Path(config["IEA_output_folder"])
output_filename = config["IEA_output_filename"]
mapping_folder = Path(config["IEA_mapping_folder"])

# Create folders if they don't exist
output_folder.mkdir(parents=True, exist_ok=True)

# --- Read mapping CSV ---
mapping_csv_path = mapping_folder / "agglomerations_mapping.csv"
mapping_df = pd.read_csv(mapping_csv_path)

# --- Load IEA data ---
IEAdata = pd.read_csv(IEA_input_path)
IEAdata["agglomeration"] = "not assigned"

# --- Assign agglomeration manually based on mapping ---
# mapping_df should have columns: ISO, agglomeration
for aggl in mapping_df["agglomeration"].unique():
    isos = mapping_df.loc[mapping_df["agglomeration"] == aggl, "ISO"].tolist()
    selection = IEAdata.loc[IEAdata["ISO"].isin(isos), "agglomeration"]
    IEAdata.loc[IEAdata["ISO"].isin(isos), "agglomeration"] = [aggl] * len(selection)

# --- Group by agglomeration and sum numeric columns ---
numeric_cols = IEAdata.select_dtypes(include="number").columns
result = IEAdata.groupby("agglomeration")[numeric_cols].sum()


# --- Insert "Unit" column before numeric values ---
result.insert(0, "Unit", "EJ/y")

# --- Scale numeric columns by 1e6 ---
result[numeric_cols] = result[numeric_cols] / 1e6


# --- Save final CSV ---
output_csv_path = output_folder / output_filename
result.to_csv(output_csv_path, index=True)
print("Reformatted IEA data saved to:", output_csv_path)