import shutil
import pandas as pd
import sys
from pathlib import Path

from lib import build_growth_factors, fill_row_with_ones, build_hybridization, reformat_elasticities

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.path_loader import load_config

module_dir = Path(__file__).resolve().parent
repo_root = module_dir.parents[2]
config = load_config(module_dir)

template_path = repo_root / config["growth_factors_template"]
mapping_path = repo_root / config["mapping_file"]
calibration_root = repo_root / config["calibration_output_root"]
preprocessed_data_root = repo_root / config["preprocessed_data_root"]
out_path = repo_root / config["out_file"]
end_year = int(config["end_year"]) if config.get("end_year") else None

template_df = pd.read_csv(template_path)
mapping_df = pd.read_csv(mapping_path)

growth_factors_df = build_growth_factors(template_df, mapping_df, calibration_root, end_year)
growth_factors_df = fill_row_with_ones(growth_factors_df, "pXj", "SERVICES", "")

out_path.parent.mkdir(parents=True, exist_ok=True)
growth_factors_df.to_csv(out_path, index=False)
print(f"Written {len(growth_factors_df)} rows to {out_path}")

# Copy ready-to-use files into preprocessed_data
copies = [
    (calibration_root / "SSP2/population.csv",             preprocessed_data_root / "calibration/population.csv"),
    (calibration_root / "regional_IOTs/EUR.csv",           preprocessed_data_root / "regional_IOTs/EUR.csv"),
    (calibration_root / "regional_IOTs/sectors/sectors.csv",           preprocessed_data_root / "indexes/sectors.csv"),
    (calibration_root / "Hybridization/energy_uses.csv",           preprocessed_data_root / "indexes/energy_uses.csv"),
    (calibration_root / "Hybridization/energy_consumers.csv",           preprocessed_data_root / "indexes/energy_consumers.csv"),
]
for src, destination in copies:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, destination)
    print(f"Copied {src.relative_to(repo_root)} → {destination.relative_to(repo_root)}")

# Reformat and copy elasticities from consumption_elasticities module
elasticities_src = calibration_root / "consumption_elasticities"
sector_order = pd.read_csv(preprocessed_data_root / "indexes/sectors.csv")["sector"].tolist()
for src_name, dst_name in [
    ("aggregated_income_elasticities.csv",     "elasticities/income_elasticities.csv"),
    ("aggregated_own_price_elasticities.csv",  "elasticities/own_price_elasticities.csv"),
    ("compensated_own_price_elasticities.csv", "elasticities/compensated_own_price_elasticities.csv"),
]:
    df = reformat_elasticities(elasticities_src / src_name, sector_order=sector_order)
    destination = preprocessed_data_root / dst_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    print(f"Written {src_name} → {destination.relative_to(repo_root)}")

# Reformat and copy elasticities from other_elasticities module
other_elasticities_src = calibration_root / "other_elasticities"
for src_name, dst_name in [
    ("aggregated_armington_elasticities.csv", "elasticities/armington_elasticities.csv"),
    ("aggregated_kl_elasticities.csv",        "elasticities/kl_elasticities.csv"),
]:
    df = reformat_elasticities(other_elasticities_src / src_name, sector_order=sector_order)
    destination = preprocessed_data_root / dst_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    print(f"Written {src_name} → {destination.relative_to(repo_root)}")

# Build hybridization_df by appending energy_trade_projection rows
combined_hybridization_df = build_hybridization(calibration_root)
destination = preprocessed_data_root / "calibration/hybridization_df.csv"
destination.parent.mkdir(parents=True, exist_ok=True)
combined_hybridization_df.to_csv(destination, index=False)
print(f"Written {len(combined_hybridization_df)} rows to {destination.relative_to(repo_root)}")
