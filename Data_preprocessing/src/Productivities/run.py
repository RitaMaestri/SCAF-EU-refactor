import pandas as pd
import os
import sys
from pathlib import Path

from lib import create_sector_template,fill_sector_productivity, reformat_df, filter_REMIND

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
	sys.path.append(str(SRC_ROOT))

from common.path_loader import load_config

module_dir = Path(__file__).resolve().parent
config = load_config(module_dir)

productivities_df_path = os.path.join(config["raw_data_root"], config["productivities_file"])
mapping_regions_path=config["mapping_regions"]
sectors_mapping_path=config["sectors_prod_mapping"]
out_path = os.path.join(config["calibration_output_root"], config["out_file"])
REMIND_path = os.path.join(config["raw_data_root"], config["remind_file"])

productivities_df = pd.read_csv(productivities_df_path, header=0)
mapping_regions_df = pd.read_csv(mapping_regions_path, header=0)
sectors_mapping_df = pd.read_csv(sectors_mapping_path, header=0)

REMIND = pd.read_csv(REMIND_path, header=0, sep=";")
filtered_REMIND = filter_REMIND(REMIND)

template_df=create_sector_template(filtered_REMIND,sectors_mapping_df)


productivities_out_df = fill_sector_productivity(template_df, productivities_df, sectors_mapping_df, mapping_regions_df)

#productivities_out_df = reformat_df(productivities_out_df)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

productivities_out_df.to_csv(out_path, index=False)
