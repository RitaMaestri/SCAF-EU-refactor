import pandas as pd
import os
import sys
from pathlib import Path

from lib import create_population_template,fill_population_template, filter_REMIND

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
	sys.path.append(str(SRC_ROOT))

from common.path_loader import load_config

module_dir = Path(__file__).resolve().parent
config = load_config(module_dir)

REMIND_path = os.path.join(config["raw_data_root"], config["remind_file"])
mapping_regions_path=config["mapping_regions"]
out_path = os.path.join(config["calibration_output_root"], config["out_file"])
population_raw_path = os.path.join(config["raw_data_root"], config["population_file"])


REMIND_raw = pd.read_csv(REMIND_path, header=0, sep=";")
REMIND = filter_REMIND(REMIND_raw)

mapping_regions_df = pd.read_csv(mapping_regions_path, header=0)

population_raw_df = pd.read_csv(population_raw_path, header=0)

template_df = create_population_template(REMIND)
filled_population_df = fill_population_template(template_df, population_raw_df, mapping_regions_df, ssp_column="pop_SSP2")
population_df_EU = filled_population_df[filled_population_df["Region"]=="EUR"]

population_df_EU.to_csv(out_path, index=False)