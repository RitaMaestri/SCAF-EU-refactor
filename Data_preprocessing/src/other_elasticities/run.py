import os
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.path_loader import load_config
from lib import aggregate_by_region, aggregate_by_sector, load_kl_weights

module_dir = Path(__file__).resolve().parent
config = load_config(module_dir)

elasticities_path = os.path.join(config["raw_data_root"], config["elasticities_file"])
armington_weights_path = os.path.join(config["raw_data_root"], config["armington_weights_file"])
kl_weights_folder = os.path.join(config["raw_data_root"], config["kl_weights_folder"])
mapping_sectors_path = config["mapping_sectors"]
mapping_regions_path = config["mapping_eur_regions"]
out_armington = os.path.join(config["calibration_output_root"], config["out_armington_elasticities"])
out_kl = os.path.join(config["calibration_output_root"], config["out_kl_elasticities"])
out_armington_w = os.path.join(config["calibration_output_root"], config["out_armington_weights"])
out_kl_w = os.path.join(config["calibration_output_root"], config["out_kl_weights"])

elasticities = pd.read_csv(elasticities_path, sep=",", index_col=0)
armington_weights = pd.read_csv(armington_weights_path, sep="|", index_col=0)
kl_weights = load_kl_weights(kl_weights_folder)
s_map = pd.read_csv(mapping_sectors_path, sep=",")
r_map = pd.read_csv(mapping_regions_path, sep=",")

# Armington elasticities
armington_el = elasticities["Armington_elasticity"]
r_agg_arm_el, r_agg_arm_w = aggregate_by_region(armington_el, armington_weights, r_map)
sr_agg_arm_el, sr_agg_arm_w = aggregate_by_sector(r_agg_arm_el, r_agg_arm_w, s_map)

# KL elasticities
kl_el = elasticities["KL_elasticity"]
r_agg_kl_el, r_agg_kl_w = aggregate_by_region(kl_el, kl_weights, r_map)
sr_agg_kl_el, sr_agg_kl_w = aggregate_by_sector(r_agg_kl_el, r_agg_kl_w, s_map)

Path(out_armington).parent.mkdir(parents=True, exist_ok=True)
sr_agg_arm_el.to_csv(out_armington)
sr_agg_kl_el.to_csv(out_kl)
sr_agg_arm_w.to_csv(out_armington_w)
sr_agg_kl_w.to_csv(out_kl_w)
