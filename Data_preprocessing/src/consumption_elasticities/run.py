import os
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.path_loader import load_config
from lib import aggregate_by_region, aggregate_by_sector, compute_compensated_price_elasticities

module_dir = Path(__file__).resolve().parent
config = load_config(module_dir)

consumption_path = os.path.join(config["raw_data_root"], config["C_hsld_tot"])
income_el_path = os.path.join(config["raw_data_root"], config["income_elasticities_file"])
price_el_path = os.path.join(config["raw_data_root"], config["price_elasticities_file"])
mapping_sectors_path = config["mapping_sectors"]
mapping_regions_path = config["mapping_eur_regions"]
out_income = os.path.join(config["calibration_output_root"], config["out_income_elasticities"])
out_price = os.path.join(config["calibration_output_root"], config["out_price_elasticities"])
out_consumption = os.path.join(config["calibration_output_root"], config["out_consumptions"])
out_compensated = os.path.join(config["calibration_output_root"], config["out_compensated_price_elasticities"])

consumption = pd.read_csv(consumption_path, sep=",", index_col=0)
income_el = pd.read_csv(income_el_path, sep=",", index_col=0)
price_el = pd.read_csv(price_el_path, sep=",", index_col=0)
s_map = pd.read_csv(mapping_sectors_path, sep=",")
r_map = pd.read_csv(mapping_regions_path, sep=",")

r_agg_ie, r_agg_pe, r_agg_cons = aggregate_by_region(consumption, income_el, price_el, s_map, r_map)
sr_agg_ie, sr_agg_pe, sr_agg_cons = aggregate_by_sector(r_agg_ie, r_agg_pe, r_agg_cons, s_map)
sr_agg_compensated_pe = compute_compensated_price_elasticities(sr_agg_pe, sr_agg_ie, sr_agg_cons)

Path(out_income).parent.mkdir(parents=True, exist_ok=True)
sr_agg_ie.to_csv(out_income)
sr_agg_pe.to_csv(out_price)
sr_agg_cons.to_csv(out_consumption)
sr_agg_compensated_pe.to_csv(out_compensated)
