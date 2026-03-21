from reformat_exiobase.reformat_EXIOBASE import reformat_EXIOBASE
from reformat_exiobase.aggregate_EXIOBASE import aggregate_EXIOBASE
from reformat_exiobase.download_EXIOBASE import download_EXIOBASE
from pathlib import Path
import sys
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.path_loader import load_config

current_file = Path(__file__).resolve()
current_path = current_file.parent



# results path
output_path = current_path / "output"
download_path = output_path / "download"
aggregation_path = output_path / "aggregation" / "one_energy"
# Set reformat_path relative to repo root
shared_config = load_config(current_path)
reformat_path = Path(shared_config["calibration_output_root"]) / "regional_IOTs"

#create paths if not present
for folder in [ output_path, download_path, aggregation_path, reformat_path]:
    folder.mkdir(parents=True, exist_ok=True)

# mappings path
sec_map_file = current_path / "mappings" / "map_sectors.csv"
reg_map_file = current_path / "mappings" / "map_regions.csv"

#download configuration
year = "2020"
#product_per_product or industry_per_industry
p_or_i = "pxp"

version = "10.5281/zenodo.3583070" # version released in february 2025

sectors_order= ["AGRICULTURE","MANUFACTURE","SERVICES","STEEL","CHEMICAL","ENERGY","TRANSPORTATION"]


#download_EXIOBASE(str(download_path),system=p_or_i, years=year,version=version)

#aggregate_EXIOBASE(reg_map_path=str(reg_map_file), sec_map_path=str(sec_map_file), output_path=str(aggregation_path), input_path=str(download_path), year=year, system=p_or_i)

reformat_EXIOBASE(aggregation_folder=str(aggregation_path), reformat_folder=str(reformat_path),energy_sectors=["ENERGY"],sectors_order=sectors_order, add_inventories=False)

pd.DataFrame({"sector": sectors_order}).to_csv(reformat_path / "sectors.csv", index=False)