import pandas as pd
import os
from utils.aggregate import aggregate_sectors,remove_unnamed_labels
import json
from pathlib import Path

# useful paths
this_folder = os.path.dirname(__file__)
config = json.load(open(this_folder+"/config.json"))



IOTs_folder=Path(config["IOTs_path"])
out_path=config["out_path"]

energy_types=["BIOMASS","COAL","OTHER ENERGY","OIL","GAS"]
energy_sector="ENERGY"


for IOT_file in IOTs_folder.glob("*.csv"):
    IOT = pd.read_csv(
    IOT_file,
    header=[0, 1],
    index_col=[0, 1],     # ← evita che stringhe vuote vengano trattate come NaN
)
    IOT=remove_unnamed_labels(IOT)
    aggregated_IOT=aggregate_sectors(IOT,energy_types,energy_sector)
    aggregated_IOT.to_csv(out_path + "/" + IOT_file.name, na_rep='', encoding='utf-8-sig')
print("Aggregated tables available at" + out_path)
