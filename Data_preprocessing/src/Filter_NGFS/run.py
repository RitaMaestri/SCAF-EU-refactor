import pandas as pd
import os
import json

from lib import filter_NGFS,compute_growth_rates

this_folder = os.path.dirname(__file__)
config = json.load(open(this_folder+"/config.json"))

NGFS_filtered_path=config["filtered_NGFS"]
NGFS_path=config["NGFS"]
out_path=config["out_path"]

filter_variables=[
    "Investment",
    "GDP|PPP|Counterfactual without damage"
]

filtered_df=filter_NGFS(NGFS_filtered_path,NGFS_path,filter_variables)
growth_rates_df=compute_growth_rates(filtered_df)

growth_rates_df.to_csv(out_path, index=False)