import pandas as pd
import os
import json

from lib import create_sector_template,fill_sector_productivity, reformat_df

this_folder = os.path.dirname(__file__)
config = json.load(open(this_folder+"/config.json"))

productivities_df_path=config["productivities"]
mapping_regions_path=config["regions_mapping"]
sectors_mapping_path=config["sectors_prod_mapping"]
out_path=config["out_path"]
filtered_NGFS_path = config["filtered_NGFS"]

productivities_df = pd.read_csv(productivities_df_path, header=0)
mapping_regions_df = pd.read_csv(mapping_regions_path, header=0)
sectors_mapping_df = pd.read_csv(sectors_mapping_path, header=0)
filtered_NGFS = pd.read_csv(filtered_NGFS_path, header=0)


template_df=create_sector_template(filtered_NGFS,sectors_mapping_df)


productivities_out_df = fill_sector_productivity(template_df, productivities_df, sectors_mapping_df, mapping_regions_df)

productivities_out_df = reformat_df(productivities_out_df)

productivities_out_df.to_csv(out_path, index=False)
