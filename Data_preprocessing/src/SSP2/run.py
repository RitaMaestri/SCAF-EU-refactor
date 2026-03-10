import pandas as pd
import os
import json

from lib import create_population_template,fill_population_template

this_folder = os.path.dirname(__file__)
config = json.load(open(this_folder+"/config.json"))

NGFS_filtered_path=config["filtered_NGFS"]
mapping_regions_path=config["mapping_regions"]
out_path=config["out_path"]
population_raw_path=config["Population"]


NGFS_filtered_df = pd.read_csv(NGFS_filtered_path, header=0)

mapping_regions_df = pd.read_csv(mapping_regions_path, header=0)

population_raw_df = pd.read_csv(population_raw_path, header=0)

template_df = create_population_template(NGFS_filtered_df)
filled_population_df = fill_population_template(template_df, population_raw_df, mapping_regions_df, ssp_column="pop_SSP2")


filled_population_df.to_csv(out_path, index=False)