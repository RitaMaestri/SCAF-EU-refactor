import csv
import sys
import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections.abc import Iterable
import json
from pathlib import Path
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

print("PYTHON:", sys.executable)
print("CWD:", os.getcwd())

from lib import (filter_REMIND,
                 aggregate_IOT_energy_consumption,
                 aggregate_energy_uses,
                 get_REMIND_units,
                 generate_output_template,
                 Energy_Values_Allocation,
                 fill_calibration_year,
                 rename_regions,
                 extract_row_until_nan,
                 get_IOT_row_labels_by_identifier,
                 aggregate_by_consumer_and_use,
                 build_availabilities_df,
                 build_priorities_dict,
                 project_variables,
                 aggregate_prices_volumes,
                 extract_regional_energy_sales_taxes,
                 convert_to_net_values,
                 compute_mean_energy_price,
                 compute_delta_volumes,
                 compute_delta_prices,
                 )


# useful paths
this_folder = os.path.dirname(__file__)
config = json.load(open(this_folder+"/config.json"))

REMIND_path=config["REMIND"]
REMIND_filtered_path=config["REMIND_filtered"]
cache_path= config["cache_path"]

IEA_mapping_path=config["IEA_mapping"]
IEA_data_path=config["IEA"]
regions_mapping_path=config["regions_mapping"]
energy_uses_path= config["energy_uses"]
energy_consumers_path= config["energy_consumers"]

map_IOT_path= config["IOT-energy_consumers"]
map_REMIND_path= config["REMIND-energy_uses"]

priorities_path = config["priorities"]
#keys_path = config["keys"]
IOTs_path= config["IOTs_path"]
out_path= config["out_path"]

energy_sales_taxes_mapping_path = config["energy_sales_taxes_mapping"]
value_unit = config["value_unit"]

regions_mapping= pd.read_csv(regions_mapping_path, header=0)
#regions=regions_mapping["region_REMIND"]
########### FILTER REMIND #############

REMIND_raw = pd.read_csv(REMIND_path, sep=';')
filtered_REMIND = filter_REMIND(REMIND_raw, REMIND_filtered_path)
############ AGGREGATE REMIND ENERGY TYPES ##############
#import mapping
map_REMIND_energy_uses = pd.read_excel(map_REMIND_path, header=0)
volume_unit, price_unit = get_REMIND_units(filtered_REMIND, map_REMIND_energy_uses)
if "volume_unit" in config:
    volume_unit = config["volume_unit"]
if "price_unit" in config:
    price_unit = config["price_unit"]

# aggregate REMIND energy types
markets_dict = aggregate_energy_uses(filtered_REMIND, map_REMIND_energy_uses, value_unit, price_unit)

REMIND_prices = markets_dict["prices"]

REMIND_volumes = markets_dict["volumes"]

REMIND_values = markets_dict["values"]


#export dataframe

#REMIND_prices.to_csv(cache_path+"prices.csv", index=False)
#REMIND_volumes.to_csv(cache_path+"volumes.csv", index=False)
#REMIND_values.to_csv(cache_path+"values.csv", index=False)

#create regional dictionaries of IOT energy consumptions dataframes
def import_IOT(IOT_path):
    # --- 1. Load the 12 CSV files into DataFrames ---
    iot_dict = {}

    # Loop through all files in the specified directory
    for file in os.listdir(IOT_path):
        if file.endswith('.csv'):
            # Extract region name from the filename 
            region_name = os.path.splitext(file)[0]
            file_path = os.path.join(IOT_path, file)

            df = pd.read_csv(file_path, header=[0, 1], index_col=[0, 1])

            iot_dict[region_name] = df
            
    # Return all three data structures
    return iot_dict

IOT_dict = import_IOT(IOTs_path)

IOT_dict = rename_regions(regions_mapping, IOT_dict)

col_label_df = pd.read_excel(map_IOT_path, sheet_name='col_label', header=0)

row_label_df = pd.read_excel(map_IOT_path, sheet_name='row_label', header=0)


# aggregate IOT energy consumptions

energy_sales_taxes_mapping = pd.read_csv(energy_sales_taxes_mapping_path, header=0)

energy_sales_taxes_dict = extract_regional_energy_sales_taxes(IOT_dict, energy_sales_taxes_mapping, "Tax_block")

IOT_gross_energy_consumption_dict = aggregate_IOT_energy_consumption(IOT_dict, col_label_df, row_label_df)

IOT_energy_consumption_dict = convert_to_net_values(IOT_gross_energy_consumption_dict,energy_sales_taxes_dict)

###################################
#### identify calibration year ####
###################################

year_cols = [c for c in REMIND_values.columns if str(c).isdigit()]

calibration_year = year_cols[0]

####################################
###### generate priorities #########
####################################

#import priorities and IEA data

priorities_df = pd.read_csv(priorities_path,header=0, index_col=False)
IEA_mapping=pd.read_csv(IEA_mapping_path, header=0)
IEA_data=pd.read_csv(IEA_data_path, header=0)

#aggregate_IEA_data

aggregated_IEA=aggregate_by_consumer_and_use(IEA_data,IEA_mapping)

aggregated_IEA = rename_regions(regions_mapping, aggregated_IEA)

energy_availability_per_consumer = build_availabilities_df(REMIND_volumes, aggregated_IEA, calibration_year)

priorities_dict = build_priorities_dict(energy_availability_per_consumer, priorities_df)

#keys_df= pd.read_csv(keys_path, index_col=0)

#import energy consumers and uses mappings

energy_consumers_df=pd.read_csv(energy_consumers_path,header=0)

energy_uses_df=pd.read_csv(energy_uses_path,header=0)


#extract consumers and uses for the optimization algorithm

mask_consumers=energy_consumers_df["allocated_with_optimization_algo"]
mask_uses=energy_uses_df["allocated_with_optimization_algo"]
energy_consumers_opt = list(energy_consumers_df[mask_consumers]["energy_consumer"])
energy_uses_opt = list(energy_uses_df[mask_uses]["energy_use"])


consumers_X_uses_df = generate_output_template(filtered_REMIND, energy_consumers_opt, energy_uses_opt, price_unit, volume_unit)
consumers_X_uses_calibration_df = consumers_X_uses_df.copy()



###############################################
###### allocate energies to consumers #########
###############################################
consumers_X_uses_calibration_df_path = cache_path+"calibration_output.csv"
print("Starting cycle")

if False:
#if os.path.isfile(consumers_X_uses_calibration_df_path):
    consumers_X_uses_calibration_df=pd.read_csv(consumers_X_uses_calibration_df_path,header=0)
else:
    for model in pd.unique(consumers_X_uses_df["Model"]):
        for scenario in pd.unique(consumers_X_uses_df["Scenario"]):
            for region in pd.unique(consumers_X_uses_df["Region"]):
                mask = ((REMIND_values["Model"] == model) &
                        (REMIND_values["Scenario"] == scenario) &
                        (REMIND_values["Region"] == region))
                
                values_to_allocate_df=REMIND_values[mask][["energy_use", calibration_year]]
                values_to_allocate = pd.Series(values_to_allocate_df[calibration_year].values, index=values_to_allocate_df["energy_use"])

                prices_to_allocate_df=REMIND_prices[mask][["energy_use", calibration_year]]
                prices_to_allocate = pd.Series(prices_to_allocate_df[calibration_year].values, index=prices_to_allocate_df["energy_use"])

                IOT_energy_consumption_df=IOT_energy_consumption_dict[region]

                IOT_energy_consumption_df_opt = IOT_energy_consumption_df[energy_consumers_opt]

                IOT_enrgy_consumption=pd.Series(IOT_energy_consumption_df_opt.iloc[0].values, index=IOT_energy_consumption_df_opt.columns)
                
                #generate VA key
                VA_row_multiindex=get_IOT_row_labels_by_identifier(row_label_df, "VA")
                
                key_VA =  extract_row_until_nan(IOT_dict[region], VA_row_multiindex )
                
                key_VA["HOUSEHOLDS"] = 0


                allocation = Energy_Values_Allocation(IOT_E_consumptions=IOT_enrgy_consumption,
                REMIND_E_uses=values_to_allocate,
                REMIND_E_prices= prices_to_allocate,
                priorities=priorities_dict[region],
                key=key_VA)


                rescaling_factor=allocation.rescale_REMIND_energy_values()

                allocation.allocate_forced_energy_values()
                allocation.adjust_key_for_forced_values()

                disaggregated_energy = allocation.compute_disaggregated_energy()
                #check the constraint error
                #print("In region ",region, "\n max vertical sum error: ",max(abs(-1 + disaggregated_energy.sum(axis=0) / allocation.REMIND_E_uses)))
                #print("\n max horizontal sum error: ",max(-1 + disaggregated_energy.sum(axis=1) / allocation.IOT_E_consumptions))

                disaggregated_prices = allocation.compute_prices_matrix()
                disaggregated_volumes = allocation.compute_volumes_matrix()

                consumers_X_uses_calibration_df=fill_calibration_year(
                    df_template=consumers_X_uses_calibration_df,
                    allocation_matrix= disaggregated_volumes,
                    model= model,
                    scenario= scenario,
                    region= region,
                    variable="Volume",
                    first_year = calibration_year)
                
                consumers_X_uses_calibration_df=fill_calibration_year(
                    df_template=consumers_X_uses_calibration_df,
                    allocation_matrix= disaggregated_prices,
                    model= model,
                    scenario= scenario,
                    region= region,
                    variable="Price",
                    first_year = calibration_year)
                
    consumers_X_uses_calibration_df.to_csv(consumers_X_uses_calibration_df_path, index=False)


projected_volumes_df = project_variables(output = consumers_X_uses_calibration_df, reference = REMIND_volumes, variable_type= "Volume")

projected_df = project_variables(output = projected_volumes_df, reference = REMIND_prices, variable_type= "Price")

projected_df.to_csv(cache_path+"projected_output.csv", index=False)

aggregated_df = aggregate_prices_volumes(projected_df, value_unit)

aggregated_df.to_csv(cache_path+"aggregated_output.csv", index=False)

#### COMPUTE DELTA VOLUMES AND PRICES #####

mean_energy_prices_df = compute_mean_energy_price(aggregated_df)

delta_mask = energy_consumers_df["allocation_exception"]

delta_label = list(energy_consumers_df[delta_mask]["energy_consumer"])[0]

delta_vol = compute_delta_volumes(IOT_energy_consumption_dict, mean_energy_prices_df, delta_label, volume_unit)

delta_price = compute_delta_prices(mean_energy_prices_df, delta_label)

#######################
### CONCATENATE DFs ###
#######################

vol_price_df = aggregated_df.loc[aggregated_df["Variable"].isin(["Volume", "Price"])].copy()

final_df = pd.concat([vol_price_df, delta_vol, delta_price], ignore_index=True)

final_df['Variable'] = final_df['Variable'].replace({'Volume': 'Energy consumption volume'})
final_df['Variable'] = final_df['Variable'] + "|" + final_df['Energy consumers']
final_df = final_df.drop(columns=['Energy consumers'])

final_df.to_csv(out_path+"hybridization_df.csv", index=False)

