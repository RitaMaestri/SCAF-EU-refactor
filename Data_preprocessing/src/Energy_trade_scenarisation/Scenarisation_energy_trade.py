import csv
import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections.abc import Iterable
import json
from pathlib import Path


import traceback


# useful paths
this_folder = os.path.dirname(__file__)
config = json.load(open(this_folder+"/config.json"))

NGFS_filtered_path = config["NGFS_filtered"]
NGFS_path = config["NGFS"]
mapping_path = config["mapping"]
IOTs_path = config["IOTs_path"]
out_path = config["out_path"]

# useful mappings
variables_names = pd.read_excel(mapping_path, sheet_name='Variables names', header=0, index_col=0)

regions_mapping = pd.read_excel(mapping_path, sheet_name='Regions mapping',header=0).set_index('SCAF region')['REMIND region']

prices_conversion = pd.read_excel(mapping_path, sheet_name='Prices conversion', header=0)


#extract M from csv

folder_path = Path(IOTs_path)
region_names = [f.stem for f in folder_path.glob('*.csv')]


# Create an empty DataFrame with no columns initially
M = pd.DataFrame([], index=['M'])
X = pd.DataFrame([], columns=['X'])
for region in region_names:

    IOT=pd.read_csv(str(IOTs_path)+region+".csv",header=[0, 1], index_col=[0, 1])

    N_sectors= int((IOT.index.get_level_values(0) == 'CI_imp').sum()- 1 )
    
    ## BUILD IMPORT VECTOR ##
    #select the import row from the IOT
    M_r = IOT.loc[[("M", "IMP")], IOT.columns[:N_sectors]]

    M_r.columns = pd.MultiIndex.from_tuples([(region, col[1]) for col in M_r.columns])
    M_r.index = M_r.index.droplevel(1)

    #concatenate the regional import to the import collection M
    M = pd.concat([M, M_r], axis=1)

    ## BUILD EXPORT VECTOR ##
    X_r =IOT.iloc[:N_sectors, IOT.columns.get_level_values(1) == "EXP"]

    X_r.columns = ["X"]
    X_r.index = pd.MultiIndex.from_tuples([(region, ind[1]) for ind in X_r.index])

    X = pd.concat([X, X_r], axis=0)

    

# EXIOBASE imports and exports as computed precedently in M_EUR2020


# reformat regions and sectors names
M = M.T
M.index = pd.MultiIndex.from_tuples(M.index, names=["Region", "Sector"])
X.index = pd.MultiIndex.from_tuples(X.index, names=["Region", "Sector"])


region_level = M.index.levels[0]
NGFS_regions = region_level.map(regions_mapping)

M.index = M.index.set_levels(NGFS_regions, level=0)
X.index = X.index.set_levels(NGFS_regions, level=0)

M.index = M.index.set_levels(M.index.levels[1].str.capitalize(), level=1)
X.index = X.index.set_levels(X.index.levels[1].str.capitalize(), level=1)

######################################
#### Aggregate other energy to oil####
######################################
IOT.loc[[("M", "IMP")], IOT.columns[:N_sectors]]
for r in NGFS_regions:
    M.loc[(r, 'Oil')] += M.loc[(r, 'Other energy')]
    M.drop((r, 'Other energy'), inplace=True)

    X.loc[(r, 'Oil')] += X.loc[(r, 'Other energy')]
    X.drop((r, 'Other energy'), inplace=True)


# output dataframe variables names
world_prices = variables_names["world_prices"]
net_trade_volumes = variables_names["net_trade_volumes"]
imp_prices = variables_names["imp_prices"]
exp_prices = variables_names["exp_prices"]
imp_volumes = variables_names["imp_volumes"]
exp_volumes = variables_names["exp_volumes"]


###### create a filtered NGFS database to lighten the program #####

if Path(NGFS_filtered_path).exists():
    NGFS = pd.read_csv(NGFS_filtered_path, index_col=0)

else:
    NGFS_total = pd.read_excel(
        NGFS_path, sheet_name='data', header=0)

    # filter world prices and regional net trade of energy
    NGFS = NGFS_total[
        (NGFS_total["Model"] == 'REMIND-MAgPIE 3.2-4.6') &
        ((NGFS_total["Region"].str.startswith('REMIND') & NGFS_total["Variable"].isin(net_trade_volumes.tolist())) |
         (NGFS_total["Region"].str.startswith('World') & NGFS_total["Variable"].isin(world_prices.tolist())))
    ]

    NGFS.to_csv("NGFS_energy_trade_filtered.csv")
    del(NGFS_total)

# filter out columns with only NANs
NGFS = NGFS.loc[:, NGFS.notna().all(axis=0)]

##### create categories for the new database #####


# model
model = np.unique(NGFS["Model"])

# scenarios
scenarios = np.unique(NGFS["Scenario"])

# regions except World
regions = np.unique(NGFS["Region"])
regions = regions[regions != "World"]

# variables
regional_volumes = net_trade_volumes.tolist() + imp_volumes.tolist() + \
    exp_volumes.tolist()
regional_prices = imp_prices.to_list() + exp_prices.to_list()
world_variables = world_prices.tolist()

# units
volumes_unit = ["EJ"]
# needs to be one of those in price_conversion
prices_unit = ["M EUR2020/" + volumes_unit[0]]


# build database

N_rows = len(scenarios) * (len(world_variables) +
                           (len(regional_volumes)+len(regional_prices)) * len(regions))

out_data = pd.DataFrame(columns=NGFS.columns, index=range(N_rows))

#fill in database

out_data["Variable"] = len(
    scenarios) * (world_variables + (regional_volumes+regional_prices)*len(regions))

out_data["Unit"] = len(scenarios) * (prices_unit * len(world_variables) + (
    volumes_unit*len(regional_volumes)+prices_unit*len(regional_prices))*len(regions))

out_data["Region"] = len(scenarios) * (["World"] * len(world_variables) +
                                       np.repeat(regions, len(regional_volumes+regional_prices)).tolist())

out_data["Scenario"] = np.repeat(scenarios, N_rows/len(scenarios))

out_data["Model"] = np.repeat(model, N_rows)


# check for duplicates in the rows

duplicates = out_data[out_data.duplicated()]

assert duplicates.empty, "Error: there are duplicate rows in out_data"


################################################################################
## define getter and setter for scenarios dataframes for enhanced readability ##
################################################################################

# by chatgpt

def get_value(df, variable=None, scenario=None, region=None, year=None):
    # Start with the full dataframe
    filtered_df = df.copy()

    # Determine which dimensions remain unconstrained
    index_cols = []

    # Apply variable filter or keep it as an index if unconstrained
    if variable is not None:
        if isinstance(variable, Iterable) and not isinstance(variable, str):
            filtered_df = filtered_df[filtered_df["Variable"].isin(variable)]
        else:
            filtered_df = filtered_df[filtered_df["Variable"] == variable]
    else:
        index_cols.append("Variable")  # Keep it as an index if unconstrained

    # Apply scenario filter or keep it as an index if unconstrained
    if scenario is not None:
        filtered_df = filtered_df[filtered_df["Scenario"] == scenario]
    else:
        index_cols.append("Scenario")

    # Apply region filter or keep it as an index if unconstrained
    if region is not None:
        filtered_df = filtered_df[filtered_df["Region"] == region]
    else:
        index_cols.append("Region")

    # Set dynamic index if needed
    if index_cols:
        filtered_df = filtered_df.set_index(index_cols)

    # If a year is specified, return that column's values (as a Series or DataFrame)
    if year is not None:
        if year in df.columns:
            return filtered_df[year]  # Return only the requested year's values
        else:
            return None  # Year not found in DataFrame

    # If no year is specified, return the filtered DataFrame with all years
    return filtered_df if not filtered_df.empty else None






def set_value(df, variable=None, scenario=None, region=None, year=None, new_value=None):

    if year is None:
        raise ValueError("You must specify a year.")

    # Start with all rows, then apply filters
    mask = pd.Series(True, index=df.index)

    if variable is not None:
        mask &= df["Variable"] == variable
    if scenario is not None:
        mask &= df["Scenario"] == scenario
    if region is not None:
        mask &= df["Region"] == region

    if mask.any():
        # Check if new_value is a scalar
        if np.isscalar(new_value):
            # If scalar, check if it's close to zero
            if np.isclose(new_value, 0, atol=1e-06):
                new_value = 0
        else:
            # If array, apply element-wise check for values close to zero
            new_value = np.array(new_value, dtype=float)
            new_value[np.isclose(new_value, 0, atol=1e-06)] = 0

        # Update all matching rows
        df.loc[mask, year] = new_value
        
        
def compute_aggregate_price(df, volume_var, price_var, year, scenario, region=None):

    # Retrieve volumes and prices
    volumes = get_value(df, variable=volume_var, year=year, scenario=scenario, region=region).to_numpy()
    prices = get_value(df, variable=price_var, year=year, scenario=scenario, region=region).to_numpy()

    # Compute total volume and value
    total_volume = volumes.sum()
    total_value = (volumes * prices).sum()

    # Raise an error if total_volume is zero
    if total_volume == 0:
        return 0
    return total_value / total_volume





#######################
##### CALIBRATION #####
#######################


# I accept values from EXIOBASE and prices from REMIND
years = out_data.columns[5:]
y_0 = years[0]

E_sec = variables_names.index[:-1]


for r in regions:
    
    regional_exp_volume=0
    regional_imp_volume=0
    regional_exp_value=0
    regional_imp_value=0
    
    for s in E_sec:
        # I pick the first scenario because the price is the same for every scenario
        price_US2010 = get_value( NGFS, variable=world_prices[s], year=y_0).iloc[0]
        price = price_US2010 * prices_conversion[prices_unit].iloc[0, 0]

        exp_volume = X.loc[(r, s)].item() / price.item()
        imp_volume = M.loc[(r, s)].item() / price.item()
        net_trade = exp_volume-imp_volume
        
        set_value(out_data, variable=exp_volumes.loc[s], region=r, year=y_0, new_value=exp_volume)
        set_value(out_data, variable=imp_volumes.loc[s], region=r, year=y_0, new_value=imp_volume)
        set_value(out_data, variable=net_trade_volumes.loc[s], region=r, year=y_0, new_value=net_trade)
        set_value(out_data, variable=world_prices.loc[s], year=y_0, new_value=price)
        set_value(out_data, variable=exp_prices.loc[s], region=r, year=y_0, new_value=price)
        set_value(out_data, variable=imp_prices.loc[s], region=r, year=y_0, new_value=price)
        
        regional_exp_volume+=exp_volume
        regional_imp_volume+=imp_volume
        regional_exp_value+=X.loc[(r, s)].item()
        regional_imp_value+=M.loc[(r, s)].item()
    
    set_value(out_data, variable=exp_volumes.loc["Energy"], region=r, year=y_0, new_value=regional_exp_volume)
    set_value(out_data, variable=imp_volumes.loc["Energy"], region=r, year=y_0, new_value=regional_imp_volume)
    set_value(out_data, variable=net_trade_volumes.loc["Energy"], region=r, year=y_0, new_value=regional_exp_volume-regional_imp_volume)



# **Compute world price (all regions and sectors)**
world_price = compute_aggregate_price(out_data, exp_volumes[E_sec], exp_prices[E_sec], y_0, scenarios[0])
set_value(out_data, variable=world_prices.loc["Energy"], year=y_0, new_value=world_price)

# **Compute regional "Energy" prices**
for r in regions:
    regional_export_price = compute_aggregate_price(out_data, exp_volumes[E_sec], exp_prices[E_sec], y_0, scenarios[0], region=r)
    set_value(out_data, variable=exp_prices.loc["Energy"], year=y_0, region=r, new_value=regional_export_price)
    
    regional_import_price = compute_aggregate_price(out_data, imp_volumes[E_sec], imp_prices[E_sec], y_0, scenarios[0], region=r)
    set_value(out_data, variable=imp_prices.loc["Energy"], year=y_0, region=r, new_value=regional_import_price)





########################################
#### EQUILIBRATE REMIND'S NET TRADE ####
########################################

# if the net trade is negative the net importers increase their balance
# if the net trade is positive the net exporters decrease their balance
def distribution_1(net_trade_ys):
    net_trade_tot = net_trade_ys.sum()

    if net_trade_tot > 0:
        net_exporters = net_trade_ys[net_trade_ys > 0]
        net_exporters -= net_exporters/sum(net_exporters) * net_trade_tot
        net_trade_ys[net_trade_ys > 0] = net_exporters

    if net_trade_tot < 0:
        net_importers = net_trade_ys[net_trade_ys < 0]
        net_importers -= net_importers/sum(net_importers) * net_trade_tot
        net_trade_ys[net_trade_ys < 0] = net_importers

    return net_trade_ys

# the net trade is balanced half by net importers and half by net exporters


def distribution_2(net_trade_ys):
    net_trade_tot = net_trade_ys.sum()

    net_exporters = net_trade_ys[net_trade_ys > 0]
    net_exporters -= net_exporters/sum(net_exporters) * net_trade_tot/2
    net_trade_ys[net_trade_ys > 0] = net_exporters

    net_importers = net_trade_ys[net_trade_ys < 0]
    net_importers -= net_importers/sum(net_importers) * net_trade_tot/2
    net_trade_ys[net_trade_ys < 0] = net_importers

    #correct to guarantee that the sum of net trades is within 1e-5
    resudual = net_trade_ys.sum()
    if abs(resudual) > 1e-5:  
        idx_max = np.argmax(np.abs(net_trade_ys))  
        net_trade_ys[idx_max] -= resudual  

    return net_trade_ys


#equilibrate net trade

for S in scenarios:
    for sec in E_sec:
        for year in years[1:]:
            net_trade_ys = get_value( NGFS, variable=net_trade_volumes[sec], scenario=S, year=year)
            
            net_trade_balanced = distribution_2(net_trade_ys).to_numpy()
                
            set_value(out_data, variable=net_trade_volumes[sec], scenario=S, year=year, new_value=net_trade_balanced)



####################################
#### SECTORAL PRICES EVOLUTION #####
####################################

for S in scenarios:
    for sec in E_sec:
        for year in years[1:]:
            price_US2010 = get_value(NGFS, variable=world_prices[sec], scenario=S, year=year)
            price = (price_US2010 *prices_conversion[prices_unit].iloc[0, 0]).item()

            set_value(out_data, variable=world_prices[sec], scenario=S, year=year, new_value=price)
            set_value(out_data, variable=imp_prices[sec], scenario=S, year=year, new_value=price)
            set_value(out_data, variable=exp_prices[sec], scenario=S, year=year, new_value=price)

#################################################
####### IMPORT AND EXPORT DISAGGREGATION ########
#################################################


# def compute_alpha(prev_imp_price, curr_imp_price, prev_exp_price, curr_exp_price, curr_imp, curr_exp, prev_net_trade, curr_net_trade)

def compute_alpha(prev_export, prev_import, curr_net_trade):
    # Convert arrays to float type to avoid 'object' dtype issues
    prev_export = np.asarray(prev_export, dtype=np.float64)
    prev_import = np.asarray(prev_import, dtype=np.float64)
    curr_net_trade = np.asarray(curr_net_trade, dtype=np.float64)

    # Compute alpha with corrected parentheses and type handling
    alpha = (curr_net_trade + np.sqrt(4 * prev_export * prev_import + np.square(curr_net_trade))) / (2 * prev_export)

    return alpha

def compute_exp_imp(prev_export, prev_import, curr_net_trade):
    # Avoid division by zero
    mask = (np.abs(prev_export) > 1e-4) & (np.abs(prev_import) > 1e-4)   
    alpha = np.ones_like(prev_export)  # Default alpha as 1
    alpha[mask] = compute_alpha(prev_export[mask], prev_import[mask], curr_net_trade[mask])

    curr_export = np.zeros_like(prev_export)
    curr_import = np.zeros_like(prev_import)

    curr_export[mask] = prev_export[mask] * alpha[mask]
    curr_import[mask] = prev_import[mask] / alpha[mask]
    
    no_export_mask = np.abs(prev_export) <= 1e-4
    # Handling the case where prev_export is 0
    
    # Case 1: curr_net_trade < 0
    neg_mask = curr_net_trade < 0
    curr_export[no_export_mask & neg_mask] = 0
    curr_import[no_export_mask & neg_mask] = -curr_net_trade[no_export_mask & neg_mask]
    
    # Case 2: curr_net_trade > 0
    pos_mask = curr_net_trade > 0
    curr_import[no_export_mask & pos_mask] = prev_import[no_export_mask & pos_mask]
    curr_export[no_export_mask & pos_mask] = curr_net_trade[no_export_mask & pos_mask] + curr_import[no_export_mask & pos_mask]
    
    # Case 3: curr_net_trade == 0
    zero_mask = curr_net_trade == 0
    curr_import[no_export_mask & zero_mask] = 0
    prev_import[mask & zero_mask] = 0  # Only modify prev_import where prev_export was nonzero
    
    # Handling the case where prev_import is 0

    no_import_mask = np.abs(prev_import) <= 1e-4
    # curr_net_trade < 0 → net importer
    neg_mask = curr_net_trade < 0
    curr_export[no_import_mask & neg_mask] = 0
    curr_import[no_import_mask & neg_mask] = -curr_net_trade[no_import_mask & neg_mask]
    
    # curr_net_trade > 0 → net exporter
    pos_mask = curr_net_trade > 0
    curr_import[no_import_mask & pos_mask] = 0
    curr_export[no_import_mask & pos_mask] = curr_net_trade[no_import_mask & pos_mask]
    
    # curr_net_trade == 0 → no trade
    zero_mask = curr_net_trade == 0
    curr_import[no_import_mask & zero_mask] = 0
    curr_export[no_import_mask & zero_mask] = 0

    
    
    return curr_export, curr_import


for S in scenarios:
    for sec in E_sec:
        for i in range(1, len(years)):  # Loop over years, skipping the first

            # Define previous and current year
            prev_year, curr_year = years[i-1], years[i]

            prev_export = get_value(out_data, variable=exp_volumes[sec], scenario=S, year=prev_year)
            prev_import = get_value(out_data, variable=imp_volumes[sec], scenario=S, year=prev_year)
            #prev_net_trade = get_value(out_data, variable=net_trade_volumes[sec], scenario=S, year=prev_year).to_numpy()
            curr_net_trade = get_value(out_data, variable=net_trade_volumes[sec], scenario=S, year=curr_year)
            
            curr_export,curr_import = compute_exp_imp(prev_export, prev_import, curr_net_trade)

            set_value(out_data, variable=exp_volumes[sec], scenario=S, year=curr_year, new_value=curr_export)
            set_value(out_data, variable=imp_volumes[sec], scenario=S, year=curr_year, new_value=curr_import)
            
            prev_export = np.asarray(prev_export, dtype=np.float64)
            prev_import = np.asarray(prev_import, dtype=np.float64)
            curr_net_trade = np.asarray(curr_net_trade, dtype=np.float64)

#fill in aggregate energy import and export volumes
for y in years:
    for S in scenarios:
        for r in regions:
            export_tot= get_value(out_data, variable=exp_volumes[E_sec], year=y, region=r,scenario=S).sum()
            set_value(out_data, variable=exp_volumes["Energy"], scenario=S, year=y,region=r, new_value=export_tot)
            
            import_tot= get_value(out_data, variable=imp_volumes[E_sec], year=y, region=r,scenario=S).sum()
            set_value(out_data, variable=imp_volumes["Energy"], scenario=S, year=y, region=r, new_value=import_tot)
            
            set_value(out_data, variable=net_trade_volumes["Energy"], scenario=S, region=r, year=y, new_value= export_tot-import_tot)


###########################################################
#### EVOLUTION OF REGIONAL PRICES OF AGGREGATED ENERGY ####
###########################################################

# **Compute world price (all regions and sectors)**
for y in years[1:]:
    for S in scenarios:
        world_price = compute_aggregate_price(out_data, exp_volumes[E_sec], exp_prices[E_sec], y, scenario=S)
        set_value(out_data, variable=world_prices.loc["Energy"], year=y, new_value=world_price)

# **Compute regional "Energy" prices**

        for r in regions:
            regional_export_price = compute_aggregate_price(out_data, exp_volumes[E_sec], exp_prices[E_sec], y, scenario=S, region=r)
            set_value(out_data, variable=exp_prices.loc["Energy"], year=y, region=r, scenario= S, new_value=regional_export_price)
        
            regional_import_price = compute_aggregate_price(out_data, imp_volumes[E_sec], imp_prices[E_sec], y, scenario=S, region=r)
            set_value(out_data, variable=imp_prices.loc["Energy"], year=y, region=r, scenario= S, new_value=regional_import_price)




#########################################################
###################  TESTS  #############################
#########################################################

#set_value(out_data, variable=net_trade_volumes.loc["Gas"], year=y, region=r, new_value=-1)


def check_trade_balance(net_trade, imp, exp, tol=1e-6):
    # Convert inputs to numpy float64 arrays
    check=False
    net_trade = np.asarray(net_trade, dtype=np.float64)
    imp = np.asarray(imp, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)

    # Replace None or NaN values with 0 to prevent errors
    net_trade = np.nan_to_num(net_trade, nan=0.0)
    imp = np.nan_to_num(imp, nan=0.0)
    exp = np.nan_to_num(exp, nan=0.0)

    # Compute the imbalance
    imbalance = net_trade + imp - exp

    # Find indices where absolute imbalance is greater than tol
    mask = np.abs(imbalance) > tol
    

    # If any imbalance is significant, print the values
    if np.any(mask):
        print("Imbalance detected at indices:", np.where(mask)[0])
        print("net_trade:", net_trade[mask])
        print("imp:", imp[mask])
        print("exp:", exp[mask])
        print("Unbalance values:", imbalance[mask])
        check= True
    return check
        

for S in scenarios:
    for r in regions:
        for sec in variables_names.index:
            for y in years:
                imp=get_value(out_data,variable=imp_volumes[sec],scenario=S,region=r,year=y)
                exp=get_value(out_data,variable=exp_volumes[sec],scenario=S,region=r,year=y)
                net_trade=get_value(out_data,variable=net_trade_volumes[sec],scenario=S,region=r,year=y)
                if check_trade_balance(net_trade, imp, exp, tol=1e-6):
                    print("Scenario: ", S)
                    print("region: ", r)
                    print("sector: ", sec)
                    print("year: ", y, "\n")
                

###### check negative variables ######   

for y in years:
    negative_rows= out_data.loc[(~out_data["Variable"].isin(net_trade_volumes)) & (out_data[y]<0)]
    if not negative_rows.empty:
        print("Negative values found")
        
###### check that the overall energy market is equilibrated ######

for S in scenarios:
    for y in years:
        imp = get_value( out_data, variable = imp_volumes["Energy"], scenario=S, year=y )
        imp_price = get_value( out_data, variable = imp_prices["Energy"], scenario=S, year=y )
        exp = get_value( out_data, variable = exp_volumes["Energy"], scenario=S, year=y )
        exp_price = get_value( out_data, variable = exp_prices["Energy"], scenario=S, year=y )
        
        tol=10e-4
        equilibrium = sum( (imp*imp_price) - (exp*exp_price) )
        if abs(equilibrium) > tol:
            print("Unbalance of the international aggregated energy market found at")
            print("Scenario: ", S)
            print("year: ", y)
            print("unbalance:",equilibrium,"M EUR 2020 \n")


############################
###### SELECT OUTPUT #######
############################
filter_valriable=["Trade|Primary Energy|Volume|Import",
                  "Trade|Primary Energy|Volume|Export",
                  "Price|Primary Energy|Import",
                  "Price|Primary Energy|Export"]

mask=out_data['Variable'].isin(filter_valriable)

out_data[mask].to_csv(out_path+"Corrected_trade_volumes.csv", index=False,encoding='utf-8-sig')

                
                
                
                
                
                
                
                
                
                
                

                
                
                
