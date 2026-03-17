import pandas as pd
import itertools
import numpy as np


def filter_REMIND(REMIND_df):
    # Drop trailing NaN-named column produced by the trailing ';' in .mif files
    REMIND = REMIND_df.loc[:, REMIND_df.columns.notna()]

    # Drop World aggregate rows
    REMIND = REMIND[REMIND["Region"] != "World"]

    # Drop year columns before 2020
    year_cols_to_drop = [c for c in REMIND.columns if str(c).isdigit() and int(c) < 2020]
    REMIND = REMIND.drop(columns=year_cols_to_drop)

    return REMIND

def create_population_template(df):
    # 1. Identify the columns that represent years
    year_cols = [col for col in df.columns if str(col).isdigit()]
    
    # 2. Get unique values for Model, Scenario, and Region
    models = df["Model"].unique()
    scenarios = df["Scenario"].unique()
    regions = df["Region"].unique()
    
    # 3. Create all possible combinations of Model, Scenario, and Region
    combos = list(itertools.product(models, scenarios, regions))
    
    # 4. Build the new DataFrame rows
    new_rows = []
    for model, scenario, region in combos:
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Population",
            "Unit": "Million people",
        }
        
        # 4a. Set all year columns to NaN (empty)
        for year in year_cols:
            row[year] = np.nan
        
        new_rows.append(row)
    
    new_df = pd.DataFrame(new_rows)
    
    # 5. Keep the columns in the same order as the original DataFrame
    ordered_cols = ["Model", "Scenario", "Region", "Variable", "Unit"] + year_cols
    new_df = new_df[ordered_cols]
    
    return new_df



def fill_population_template(template_df, population_df, mapping_df, ssp_column="pop_SSP2"):
    """
    Fill the population template DataFrame with populations from SSP data.
    
    Parameters:
    - template_df: template DataFrame (Model x Scenario x Region)
    - population_df: DataFrame with population data (dummy,dummy,pop_SSP1,...)
    - mapping_df: DataFrame with columns ['region_SCAF','region_REMIND'] for mapping
    - ssp_column: column in population_df to use (default 'pop_SSP2')
    
    Returns:
    - template_df_filled: DataFrame with populations filled
    """
    
    # 1. Create mapping dictionary: SCAF region -> REMIND region
    region_map = dict(zip(mapping_df["region_SCAF"], mapping_df["region_REMIND"]))
    
    # 2. Identify year columns in template (numeric)
    year_cols = [col for col in template_df.columns if str(col).isdigit()]
    
    # 3. Copy template
    filled_df = template_df.copy()
    
    # 4. Fill population
    # First, map REMIND region to SCAF region in population_df
    # Reverse mapping: REMIND region -> SCAF region
    reverse_map = {v: k for k, v in region_map.items()}
    
    for idx, row in filled_df.iterrows():
        ngfs_region = row["Region"]
        # Find corresponding SCAF region
        scaf_region = reverse_map.get(ngfs_region, None)
        if scaf_region is None:
            # If no mapping found, leave NaN
            continue

        # Fill all year columns with the SSP column
        for year in year_cols:
            # If year exists in population_df, otherwise leave NaN
            # Population_df has years like 2005,2010,... so we can interpolate if needed
            pop_row = population_df[(population_df["dummy.1"] == scaf_region) & (population_df["dummy"] == int(year))]

            filled_df.at[idx, year] = pop_row[ssp_column].values

    return filled_df
