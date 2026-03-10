import pandas as pd
import itertools
import numpy as np

def create_sector_template(template_df, sectors_df):
    """
    Create a template DataFrame similar to population template, 
    but with an additional 'Sector' column and all combinations.
    
    Parameters:
    - template_df : DataFrame
        Original template with Model, Scenario, Region, Variable, Unit, Year columns
    - sectors_csv_path : str
        Path to CSV containing column 'sectors'
    
    Returns:
    - new_df : DataFrame
    return new_df
        Template with all combinations of Model x Scenario x Region x Sector,
        Variable and Unit preserved, Year columns empty (NaN)
    """
    
    # 1. Identify year columns
    year_cols = [col for col in template_df.columns if str(col).isdigit()]
    
    # 2. Read sectors CSV
    sectors_list = sectors_df['sector_SCAF'].tolist()
    
    # 3. Get unique Model, Scenario, Region, Variable, Unit combinations from template
    unique_rows = template_df[['Model', 'Scenario', 'Region']].drop_duplicates()
    
    # 4. Create all combinations with sectors
    new_rows = []
    for _, row in unique_rows.iterrows():
        for sector in sectors_list:
            new_row = {
                "Model": row["Model"],
                "Scenario": row["Scenario"],
                "Region": row["Region"],
                "Variable": "Productivity growth rate",
                "Unit": "",
                "Sector": sector
            }
            # Year columns empty
            for year in year_cols:
                new_row[year] = np.nan
            new_rows.append(new_row)
    
    # 5. Create DataFrame and order columns
    ordered_cols = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Sector'] + year_cols
    new_df = pd.DataFrame(new_rows)[ordered_cols]
    
    return new_df

def cumulative_growth_rate(annual_growth_factor, delta_years):
    """
    Calculate cumulative growth rate given an annual growth factor.

    Parameters:
    - annual_growth_factor : float or array-like
        Annual productivity growth factor (e.g., 1.02 = +2% per year)
    - delta_years : int or array-like
        Number of years since the calibration year

    Returns:
    - cumulative_growth : float or array-like
        Cumulative growth rate relative to the calibration year
        (as fraction, e.g., 0.10 = +10%)
    """
    return (annual_growth_factor ** delta_years) - 1



def fill_sector_productivity(template_df, productivities_df, map_df, mapping_regions_df, calibration_year=None):
    """
    Fill the sector template with productivity growth rates.

    Parameters:
    - template_df : DataFrame
        Template with columns Model, Scenario, Region, Variable, Unit, Sector, year columns
    - productivities_df : DataFrame
        Columns: r=region, t=year, agr/man/ser
    - map_df : DataFrame
        Mapping of sector_SCAF -> sector_prod (agr/man/ser)
    - mapping_regions_df : DataFrame
        Mapping of region codes to NGFS names
    - calibration_year : int, optional
        If None, takes the minimum year in template columns

    Returns:
    - filled_df : DataFrame
    """

    filled_df = template_df.copy()

    # Identify year columns
    year_cols = [col for col in filled_df.columns if str(col).isdigit()]
    calibration_year = year_cols[0]

    
    # Build reverse mapping: NGFS region -> SCAF region
    reverse_region_map = {v: k for k, v in zip(mapping_regions_df["region_SCAF"], mapping_regions_df["region_NGFS"])}
    
        # Fill productivity
    for idx, row in filled_df.iterrows():
        ngfs_region = row["Region"]
        sector_name = row["Sector"]

        # Map NGFS region to SCAF
        scaf_region = reverse_region_map.get(ngfs_region)


        # Map sector to sector_prod
        sector_prod_col = map_df.loc[map_df["sector_SCAF"] == sector_name, "sector_prod"].values[0]
        if pd.isna(sector_prod_col):
            # No productivity mapping → fill zeros
            for year in year_cols:
                filled_df.at[idx, year] = 0.0
        else:
            # Filter productivity data for this region and year 2015 to 2050
            region_prod = productivities_df[(productivities_df["r"] == scaf_region) & (productivities_df["t"] == 2015)]

            growth_factor = region_prod[sector_prod_col].values[0]

            # Compute cumulative growth for each template year
            for year in year_cols:
                delta_years = int(year) - int(calibration_year)
                filled_df.at[idx, year] = cumulative_growth_rate(growth_factor, delta_years)

    return filled_df


def reformat_df(df):

    final_df=df.copy()
    # 2. Appendi il contenuto di "Energy consumers" a "Variable" separato da "|"
    final_df['Variable'] = final_df['Variable'] + "|" + final_df['Sector']
    
    # 3. Elimina la colonna "Energy consumers"
    final_df = final_df.drop(columns=['Sector'])
    return final_df