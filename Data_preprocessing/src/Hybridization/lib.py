import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections.abc import Iterable
import json
from pathlib import Path
from scipy.optimize import minimize
import itertools
from math import isclose

def filter_REMIND(REMIND_df, output_path):
    # Drop trailing NaN-named column produced by the trailing ';' in .mif files
    REMIND = REMIND_df.loc[:, REMIND_df.columns.notna()]

    # Drop World aggregate rows
    REMIND = REMIND[REMIND["Region"] != "World"]

    # Drop year columns before 2020
    year_cols_to_drop = [c for c in REMIND.columns if str(c).isdigit() and int(c) < 2020]
    REMIND = REMIND.drop(columns=year_cols_to_drop)

    REMIND.to_csv(output_path, index=False)
    return REMIND

#################################################################
####### AUGMENT THE REMIND DATAFRAME WITH SUPPLEMENTARY DATA ######
#################################################################

def augment_REMIND(REMIND_filtered, sup_data, disaggregation_data=None):

    def check_nomenclature_compatibility(REMIND_filtered,sup_data):
            # --- Check unique values consistency ---
        key_cols = ["Model", "Region", "Scenario"]
        for col in key_cols:
            filtered_unique = set(REMIND_filtered[col].unique())
            sup_unique = set(sup_data[col].unique())

            if filtered_unique != sup_unique:
                missing_in_sup = filtered_unique - sup_unique
                missing_in_filtered = sup_unique - filtered_unique

                print(f"⚠️ Warning: mismatch in column '{col}':")
                if missing_in_sup:
                    print(f"  Present in REMIND_filtered but missing in supplementary_data: {missing_in_sup}")
                if missing_in_filtered:
                    print(f"  Present in supplementary_data but missing in REMIND_filtered: {missing_in_filtered}")
            else:
                pass

    def check_duplicate_rows(REMIND_filtered,sup_data):
        merge_keys = ["Model", "Scenario", "Region", "Variable"]
        common_rows = REMIND_filtered.merge(
            sup_data[merge_keys], on=merge_keys, how="inner"
        )

        if not common_rows.empty:
            print(
                f"⚠️ Warning: {len(common_rows)} overlapping rows found "
                f"based on {merge_keys}. These may be duplicated entries."
            )
            print("Example of overlaps:")
            print(common_rows.head())
        else:
            pass


    def add_supplementary_data(REMIND_filtered, sup_data):

        sup_data.columns = sup_data.columns.map(str)
        sup_data=sup_data[sup_data["Region"] != "World"]

        
        check_nomenclature_compatibility(REMIND_filtered,sup_data)
        check_duplicate_rows(REMIND_filtered,sup_data)

        REMIND_augmented = pd.concat([REMIND_filtered,sup_data], join="inner")

        return REMIND_augmented
    
    def disaggregate_volumes(REMIND_augmented, disaggregation_data):
        #to implement in case we get the composition of LDV energy
        pass

    #augment data
    REMIND_augmented = add_supplementary_data(REMIND_filtered, sup_data)
    return REMIND_augmented

#####################################################
####### GENERATE ENERGY CONSUMPTION DATAFRAMES ######
#####################################################
def extract_regional_energy_sales_taxes(dfs: dict[str, pd.DataFrame],
                       mapping_df: pd.DataFrame,
                       block_name: str) -> dict[str, pd.Series]:
    """
    Applies extract_block_and_sum() to each DataFrame in a dictionary.

    Args:
        dfs (dict[str, pd.DataFrame]): Dictionary of DataFrames to process.
        mapping_df (pd.DataFrame): Mapping definition DataFrame.
        block_name (str): Block name to extract.

    Returns:
        dict[str, pd.Series]: Dictionary of Series, same keys as input.
    """
    results = {}

    for key, df in dfs.items():
        try:
            series = extract_sales_taxes(df, mapping_df, block_name)
            results[key] = series
        except KeyError as e:
            print(f" Warning: {key} skipped — missing rows/cols for block '{block_name}': {e}")
        except Exception as e:
            print(f" Error processing {key}: {e}")

    return results

def extract_sales_taxes(df: pd.DataFrame, mapping_df: pd.DataFrame, block_name: str) -> pd.Series:
    """
    Extracts the energy sales taxes matrix block from the IOT
    using a mapping of the IOT row and column indexes corresponding to the energy sales taxes
    including the respective consumer column.
    Groups by 'consumer' to aggregate taxes values (e.g. sectors or households).

    Args:
        df (pd.DataFrame): The IOT with MultiIndex rows and columns.
        mapping_df (pd.DataFrame): The mapping definition with columns:
            ['block_name', 'type', 'level1', 'level2', 'consumer'].
        block_name (str): The block name to extract (value in 'block_name' column ("Tax block")).

    Returns:
        pd.Series: Aggregated Series with 'consumer' as index and sum as values.
    """

    # Filter mapping for the chosen block
    map_block = mapping_df[mapping_df["block_name"] == block_name]

    # Extract row and column definitions
    row_data = map_block[map_block["type"] == "row"][["level1", "level2", "consumer"]]
    col_tuples = list(
        map_block[map_block["type"] == "col"][["level1", "level2"]].itertuples(index=False, name=None)
    )

    # Extract subset of df
    row_tuples = list(row_data[["level1", "level2"]].itertuples(index=False, name=None))
    sub_df = df.loc[row_tuples, col_tuples]

    # Name index levels
    sub_df.index = pd.MultiIndex.from_tuples(sub_df.index, names=["Category", "Subcategory"])

    # Build mapping (row tuple → consumer)
    consumer_map = {(r.level1, r.level2): r.consumer for r in row_data.itertuples(index=False)}

    # Add consumer column
    sub_df = sub_df.copy()
    sub_df["consumer"] = sub_df.index.map(consumer_map)

    # Group and sum by consumer
    summed = sub_df.groupby("consumer").sum(numeric_only=True).squeeze()

    return summed




def aggregate_IOT_energy_consumption(iot_data, col_label_df, row_label_df):
    """
    Aggregates regional Input-Output Table (IOT) energy consumption data 
    using preloaded mapping and regional data.

    This function takes as input:
        - preloaded IOT DataFrames for multiple regions,
        - column and row mapping DataFrames,
        - and a region renaming mapping,
    and produces standardized energy consumption values per region, 
    aggregated by SCAF_name.

    Parameters
    ----------
    iot_dataiot_data : dict[str, pd.DataFrame]
        Dictionary where each key is a region name (e.g., "REMIND_AFR") 
        and each value is the corresponding IOT DataFrame.
        Each DataFrame is expected to have a two-level MultiIndex 
        for both rows and columns (Category, Subcategory).

    col_label_df : pd.DataFrame
        Mapping between IOT column labels and standardized consumer names.
        Expected columns:
            - "Category"
            - "Subcategory"
            - "SCAF" (standardized consumer name)

    row_label_df : pd.DataFrame
        Mapping between IOT row labels and their type.
        Expected columns:
            - "Category"
            - "Subcategory"
            - "SCAF"
        Rows where SCAF == "ENERGY" are used for aggregation.

    regions_mapping : pd.DataFrame
        Mapping between SCAF and REMIND region names.
        Expected columns:
            - "region_SCAF"
            - "region_REMIND"

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary where:
            - keys =  region identifiers 
            - values = single-row DataFrames containing aggregated 
              energy consumption per SCAF consumer.
    """

    energy_row_label = get_IOT_row_labels_by_identifier(row_label_df, "ENERGY")


    # Build dictionary of arrays
    energy_dict = {}

    for region, df in iot_data.items():
        # --- 1. Filter rows corresponding to ENERGY ---
        df_energy = df.loc[df.index.isin(energy_row_label)]

        # Sum vertically and assign label 'ENERGY'
        df_energy = pd.DataFrame(df_energy.sum(axis=0)).T
        df_energy.index = ["ENERGY"]

        # --- 2. Map columns to SCAF_name and group ---
        # Convert columns MultiIndex to DataFrame for augmenting it with SCAF consumers column
        IOT_cols = pd.DataFrame(df_energy.columns.tolist(), columns=["Category", "Subcategory"])
        augmented_IOT_cols = IOT_cols.merge(col_label_df, on=["Category", "Subcategory"], how="left")

        # Keep only mapped columns
        valid_mask = augmented_IOT_cols["SCAF"].notna().values
        df_energy = df_energy.loc[:, valid_mask]
        filtered_IOT_cols = augmented_IOT_cols.loc[valid_mask]

        # Group by SCAF_name and sum
        df_energy.columns = filtered_IOT_cols["SCAF"].values
        df_energy = df_energy.groupby(df_energy.columns, axis=1, sort=False).sum()

        # --- 3. Convert to array and store ---
        energy_dict[region] = df_energy

    return energy_dict
    
def convert_to_net_values(dfs_consumption: dict[str, pd.DataFrame],
                              series_dict: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    """
    Subtracts the energy sales taxes (series_dict) from the corresponding
    energy consumption (dfs_consumption). If the tax is missing it returns 
    the original conusmption.

    Args:
        dfs_consumption (dict[str, pd.DataFrame]): Original energy
        consumption tables (gross of taxes).
        series_dict (dict[str, pd.Series]): Results from 
        extract_regional_energy_sales_taxes().

    Returns:
        dict[str, pd.DataFrame]: energy consumption df net of taxes.
    """

    adjusted = {}

    for key in dfs_consumption:
        df = dfs_consumption[key].copy()
        ser = series_dict.get(key)

        # Keep only columns that match consumers in the Series index
        common_cols = df.columns.intersection(ser.index)

        # Vectorized subtraction on the selected columns
        df[common_cols] = df[common_cols].sub(ser[common_cols], axis=1)

        adjusted[key] = df


    return adjusted

#####################################################
############## AGGREGATE ENERGY USES ###############
#####################################################

def get_REMIND_units(augmented_REMIND: pd.DataFrame, map_REMIND_energy_uses: pd.DataFrame) -> tuple:
    """
    Infer volume and price units from the REMIND data by looking up
    the Unit column for each variable listed in the mapping.

    Raises ValueError if variables within the same type have inconsistent units.

    Returns
    -------
    tuple[str, str]
        (volume_unit, price_unit)
    """
    var_unit = (
        augmented_REMIND[["Variable", "Unit"]]
        .drop_duplicates(subset=["Variable"])
        .set_index("Variable")["Unit"]
    )

    def collect_units(variables, label):
        units = set()
        for var in variables.dropna().unique():
            if var not in var_unit.index:
                raise ValueError(f"Variable '{var}' ({label}) not found in augmented_REMIND")
            units.add(var_unit[var])
        if len(units) != 1:
            raise ValueError(f"{label} variables have inconsistent units: {units}")
        return units.pop()

    volume_unit = collect_units(map_REMIND_energy_uses["REMIND_volume"], "REMIND_volume")
    price_unit  = collect_units(map_REMIND_energy_uses["REMIND_price"],  "REMIND_price")

    return volume_unit, price_unit


def aggregate_energy_uses(remind, mapping, value_unit, price_unit):

    def create_energy_uses_volumes(remind, mapping):
        """
        Create the Energy_uses_volumes dataset.

        Steps:
        1. Keep only REMIND rows whose 'Variable' appears in mapping['REMIND_volume'].
        2. Add the corresponding 'energy_use' column from mapping.
        3. Group by Model, Scenario, Region, energy_use, and Unit.
        4. Sum all year columns.
        5. Return a clean DataFrame with aggregated results.
        """
        # Identify year columns (4-digit strings like "2020", "2025", etc.)
        year_cols = [c for c in remind.columns if str(c).isdigit()]
        
        # Filter REMIND to keep only rows mapped in REMIND_volume
        filtered_remind = remind[remind['Variable'].isin(mapping['REMIND_volume'])].copy()
        
        # Merge with mapping to bring in the corresponding energy_use
        merged_df = pd.merge(
            filtered_remind,
            mapping[['REMIND_volume', 'energy_use']],
            left_on='Variable',
            right_on='REMIND_volume',
            how='left'
        )
        
        # Group by key identifiers and sum all year columns
        grouped_df = (
            merged_df
            .groupby(['Model', 'Scenario', 'Region', 'energy_use', 'Unit'])[year_cols]
            .sum()
            .reset_index()
        )

        # Ensure column order is consistent
        grouped_df = grouped_df[['Model', 'Scenario', 'Region', 'energy_use', 'Unit'] + year_cols]
        
        return grouped_df


    def create_energy_use_values(remind, mapping, value_unit):
        """
        Create the energy_use_values dataset in two steps:
        
        1. Build a label matrix that contains every combination of:
        (Model, Scenario, Region) × (REMIND_volume, REMIND_price, energy_use)
        - No merges, only explicit cartesian product.
        - Preserves row order from both sources.

        2. Add yearly columns (e.g., 2020, 2025, ...) computed as:
            REMIND[Variable == REMIND_volume] * REMIND[Variable == REMIND_price]
        for matching Model, Scenario, Region.
        - Fully vectorized, no loops.
        """
        # Identify year columns (those that contain only digits)
        year_cols = [c for c in remind.columns if str(c).isdigit()]

        # 1. Build the label dataset (explicit cartesian product)
        unique_combos = remind[['Model', 'Scenario', 'Region']].drop_duplicates().reset_index(drop=True)
        unique_combos = unique_combos[unique_combos["Region"] != "World"]

        n_unique = len(unique_combos)
        n_mapping = len(mapping)

        # Repeat each dataset to create the full cartesian product
        extended_combos = unique_combos.loc[np.repeat(unique_combos.index.values, n_mapping)].reset_index(drop=True)
        extended_mapping = pd.concat([mapping] * n_unique, ignore_index=True)

        # Combine horizontally
        labels_df = pd.concat(
            [extended_combos, extended_mapping],
            axis=1
        )
        
        volume_labels = labels_df.drop(['REMIND_price', 'energy_use'], axis=1).rename(columns={'REMIND_volume': 'Variable'})
        price_labels = labels_df.drop(['REMIND_volume', 'energy_use'], axis=1).rename(columns={'REMIND_price': 'Variable'})

        multi_index_remind = remind.set_index(['Model', 'Scenario', 'Region', 'Variable'])

        keys_vol = list(zip(volume_labels['Model'], volume_labels['Scenario'], volume_labels['Region'], volume_labels['Variable']))
        keys_price = list(zip(price_labels['Model'], price_labels['Scenario'], price_labels['Region'], price_labels['Variable']))

        vol_values = multi_index_remind.loc[keys_vol, year_cols]
        price_values = multi_index_remind.reindex(keys_price)[year_cols].to_numpy()

        year_matrix = np.nan_to_num(vol_values) * np.nan_to_num(price_values)

        result = labels_df.copy()
        result[year_cols] = year_matrix

        # 3. Aggregate by energy_use, summing across mappings that share the same energy_use
        grouped = (
            result.groupby(['Model', 'Scenario', 'Region', 'energy_use'])[year_cols]
            .sum()
            .reset_index()
        )
        # Add Unit
        col_index = grouped.columns.get_loc("energy_use") + 1  # posizione dopo "energy_use"
        grouped.insert(col_index, "Unit", value_unit)

        return grouped

    def create_energy_use_prices(values, volumes, price_unit):
        keys = ["Model", "Scenario", "Region", "energy_use"]

        values_ordered = values.set_index(keys)
        volumes = volumes.set_index(keys)
        volumes_ordered = volumes.loc[values_ordered.index]

        if not values_ordered.index.equals(volumes_ordered.index):
            raise ValueError("❌ The labels in the volumes dataframe do not equal the labels in the values dataframe. Impossible to compute REMIND prices")

        year_cols = [c for c in values_ordered.columns if str(c).isdigit()]

        ratio=pd.DataFrame(index=values_ordered.index, columns=(["Unit"] + year_cols))
        ratio["Unit"] = price_unit
        ratio[year_cols] = values_ordered[year_cols].astype(float) / volumes_ordered[year_cols].astype(float)

        return ratio.reset_index()

    volumes = create_energy_uses_volumes(remind, mapping)
    values = create_energy_use_values(remind, mapping, value_unit)
    prices = create_energy_use_prices(values, volumes, price_unit)

    markets_dict={"prices":prices, "values":values, "volumes":volumes}

    return markets_dict

###################################################################
############ CREATE ENERGY ALLOCATION OUTPUT DATABASE #############
###################################################################

def generate_output_template(REMIND,energy_consumers,energy_uses, price_unit, volume_unit):

    models=np.unique(REMIND["Model"])
    scenarios=np.unique(REMIND["Scenario"])
    regions=np.unique(REMIND["Region"])
    variable_type=["Volume", "Price"]
    combinations = list(itertools.product(models, scenarios, regions, variable_type, energy_consumers,energy_uses))

    year_cols = [c for c in REMIND.columns if str(c).isdigit()]

    # Creazione DataFrame
    df_output = pd.DataFrame(combinations, columns=['Model', 'Scenario', 'Region', "Variable","Energy consumers", "Energy uses"])
    df_output["Unit"] = np.nan
    df_output.loc[df_output["Variable"] == "Volume", "Unit"] = volume_unit
    df_output.loc[df_output["Variable"] == "Price", "Unit"] = price_unit

    df_output[year_cols] = np.nan
    

    return df_output


def fill_calibration_year(
    df_template: pd.DataFrame,
    allocation_matrix: pd.DataFrame,
    model: str,
    scenario: str,
    region: str,
    variable: str,
    first_year: str
) -> pd.DataFrame:
    """
    Assigns values from an allocation matrix (energy consumers × energy uses)
    to the corresponding rows in the template for a given Model, Scenario, Region, and Variable.
    When a value is close to zero it sets it to zero.

    Parameters
    ----------
    df_template : pd.DataFrame
        Template generated by `generate_output_template`.
    allocation_matrix : pd.DataFrame
        Matrix with rows = energy consumers, columns = energy uses, containing numeric values.
    model : str
        Model name to filter.
    scenario : str
        Scenario name to filter.
    region : str
        Region name to filter.
    variable : str
        Variable name (typically "Volume" or "Price").

    Returns
    -------
    pd.DataFrame
        The updated template DataFrame with assigned values for the specified combination.
    """

    # Work on a copy to avoid modifying the original DataFrame directly
    df = df_template.copy()

    # Identify the first year column (lowest numeric column name)
    year_cols = [c for c in df.columns if str(c).isdigit()]
    if not year_cols:
        raise ValueError("No numeric year columns found in the template.")
    first_year = str(min(map(int, year_cols)))

    # Create a mask for rows matching the given model, scenario, region, and variable
    mask = (
        (df["Model"] == model)
        & (df["Scenario"] == scenario)
        & (df["Region"] == region)
        & (df["Variable"] == variable)
    )

    # Loop over all pairs of (energy consumer, energy use)
    for consumer in allocation_matrix.index:
        for use in allocation_matrix.columns:
            value = allocation_matrix.loc[consumer, use]
            if isclose(value, 0, abs_tol = 1e-10):
                value = 0

            # Assign the value from the matrix to the matching cell in the first year column
            df.loc[
                mask
                & (df["Energy consumers"] == consumer)
                & (df["Energy uses"] == use),
                first_year
            ] = value

    return df


#####################################################
############ ALLOCATE ENERGY TO SECTORS #############
#####################################################
class Energy_Values_Allocation:

    def __init__(self,
        IOT_E_consumptions,
        REMIND_E_uses,
        REMIND_E_prices,
        priorities,
        key):
        """
        Initializes the energy allocation model linking Input-Output consumers with energy uses.

        Args:
            IOT_E_consumptions (pd.Series): 
                Series containing total energy consumption by consumer. 
                The index must represent consumer names.
                
            REMIND_E_uses (pd.Series): 
                Series containing the energy uses in value from REMIND data. 
                The index must represent energy use categories.

            REMIND_E_prices (pd.Series):
                Series containing the price per energy use from REMIND data. 
                The index must represent energy use categories.

            priorities (pd.DataFrame): 
                DataFrame defining which energy uses are available for which consumers. 
                Must contain the following columns:
                    - 'energy_use' (str): the energy use category (e.g., "R&C", "LDV").
                    - 'consumer' (str): the consumer name.
                    - 'availability' (float): a value between 0 and 1 indicating 
                    how much of the given energy use is available to the specified consumer.
                
                Example:
                    | energy_use | consumer         | availability |
                    |-------------|----------------|---------------|
                    | R&C         | HOUSEHOLDS     | 0.6666667     |
                    | LDV         | HOUSEHOLDS     | 1.0           |
                    | PAS         | TRANSPORTATION | 1.0           |


            key (pd.Series or pd.DataFrame): 
                Disaggregation key used to allocate energy uses to consumers.
                - If a Series: its index must match consumer names.
                - If a DataFrame: its index must match consumer names and its columns 
                must match energy use categories (same as `REMIND_E_uses.index`).
                
                This key determines the relative weight of each consumer (or consumer–energy pair)
                in the allocation process.

        Raises:
            ValueError: If `key` is not a pandas Series or DataFrame.
        """
           
        self.IOT_E_consumptions =IOT_E_consumptions.copy()
        self.REMIND_E_uses =REMIND_E_uses.copy()
        self.REMIND_E_prices=REMIND_E_prices.copy()
        self.priorities = priorities.copy()

        self.consumers = list(self.IOT_E_consumptions.index).copy()
        self.energy_uses = list(self.REMIND_E_uses.index).copy()

        empty_consumerXuses_matrix = pd.DataFrame(index=self.consumers, columns=self.energy_uses)
        
        #calibration matrices

        self.volume_matrix = empty_consumerXuses_matrix.copy()
        self.prices_matrix = empty_consumerXuses_matrix.copy()
        self.values_matrix = empty_consumerXuses_matrix.copy()

        self.scaling_factor=False

        #forced values

        self.forced_energy_values = empty_consumerXuses_matrix.copy()

        self.E_to_allocate = REMIND_E_uses.copy()


        if isinstance(key, pd.DataFrame) or isinstance(key, pd.Series):
            self.key_df = self.__set_key_df(key)
        else: 
            raise ValueError("Key must be a Series or a DataFrame")

    #####################################
    ########## PRIVATE METHODS ##########
    #####################################

    def __apply_key_to_all_energy_uses(self, key):
        """
        Convert a single N-dimensional key vector into an NxM DataFrame,
        replicating the same key for all columns.
        Handles Series indexed differently from self.consumers.
        """
        M = len(self.energy_uses)

        # Reindex to match self.IOT_E_consumptions
        key_aligned = key.reindex(self.consumers).astype(float)

        # Replicate for all columns
        key_matrix = np.tile(key_aligned.values.reshape(-1, 1), M)

        # Store as DataFrame
        key_df = pd.DataFrame(key_matrix,
                                index=self.consumers,
                                columns=self.energy_uses)

        return key_df

    def __normalize_key_df(self, key_df):
        """
        Normalize a key DataFrame column-wise so that each column sums to 1.
        Keeps relative proportions intact.
        
        Args:
            key_df (pd.DataFrame): input DataFrame with shape (N, M)
        
        Returns:
            pd.DataFrame: normalized DataFrame
        """
        normalized_df = key_df.copy()
        
        for col in normalized_df.columns:
            col_sum = normalized_df[col].sum()
            if col_sum == 0:
                raise ValueError(f"The sum of column '{col}' is zero, cannot normalize.")
            normalized_df[col] /= col_sum
        
        return normalized_df

    def __set_key_df(self, key):
        """
        Assign self.key_df from key input.
        - If key is a matrix/ndarray: convert to DataFrame using self.consumers and self.energy_uses for index/columns.
        - If key is a DataFrame: reorder rows/columns according to self.consumers and self.energy_uses.
        """
        # If key is a DataFrame order it
        if isinstance(key, pd.DataFrame):
            # Reorder rows and columns
            if key.shape != (len(self.consumers), len(self.energy_uses)):
                raise ValueError(f"Key array shape {key.shape} does not match (N={len(self.consumers)}, M={len(self.energy_uses)})")
            key_df = key.reindex(index=self.consumers, columns=self.energy_uses)

        # If key is a Series turn it to a df
        elif isinstance(key, pd.Series):
            if len(key) != len(self.consumers):
                raise TypeError("key must be the same length as the number of consumers")
            else:
                key_df= self.__apply_key_to_all_energy_uses(key)

        _normalized_key_df = self.__normalize_key_df(key_df)
        return _normalized_key_df

    def __fill_nans_with_zero_in_column(self, energy_use):
        """Replace NaN values with 0 in the given energy_use column."""

        self.forced_energy_values[energy_use] = (self.forced_energy_values[energy_use].astype(float)
                                                .fillna(0)
                                                )
        
    def __fill_nans_with_zero_in_row(self, consumer):
        """Replace NaN values with 0 in the given consumer row."""

        self.forced_energy_values.loc[consumer] = (
            self.forced_energy_values.loc[consumer]
            .astype(float)
            .fillna(0)
        )

    def __update_forced_energy_values(self, value, energy_use, consumer):
        self.forced_energy_values.loc[consumer, energy_use] = value
        self.E_to_allocate[energy_use] -= value

    def __check_energy_balance(self, energy_use):
        """Verify that the total allocated energy for a given type matches the expected REMIND_E_uses value.
        Returns True if consistent, False otherwise.
        """
        allocated_sum = self.forced_energy_values[energy_use].sum()
        expected = self.REMIND_E_uses[energy_use]
        if not np.isclose(allocated_sum, expected, rtol=1e-6):
            raise ValueError("For"+ energy_use +", allocated sum ≠ expected ")
    
    def __set_forced_energy_percentages(self):
        self.forced_energy_percentages = self.forced_energy_values.div(self.REMIND_E_uses)
    
    def adjust_rows_to_target(self):
        """
        Adjust each row of self.values_matrix so that its sum matches the corresponding
        target in self.IOT_E_consumptions. The adjustment is proportional to the existing
        values in the row. Works even if the indices of IOT_E_consumptions are in a
        different order than the rows of values_matrix.

        Returns
        -------
        pd.DataFrame
            Adjusted DataFrame with row sums matching row targets.
        """


        D = self.values_matrix.astype(float).copy()

        # Current row sums
        row_sums = D.sum(axis=1).values  # shape (N,)

        #same order as the rows in the df
        row_targets_aligned = self.IOT_E_consumptions.reindex(self.values_matrix.index).values

        # Row errors
        row_errors = row_targets_aligned - row_sums  # shape (N,)

        # Row proportions
        proportions = np.divide(D.values, row_sums[:, None],
                                out=np.zeros_like(D.values),
                                where=row_sums[:, None] != 0)

        # Apply adjustment
        D.values[:] = D.values + proportions * row_errors[:, None]

        return D
    
    #####################################
    ########## PUBLIC METHODS ###########
    #####################################


    def rescale_REMIND_energy_values(self):

        """Rescales the REMIND energy uses values in order for the total energy value (for one region)
        to match with the total IOT energy consumption
        """        
        self.scaling_factor = self.IOT_E_consumptions.sum()/self.REMIND_E_uses.sum()

        self.REMIND_E_uses *= self.scaling_factor
        self.E_to_allocate*= self.scaling_factor

        return self.scaling_factor
    

    def allocate_forced_energy_values(self):    
        """
        Build the fixed_energy (NxM) matrix with 'forced' energy allocations based on priorities and availabilities.
        """
        fill_order=pd.unique(self.priorities["energy_use"])
        
        for energy_use in fill_order:
            if self.E_to_allocate[energy_use] <= 0:
                continue  # nothing left to allocate

            priority_consumers = self.priorities.loc[ self.priorities['energy_use'] == energy_use, "consumer"]

            for consumer in priority_consumers:
                availability_for_consumer = self.priorities.loc[ (self.priorities['energy_use'] == energy_use) & 
                                                          (self.priorities['consumer'] == consumer),
                                                          "availability"].item()
                
                availability_target= availability_for_consumer * self.REMIND_E_uses[energy_use]

                energy_availability = min(self.E_to_allocate[energy_use],availability_target)
                
                consumer_capacity = self.IOT_E_consumptions[consumer] - np.nansum(self.forced_energy_values.loc[consumer])
                
                if consumer_capacity <= 0:
                    continue

                if consumer_capacity >= energy_availability:
                    self.__update_forced_energy_values(energy_availability, energy_use, consumer)

                    if self.E_to_allocate[energy_use] == 0:
                        self.__fill_nans_with_zero_in_column(energy_use)
                        self.__check_energy_balance(energy_use)
                        break

                    if consumer_capacity == energy_availability:
                        self.__fill_nans_with_zero_in_row(consumer)
                
                else:
                    self.__update_forced_energy_values(consumer_capacity, energy_use, consumer)
                    self.__fill_nans_with_zero_in_row(consumer)

        self.__set_forced_energy_percentages()
    
    def adjust_key_for_forced_values(self):
        """
        Adjust self.key_df accounting for self.forced_energy_percentages:
        - For each energy type, the forced energy consumptions are detected and forced in self.key_df.
        - Remaining percentages are distributed proportionally to key_df and in such a way that the disaggregation key percentges sum to 1.
        - Result overwrites self.key_df.
        """
        # Make a copy to modify
        adjusted_key = self.key_df.copy()

        # Non forced fraction of energy per energy useIOT_E_consumptions
        non_forced_fraction = 1 - self.forced_energy_percentages.sum(axis=0, skipna=True)

        # Iterate over each energy use (column) by name
        for energy_use in self.energy_uses:
            col = self.forced_energy_percentages[energy_use]
            surplus_fraction = non_forced_fraction[energy_use]

            # Mask for entries to distribute (NaN = non_forced)
            non_forced_mask = col.isna()

            # Proportions of key_df for the non_forced entries
            key_non_forced = adjusted_key.loc[non_forced_mask, energy_use]
            key_non_forced_sum = key_non_forced.sum()

            if key_non_forced_sum > 0:
                # scale the non_forced entries proportionally
                adjusted_key.loc[non_forced_mask, energy_use] = key_non_forced / key_non_forced_sum * surplus_fraction
            else:
                # if all remaining key entries are zero, leave zeros (or could distribute uniformly)
                adjusted_key.loc[non_forced_mask, energy_use] = 0

            # assign fixed values directly
            adjusted_key.loc[~non_forced_mask, energy_use] = col[~non_forced_mask]

        # Store adjusted key
        self.key_df = adjusted_key
        return self.key_df
    
    

    def compute_disaggregated_energy(self):
        """
        Fill the distribution matrix (NxM) minimizing the distance from the adjusted key,
        while respecting fixed elements and balance constraints.
        Returns:
            A pd.DataFrame with indexes equal to the consumers and columns equal to the energy uses. 
            Every element corresponds to the consumption of a consumer of a specific energy use, in value.
        """

        # Retrieve adjusted key, fixed mask, and fixed percentages
        key_df = self.key_df.copy()
        forced_mask = ~self.forced_energy_percentages.isna()
        forced_percentages = self.forced_energy_percentages.values

        # Initial guess: start from adjusted key but apply fixed values
        D0 = key_df.values.copy()
        D0[forced_mask] = forced_percentages[forced_mask]

        # Flatten the free variables
        x0 = D0[~forced_mask]

        def reconstruct_D(x):
            """Rebuild full matrix from x and fixed cells."""
            D = forced_percentages.copy()
            D[~forced_mask] = x
            return D


        def objective(x):
            """Quadratic distance from adjusted key."""
            D = reconstruct_D(x)

            return np.sum((D - key_df.values) ** 2)

        # Constraints vectorized

        # Identify rows and columns that are *fully fixed* (no NaNs)
        fully_fixed_columns = ~self.forced_energy_percentages.isna().any(axis=0)
        fully_fixed_rows = ~self.forced_energy_percentages.isna().any(axis=1)


        # Build masks for active constraints (only for partially free rows/columns)
        active_col_mask = ~fully_fixed_columns
        active_row_mask = ~fully_fixed_rows

        # Filtered constraint functions
        def active_column_sum_constraints(x):
            D = reconstruct_D(x)
            return D.sum(axis=0)[active_col_mask] - 1.0

        def active_consumer_balance_constraints(x):
            D = reconstruct_D(x)
            balance_error = (
                D.dot(self.REMIND_E_uses.values)[active_row_mask]
                - self.IOT_E_consumptions.values[active_row_mask]
            )
            return balance_error / self.IOT_E_consumptions.values[active_row_mask]


        def total_sq_violation(x):
            """Sum of squared violations for both sets of constraints."""
            col_violation = active_column_sum_constraints(x)
            row_violation = active_consumer_balance_constraints(x)
            return np.sum(col_violation**2) + np.sum(row_violation**2)

        constraints = [
            {'type': 'eq', 'fun': total_sq_violation}
        ]

        # Bounds for unfixed variables
        bounds = [(0.0, 1.0)] * len(x0)

        # Add noise 
        x0=abs(x0+np.random.normal(0,np.mean(x0)/20,x0.shape))
        # Solve optimization
        result = minimize(
            objective, x0, bounds=bounds, constraints=constraints, method='SLSQP', options={'maxiter': 10000}
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        # Reconstruct final matrix
        self.values_matrix = (pd.DataFrame(reconstruct_D(result.x), index=key_df.index, columns=key_df.columns))*self.REMIND_E_uses

        self.values_matrix=self.adjust_rows_to_target()

        # Return as DataFrame with proper labels
        return self.values_matrix


    def compute_prices_matrix(self):

        self.prices_matrix[:] = self.REMIND_E_prices[self.prices_matrix.columns].values
        self.prices_matrix*=self.scaling_factor
        return self.prices_matrix
    
    def compute_volumes_matrix(self):
        self.volumes_matrix = self.values_matrix/self.prices_matrix
        return self.volumes_matrix

##########################################
####### GENERATE KEY BASED ON VA #########
##########################################

def extract_row_until_nan(df, row_label):
    """
    Extracts values from a specific (Category, Subcategory) row of a 
    multi-indexed DataFrame until the first NaN column value is found.
    Uses the Subcategory level of the columns as the Series index.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex rows and MultiIndex columns.
    row_label : str
        Label for the first row index level (e.g. "VA").
    sub_label : str
        Label for the second row index level (e.g. "∑").

    Returns
    -------
    tuple[pd.Series, tuple | None]
        - pd.Series: values up to the first NaN, indexed by the column Subcategory level.
        - tuple | None: the column MultiIndex (Category, Subcategory) where NaN was first found,
          or None if the row has no NaNs.
    """

    # --- Step 1: extract row ---
    row = df.loc[row_label]

    # --- Step 2: find index of the first nan NaN ---
    notna_mask = row.notna()
    cutoff = np.argmax(~notna_mask.values)  # primo False

    # --- Step 3: cut until valid column ---
    valid_part = row.iloc[:,:cutoff]

    # --- Step 4: usa il livello 'Subcategory' del MultiIndex delle colonne ---
    if isinstance(valid_part.index, pd.MultiIndex):
        sub_index = valid_part.columns.get_level_values("Subcategory")
    else:
        raise("No multiindex column labels in the IOT")

    series = pd.Series(valid_part.values.flatten(), index=sub_index)
    
    return series


def get_IOT_row_labels_by_identifier(row_label_df, row_label_identifier):
    """
    Extracts (Category, Subcategory) tuples from a row mapping DataFrame
    based on a given label identifier (e.g. "ENERGY").

    Parameters
    ----------
    row_label_df : pd.DataFrame
        Mapping between IOT row labels and their types.
        Expected columns: [Category, Subcategory, LabelType]
    label_type : str
        The label type to filter for (e.g., "ENERGY", "AGRICULTURE").

    Returns
    -------
    list[tuple[str, str]]
        List of (Category, Subcategory) tuples matching the specified label type.
    """
    mask = row_label_df.iloc[:, 2] == row_label_identifier
    rows = row_label_df.loc[mask]
    first_col = rows.iloc[:, 0].tolist()
    second_col = rows.iloc[:, 1].tolist()
    return list(zip(first_col, second_col))


############################
###### MISCELLANEOUS #######
############################

def rename_regions(regions_mapping, regions_obj):
    """
    Rename region names in a dictionary or in a DataFrame using a mapping table.

    Parameters
    ----------
    regions_mapping : pd.DataFrame
        DataFrame with columns ["region_SCAF", "region_REMIND"].
    regions_obj : dict | pd.DataFrame
        Object to rename:
            - dict[str, Any]: keys are region names
            - DataFrame: one column contains region names (guessed automatically)

    Returns
    -------
    dict | pd.DataFrame
        Object with region names replaced according to mapping.
    """

    # Create mapping dictionary
    mapping = dict(zip(regions_mapping["region_SCAF"], regions_mapping["region_REMIND"]))

    # Case 1: dictionary
    if isinstance(regions_obj, dict):
        return {mapping.get(k, k): v for k, v in regions_obj.items()}

    # Case 2: DataFrame
    elif isinstance(regions_obj, pd.DataFrame):
        df = regions_obj.copy()

        # Try to guess the region column
        region_col = None
        for c in df.columns:
            if df[c].dtype == object and df[c].isin(mapping.keys()).any():
                region_col = c
                break

        if region_col is None:
            raise ValueError(
                "Could not find a region column matching mapping keys "
                "(expected one containing region_SCAF values)."
            )

        # Rename the values in that column
        df[region_col] = df[region_col].replace(mapping)

        return df

    else:
        raise TypeError("regions_obj must be either a dict or a pandas DataFrame")

##########################################
##### generate priorities from IEA #######
##########################################

def aggregate_by_consumer_and_use(IEA_data, IEA_mapping):
    """
    Aggregate energy data by (consumer, energy_use)
    using mapping information from IEA_name.

    Parameters
    ----------
    IEA_data : pd.DataFrame
        DataFrame containing energy data.
        Example columns: ['agglomeration', 'Unit', 'Residential', 'Agriculture&forestry', 'Fishing']
    IEA_mapping : pd.DataFrame
        DataFrame mapping 'IEA_name' -> ('consumer', 'energy_use').
        Example columns: ['IEA_name', 'consumer', 'energy_use']

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with MultiIndex columns (consumer, energy_use).
    """

    # Create a dictionary mapping from IEA_name to tuples (consumer, use)
    mapping = dict(zip(
        IEA_mapping["IEA_name"],
        zip(IEA_mapping["energy_consumer"], IEA_mapping["energy_use"])
    ))

    df = IEA_data.copy()

    # Find columns that exist in the mapping
    mapped_cols = [c for c in df.columns if c in mapping]

    # Extract only the mapped columns
    df_mapped = df[mapped_cols].copy()

    # Assign MultiIndex columns (consumer, energy_use)
    tuples = [mapping[c] for c in df_mapped.columns]
    df_mapped.columns = pd.MultiIndex.from_tuples(tuples, names=["consumer", "energy_use"])

    # Group by MultiIndex and sum duplicate columns
    df_grouped = df_mapped.groupby(level=[0, 1], axis=1).sum()

    # Add back the non-mapped columns (e.g., agglomeration, Unit)
    non_mapped_cols = [c for c in df.columns if c not in mapping]
    df_result = pd.concat([df[non_mapped_cols], df_grouped], axis=1)

    return df_result


##################################################################
############### ASSING PRIOROTIES WITH IEA DATA ##################
##################################################################


#it assumes the values are the same for all scenarios and models. 
def compute_availability_for_pair(volumes, aggregated_IEA, region, consumer, energy_use, calibration_year):
    """
    Compute the availability ratio for a specific region, consumer, and energy_use.

    Parameters
    ----------
    volumes : pd.DataFrame
        DataFrame containing energy volumes by scenario, region, and energy use.
        Must include columns: ["Scenario", "Region", "energy_use", <year_columns>].
    aggregated_IEA : pd.DataFrame
        DataFrame containing regional IEA data with MultiIndex columns
        (consumer, energy_use).
    region : str
        Region name to filter.
    consumer : str
        Consumer label (e.g., "HOUSEHOLD").
    energy_use : str
        Energy use label (e.g., "R&C").
    calibration_year : str or int
        Column name (year) used for calibration.

    Returns
    -------
    float
        The computed availability ratio: IEA_volume / energy_use_volume.
    """
    model = volumes["Model"][0]
    scenario = volumes["Scenario"][0]
    # Filter the `volumes` DataFrame for the given scenario, region, and energy use
    mask = (
        (volumes["Model"]== model)
        & (volumes["Scenario"] == scenario)
        & (volumes["Region"] == region)
        & (volumes["energy_use"] == energy_use)
    )

    # Extract the calibration year value (single numeric value)
    energy_use_volume = volumes.loc[mask, calibration_year].item()

    # Extract the corresponding IEA value for the same region and consumer/use pair
    IEA_volume = aggregated_IEA.loc[
        aggregated_IEA["agglomeration"] == region, [(consumer, energy_use)]
    ].values[0][0]

    # Compute the ratio
    return IEA_volume / energy_use_volume


def build_availabilities_df(volumes, aggregated_IEA, calibration_year):
    """
    Compute all availability ratios for all (region, consumer, energy_use) combinations.

    Raises an error if any combination cannot be found.

    Parameters
    ----------
    volumes : pd.DataFrame
        Energy data by scenario and region.
    aggregated_IEA : pd.DataFrame
        Aggregated IEA data with (consumer, energy_use) columns.
    IEA_mapping : pd.DataFrame
        Mapping between IEA_name, consumer, and energy_use.
    scenario : str
        Scenario name to use for filtering.
    calibration_year : str or int
        Year to use as the calibration column.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ["region", "consumer", "energy_use", "availability"].
    """

    # Create list of (consumer, energy_use) pairs from mapping
    pairs = [col for col in aggregated_IEA.columns if isinstance(col, tuple)]
    records = []

    # Loop through each region and each (consumer, use) pair
    for region in aggregated_IEA["agglomeration"]:
        for consumer, energy_use in pairs:
            # Compute availability ratio; will raise KeyError or ValueError if missing
            availability = compute_availability_for_pair(
                volumes, aggregated_IEA, region, consumer, energy_use, calibration_year
            )

            # Append the result
            records.append({
                "region": region,
                "consumer": consumer,
                "energy_use": energy_use,
                "availability": availability
            })

    # Combine all results into a single DataFrame
    return pd.DataFrame(records)



def build_priorities_dict(out_df: pd.DataFrame, priorities_df: pd.DataFrame) -> dict:
    """
    Build a dictionary of priorities per region, filling 'to_fill' entries
    from availability data.

    Parameters
    ----------
    out_df : pd.DataFrame
        DataFrame containing availability per region, consumer, and energy_use.
        Expected columns: ['region', 'consumer', 'energy_use', 'availability']
        'availability' must be numeric float.

    priorities_df : pd.DataFrame
        Template DataFrame with columns ['energy_use', 'consumer', 'availability'].
        Cells with 'to_fill' will be replaced.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary where keys are regions, values are DataFrames shaped like
        priorities_df, with 'to_fill' cells replaced by corresponding availability.
    """
    priorities_dict = {}

    # Get list of regions in out_df
    regions = out_df['region'].unique()

    for region in regions:
        # Copy the template for each region
        region_df = priorities_df.copy()

        # Find all rows that need filling
        mask_fill = region_df['availability'] == 'to_fill'

        # Iterate only over rows to fill
        for idx in region_df[mask_fill].index:
            consumer = region_df.at[idx, 'consumer']
            energy_use = region_df.at[idx, 'energy_use']

            # Extract availability from out_df
            availability_row = out_df[
                (out_df['region'] == region) &
                (out_df['consumer'] == consumer) &
                (out_df['energy_use'] == energy_use)
            ]

            if availability_row.empty:
                raise ValueError(
                    f"No availability found for region={region}, consumer={consumer}, energy_use={energy_use}"
                )

            region_df.at[idx, 'availability'] = float(availability_row['availability'].values[0])

        # Ensure availability column is float
        region_df['availability'] = region_df['availability'].astype(float)

        # Store in dict
        priorities_dict[region] = region_df

    return priorities_dict

###############################################
####### PROJECT VOLUMES AND PRICES ############
###############################################


def project_variables(output: pd.DataFrame, reference: pd.DataFrame, variable_type: str) -> pd.DataFrame:
    """
    Evolve a given variable type (exclusively 'Volume' or 'Price') in the 'output' DataFrame
    based on growth rates from a reference DataFrame.

    The function:
      - Selects rows in 'output' where Variable == variable_type
      - Matches (Model, Scenario, Region, Energy uses) with the reference DataFrame
      - Computes growth factors from reference (relative to 2020)
      - Applies those factors to evolve 2020 values for all later years

    Parameters
    ----------
    output : pd.DataFrame
        DataFrame containing base-year (2020) values and NaN for later years.
    reference : pd.DataFrame
        DataFrame containing the time evolution for the variable type (e.g. volumes or prices).
    variable_type : str
        The variable type to evolve ('Volume' or 'Price').

    Returns
    -------
    pd.DataFrame
        A copy of 'output' with evolved values for the specified variable type.
    """
    
    # Identify the year columns (e.g. "2020", "2025", "2030")
    year_cols = [col for col in output.columns if str(col).isdigit()]
    
    calibration_year = year_cols[0]
    # Copy to avoid modifying the original
    out = output.copy()
    
    # Ensure consistent naming between DataFrames
    reference = reference.rename(columns={'energy_use': 'Energy uses'})
    
    # Select only rows with the given variable type
    mask = out['Variable'] == variable_type

    # Iterate through each unique combination of keys
    for (model, scenario, region, energy_use) in out.loc[mask, ['Model', 'Scenario', 'Region', 'Energy uses']].drop_duplicates().itertuples(index=False):
        # Select matching row in reference
        ref_row = reference[
            (reference['Model'] == model) &
            (reference['Scenario'] == scenario) &
            (reference['Region'] == region) &
            (reference['Energy uses'] == energy_use)
        ]

        if ref_row.empty:
            continue  # Skip if no match (should not happen)

        # Compute growth factors relative to 2020
        base = ref_row.iloc[0][calibration_year]
        growth_factors = {year: ref_row.iloc[0][year] / base for year in year_cols}

        # Select corresponding rows in output (same keys)
        sel = (
            (out['Model'] == model) &
            (out['Scenario'] == scenario) &
            (out['Region'] == region) &
            (out['Energy uses'] == energy_use) &
            mask
        )

        # Apply growth rates to evolve each year's value
        base_values = out.loc[sel, calibration_year]
        # Apply growth rates in a fully vectorized way
        target_years = year_cols[1:]
        growth_series = pd.Series(growth_factors)
        #reshape base values to a col array and apply the growth rates
        out.loc[sel, target_years] = base_values.values[:, None] * growth_series[target_years].values

    return out




def aggregate_prices_volumes(df: pd.DataFrame, value_unit: str):
    
    """
    Orchestrates the pipeline:
    - Aggregate volumes
    - Compute values
    - Aggregate values
    - Aggregate prices
    """
    def aggregate_volumes(vol_df, year_cols):
        """
        Aggregate volumes by summing across Energy uses.
        Returns:
        Model, Scenario, Region, Variable='Volume',
        Energy consumers, Unit, <years>
        """
        group_cols = ['Model', 'Scenario', 'Region', 'Energy consumers', 'Unit']

        vol_agg = vol_df.groupby(group_cols, as_index=False)[year_cols].sum()

        # Insert Variable after Region
        vol_agg.insert(vol_agg.columns.get_loc('Region') + 1, 'Variable', 'Volume')

        return vol_agg


    def compute_values(vol_df, price_df, year_cols):
        """
        Compute VALUES = price × volume.
        Indexed by:
        (Model, Scenario, Region, Energy consumers, Energy uses)
        """
        keys = ['Model', 'Scenario', 'Region', 'Energy consumers', 'Energy uses']

        price_idx = price_df.set_index(keys)[year_cols]
        vol_idx = vol_df.set_index(keys)[year_cols].reindex(price_idx.index)

        value_matrix = price_idx.values * vol_idx.values

        values_df = pd.DataFrame(value_matrix,
                                index=price_idx.index,
                                columns=year_cols)
        return values_df


    def aggregate_values(values_df, year_cols, value_unit):
        """
        Aggregate VALUES across Energy uses.
        Returns DF with Variable='Values'.
        """
        group_cols = ['Model', 'Scenario', 'Region', 'Energy consumers']

        values_agg = (
            values_df.groupby(level=group_cols)
            .sum()
            .reset_index()
        )

        # Insert Variable correctly
        values_agg.insert(values_agg.columns.get_loc('Region') + 1,
                        'Variable', 'Values')

        values_agg['Unit'] = value_unit

        return values_agg


    def aggregate_prices(vol_agg, values_df, year_cols, price_unit):
        """
        Compute weighted average prices:
        price = Σ(values) / Σ(volumes)
        """
        group_cols = ['Model', 'Scenario', 'Region', 'Energy consumers']

        value_sum = values_df.groupby(level=group_cols).sum()
        vol_sum = vol_agg.set_index(group_cols)[year_cols]

        price_matrix = value_sum / vol_sum

        price_agg = price_matrix.reset_index()

        # Insert Variable
        price_agg.insert(price_agg.columns.get_loc('Region') + 1,
                        'Variable', 'Price')

        price_agg['Unit'] = price_unit

        return price_agg
    
    
    year_cols = [c for c in df.columns if str(c).isdigit()]

    vol_df = df[df['Variable'] == 'Volume'].copy()
    price_df = df[df['Variable'] == 'Price'].copy()

    # Extract units directly as strings
    price_unit = price_df['Unit'].iloc[0] if not price_df.empty else None

    # 1) Aggregate volumes
    vol_agg = aggregate_volumes(vol_df, year_cols)

    # 2) Compute raw VALUES
    disaggregated_values = compute_values(vol_df, price_df, year_cols)

    # 3) Aggregate VALUES
    values_agg = aggregate_values(disaggregated_values, year_cols, value_unit)

    # 4) Aggregate weighted average prices
    price_agg = aggregate_prices(vol_agg, disaggregated_values, year_cols, price_unit)

    # 5) Append results
    out = pd.concat([vol_agg, price_agg, values_agg], ignore_index=True)

    return out



def compute_mean_energy_price(df: pd.DataFrame):
    """
    Compute region-wide mean energy prices using aggregated Values and Volumes.
    """

    year_cols = [c for c in df.columns if str(c).isdigit()]

    # Extract needed slices
    values_df = df[df["Variable"] == "Values"].copy()
    volumes_df = df[df["Variable"] == "Volume"].copy()

    # Keep price unit (same for all aggregated entries)
    price_unit = df.loc[df["Variable"] == "Price", "Unit"].iloc[0]

    keys_consumer = ["Model", "Scenario", "Region", "Energy consumers"]
    group_keys = ["Model", "Scenario", "Region"]

    # Convert to indexed matrices
    values_idx = values_df.set_index(keys_consumer)[year_cols]
    volumes_idx = volumes_df.set_index(keys_consumer)[year_cols]

    # Sum across consumers (regional totals)
    total_values = values_idx.groupby(level=group_keys).sum()
    total_volumes = volumes_idx.groupby(level=group_keys).sum()

    # Compute mean prices
    mean_price = total_values / total_volumes
    mean_price = mean_price.reset_index()

    # Add metadata
    mean_price.insert(mean_price.columns.get_loc("Region") + 1,
                      "Variable", "Mean price")
    mean_price.insert(mean_price.columns.get_loc("Variable") + 1,
                      "Unit", price_unit)

    return mean_price



def compute_specific_margin_rates(df: pd.DataFrame):
    """
    Compute specific margin rates for each energy consumer:
        SMR = (price_consumer / regional_mean_price) - 1
    """

    year_cols = [c for c in df.columns if str(c).isdigit()]

    # Extract Prices and Volumes
    price_df = df[df["Variable"] == "Price"].copy()
    vol_df = df[df["Variable"] == "Volume"].copy()

    keys = ["Model", "Scenario", "Region", "Energy consumers"]

    price_idx = price_df.set_index(keys)[year_cols]
    vol_idx   = vol_df.set_index(keys)[year_cols]

    # ---------------------------------------------------------
    # 1) Compute mean prices using the modular function
    # ---------------------------------------------------------
    mean_price_df = compute_mean_energy_price(df)

    mean_price_idx = mean_price_df.set_index(
        ["Model", "Scenario", "Region"]
    )[year_cols]

    # Align mean price to each consumer
    mean_price_aligned = mean_price_idx.reindex(price_idx.index.droplevel("Energy consumers"))

    # ---------------------------------------------------------
    # 2) Compute SMR = price / mean_price - 1
    # ---------------------------------------------------------
    smr_values = (price_idx.values / mean_price_aligned.values) - 1

    smr_df = pd.DataFrame(smr_values,
                          index=price_idx.index,
                          columns=year_cols).reset_index()

    # Metadata
    smr_df.insert(smr_df.columns.get_loc("Region") + 1,
                  "Variable", "Specific margin rate")
    smr_df.insert(smr_df.columns.get_loc("Variable") + 1,
                  "Unit", "")

    return smr_df


def compute_delta_volumes(IOT_energy_consumption_dict, mean_price_df, delta_label, volume_unit):
    """
    Compute the Volume of DELTA for each region using:
    
        DELTA_volume_2020 = IOT_DELTA_value / mean_price_region_2020
        DELTA_volume_future_years = 0

    Parameters
    ----------
    IOT_energy_consumption_dict : dict[str → DataFrame]
        Keys = region names.
        Values = df with a row "ENERGY" and a column "DELTA".

    mean_price_df : pd.DataFrame
        Output from compute_mean_prices().
        Must contain:
        Model, Scenario, Region, Unit, and year columns.

    Returns
    -------
    pd.DataFrame
        With columns:
        Model, Scenario, Region, Variable='Volume',
        Energy consumers='DELTA', Unit='EJ/y', <years...>
    """

    # --- identify year columns ---
    year_cols = [c for c in mean_price_df.columns if str(c).isdigit()]

    rows = []

    # Iterate over all Model-Scenario-Region combinations
    for _, row in mean_price_df.iterrows():
        model = row["Model"]
        scenario = row["Scenario"]
        region = row["Region"]

        # Corresponding IOT regional dataframe
        iot_df = IOT_energy_consumption_dict[region]

        # DELTA value from IOT
        delta_value = iot_df.loc["ENERGY",delta_label]

        # Mean price for that region and this Model-Scenario
        mean_price_calibration = row[year_cols[0]]

        # Compute delta volume
        volumes = pd.Series(index=year_cols, dtype=float)
        volumes.iloc[0] = delta_value / mean_price_calibration   # 2020
        volumes.iloc[1:] = 0.0                            # later years

        rows.append({
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Volume",
            "Energy consumers": delta_label,
            "Unit": volume_unit,
            **volumes.to_dict()
        })

    return pd.DataFrame(rows)

def generate_delta_smr_zero(delta_vol_df, year_cols=None):
    """
    Generate a DataFrame of Specific Margin Rates for DELTA energy consumer
    filled with zeros.

    Parameters
    ----------
    delta_vol_df : pd.DataFrame
        Output of compute_delta_volumes() containing Volume rows for DELTA.
    year_cols : list[str], optional
        List of year columns. If None, detected automatically.

    Returns
    -------
    pd.DataFrame
        SMR rows for DELTA with all values 0.
    """
    if year_cols is None:
        year_cols = [c for c in delta_vol_df.columns if str(c).isdigit()]

    smr_delta_rows = delta_vol_df.copy()
    smr_delta_rows["Variable"] = "Specific margin rate"
    smr_delta_rows["Unit"] = ""
    smr_delta_rows[year_cols] = 0.0

    return smr_delta_rows


def append_delta_after_consumers(df, delta_df, delta_label="DELTA"):
    """
    Append the DELTA row after all other energy consumers for each 
    combination of Model, Scenario, Region, Variable.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame with multiple energy consumers per group.
    delta_df : pd.DataFrame
        DELTA rows (one per Model, Scenario, Region, Variable).
    delta_label : str
        Name of the DELTA consumer.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with DELTA appended after all other consumers 
        for each Model, Scenario, Region, Variable combination.
    """
    combined_rows = []

    group_cols = ["Model", "Scenario", "Region", "Variable"]

    # Iterate over each group of the original df
    for keys, group in df.groupby(group_cols, sort=False):
        # All normal consumers in this group
        group_normal = group.copy()

        # Get corresponding DELTA row from delta_df
        delta_row = delta_df[
            (delta_df["Model"] == keys[0]) &
            (delta_df["Scenario"] == keys[1]) &
            (delta_df["Region"] == keys[2]) &
            (delta_df["Variable"] == keys[3])
        ]

        # Concatenate normal + DELTA
        combined_group = pd.concat([group_normal, delta_row], ignore_index=True)

        combined_rows.append(combined_group)

    # Concatenate all groups
    combined_df = pd.concat(combined_rows, ignore_index=True)
    return combined_df



def append_delta_volumes_and_smr(vol_df, delta_vol_df, smr_df, delta_label="DELTA"):
    """
    Append DELTA volumes and SMR rows to the original DataFrames.
    DELTA is always appended as the last energy consumer for each
    Model/Scenario/Region/Variable combination. SMR of DELTA are zeros.

    Parameters
    ----------
    vol_df : pd.DataFrame
        Aggregated volumes from aggregate_prices_volumes().
    delta_vol_df : pd.DataFrame
        DELTA volumes from compute_delta_volumes().
    smr_df : pd.DataFrame
        Specific margin rates for normal energy consumers.
    delta_label : str, default "DELTA"
        Name of the DELTA energy consumer.

    Returns
    -------
    combined_vol_df : pd.DataFrame
        Volumes including DELTA as last consumer per group.
    combined_smr_df : pd.DataFrame
        SMR including DELTA rows (all zeros) as last consumer per group.
    """

    # --- 1) Append DELTA to volumes ---
    combined_vol_df = append_delta_after_consumers(vol_df, delta_vol_df, delta_label=delta_label)

    # --- 2) Generate DELTA SMR rows (all zeros) ---
    year_cols = [c for c in delta_vol_df.columns if str(c).isdigit()]
    delta_smr_df = generate_delta_smr_zero(delta_vol_df, year_cols=year_cols)

    # --- 3) Append DELTA to SMR ---
    combined_smr_df = append_delta_after_consumers(smr_df, delta_smr_df, delta_label=delta_label)

    final_df=pd.concat([combined_vol_df, combined_smr_df])

    final_df['Variable'] = final_df['Variable'].replace({'Volume': 'Energy consumption volume'})
    
    # 2. Appendi il contenuto di "Energy consumers" a "Variable" separato da "|"
    final_df['Variable'] = final_df['Variable'] + "|" + final_df['Energy consumers']
    
    # 3. Elimina la colonna "Energy consumers"
    final_df = final_df.drop(columns=['Energy consumers'])

    return final_df


def compute_delta_prices(mean_energy_prices_df: pd.DataFrame, delta_label: str):
    """
    Build a Price time series for the DELTA energy consumer.

    The calibration year (first year column) is set to the region mean energy
    price; all subsequent year columns are set to NaN.

    Parameters
    ----------
    mean_energy_prices_df : pd.DataFrame
        Output of compute_mean_energy_price(). Must contain columns:
        Model, Scenario, Region, Variable, Unit, and year columns.
    delta_label : str
        Name of the DELTA energy consumer.

    Returns
    -------
    pd.DataFrame
        With columns:
        Model, Scenario, Region, Variable='Price',
        Energy consumers=delta_label, Unit, <years>
    """
    year_cols = [c for c in mean_energy_prices_df.columns if str(c).isdigit()]

    delta_price = mean_energy_prices_df[["Model", "Scenario", "Region", "Unit"] + year_cols].copy()
    delta_price["Variable"] = "Price"
    delta_price["Energy consumers"] = delta_label

    for y in year_cols[1:]:
        delta_price[y] = float("nan")

    return delta_price[["Model", "Scenario", "Region", "Variable", "Energy consumers", "Unit"] + year_cols]