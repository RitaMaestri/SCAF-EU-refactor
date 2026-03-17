import pandas as pd
import numpy as np
import sys
import os
import math

# Aggiunge la cartella Hybridization (parent di test/) al path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lib import Energy_Values_Allocation, aggregate_energy_uses, aggregate_IOT_energy_consumption, aggregate_by_consumer_and_use, rename_regions, build_availabilities_df, filter_REMIND

#####################################
### TEST Energy_Values_Allocation ###
#####################################
IOT_E_dict = {"AGRICULTURE": 12.449, "MANUFACTURE": 125.362, "SERVICES": 444.915, "STEEL": 5.231, "CHEMICAL": 56.424, "ENERGY": 389.904, "TRANSPORTATION":23.441, "HOUSEHOLDS":295.832}

REMIND_E_dict = {"LDV": 392.324940090403, 
            "PAS": 80.782406009737,
                "FRG": 212.209853448321,
                "STE": 17.9798067109948,
                "CHE": 176.355138536102, 
                "AGR": 16.1317836253807, 
                "MAN":141.185402187977, 
                "R&C": 470.592, 
                "PRIM":441.026078920331
}
REMIND_E_prices ={"LDV": 28.4900345280386, 
            "PAS": 28.8146514024431,
                "FRG": 28.289858973512,
                "STE": 16.188147835388,
                "CHE": 18.4137874973537, 
                "AGR": 19.1238875563563, 
                "MAN":19.1238875563563, 
                "R&C": 23.6071704615138, 
                "PRIM":8.0177258221305
}


IOT_E = pd.Series(IOT_E_dict)

REMIND_E = pd.Series(REMIND_E_dict)

REMIND_E_prices = pd.Series(REMIND_E_prices)

priorities_path = os.path.join(os.path.dirname(__file__), "priorities_test.csv")
key_path = os.path.join(os.path.dirname(__file__), "key_1D.csv")
key_df_path = os.path.join(os.path.dirname(__file__), "key_matrix.csv")

priorities = pd.read_csv(priorities_path,header=0, index_col=False)

key_csv = pd.read_csv(key_path, header=None, names=["sector", "value"])

key_df= pd.read_csv(key_df_path, index_col=0)

key = pd.Series(key_csv['value'].values, index=key_csv['sector'])

def test_energy_allocation():

    allocation = Energy_Values_Allocation(IOT_E_consumptions=IOT_E,
            REMIND_E_uses=REMIND_E,
            REMIND_E_prices= REMIND_E_prices,
            priorities=priorities,
            key=key_df)

    rescaling_factor=allocation.rescale_REMIND_energy_values()

    if allocation.REMIND_E_uses.sum()== allocation.IOT_E_consumptions.sum():
        print("Test passed: REMIND energy values correctly rescaled ✅")

    allocation.allocate_forced_energy_values()
    allocation.adjust_key_for_forced_values()

    # Check if the key is correctly adjusted

    print("The key is correctly adjusted: ",
        math.isclose(allocation.key_df.loc["MANUFACTURE","LDV"]/allocation.key_df.loc["SERVICES","LDV"],
                    key_df.loc["MANUFACTURE","LDV"]/key_df.loc["SERVICES","LDV"]))

    disaggregated_energy = allocation.compute_disaggregated_energy()
    disaggregated_prices = allocation.compute_prices_matrix()
    disaggregated_volumes = allocation.compute_volumes_matrix()

    #check the constraint error
    print("max vertical sum error: ",max(abs(-1 + disaggregated_energy.sum(axis=0) / allocation.REMIND_E_uses)))
    print("max horizontal sum error: ",max(-1 + disaggregated_energy.sum(axis=1) / allocation.IOT_E_consumptions))
    return disaggregated_energy, disaggregated_prices, disaggregated_volumes


##################################################
##### TEST build_energy_consumptions_dict ########
##################################################


def test_energy_array_for_CAZ(IOT_path, mappings_path, test_path):
    """
    Test that the ENERGY array for CAZ matches the expected result in test_path.
    
    Parameters:
    - iot_data: dict of IOT DataFrames (output of generate_energy_consumptions_array)
    - col_label_df: DataFrame mapping IOT columns to SCAF_name
    - row_label_df: DataFrame mapping IOT rows
    - test_path: str, path to CSV/Excel file containing the expected array for CAZ
    """
    # --- Compute energy arrays using the function ---
    energy_arrays = aggregate_IOT_energy_consumption(IOT_path,mappings_path)
    
    expected_df = pd.read_excel(test_path, index_col=0, header=0)
    
    expected_array = expected_df.to_numpy()
    
    # --- Extract CAZ result ---
    CAZ_array = energy_arrays.get("CAZ")
    if CAZ_array is None:
        raise AssertionError("No data found for region 'CAZ'")
    
    # --- Compare ---
    if not np.allclose(CAZ_array, expected_array, atol=1e-6):
        raise AssertionError("CAZ energy array does not match the expected result")
    
    print("Test passed: CAZ energy array matches the expected result ✅")


IOT_path = "/home/rita/Documents/Tesi/Projects/SCAF-IAMAX Original/Data_preprocessing/data_calibration_evolution/regional_IOTs/new/"
map_IOT_path = "/home/rita/Documents/Tesi/Projects/SCAF-IAMAX Original/Data_preprocessing/src/Hybridization/mappings/IOT-energy_consumers.xlsx"
test_path = os.path.join(os.path.dirname(__file__), "test_IOT_mapping_CAZ.xlsx")


#test_energy_array_for_CAZ(IOT_path, map_IOT_path, test_path)

##################################################
##### TEST aggregate_energy_types ################
##################################################

REMIND_augmented_path = "/home/rita/Documents/Tesi/Projects/SCAF-IAMAX Original/Data_preprocessing/src/Hybridization/cache/NGFS_hybridization_augmented.csv"
map_REMIND_energy_uses_path = "/home/rita/Documents/Tesi/Projects/SCAF-IAMAX Original/Data_preprocessing/src/Hybridization/mappings/NGFS_energy_uses.xlsx"
test_path = os.path.join(os.path.dirname(__file__), "test_aggregation_energies.xlsx")


def test_aggregate_energy_types(REMIND_augmented_path, map_REMIND_energy_uses_path, test_path):

    test=pd.read_excel(test_path,sheet_name="Sheet1",header=0)
    result = aggregate_energy_types(REMIND_augmented_path, map_REMIND_energy_uses_path)["values"]

    selected_row =result[
        (result["Model"] == test.loc[0, "Model"]) &
        (result["Scenario"] == test.loc[0, "Scenario"]) &
        (result["Region"] == test.loc[0, "Region"]) &
        (result["energy_use"] == test.loc[0, "energy_use"])
    ] 

    tolerance = 1e-6
    all_close = np.allclose(selected_row.iloc[0,4:].astype(float), test.iloc[0,4:].astype(float), atol=tolerance)

    print("🔍 Running aggregation consistency test:")
    print(f"  ➤ Function: aggregate_energy_types()")
    print(f"  ➤ Checking that the aggregated values match the expected reference data in '{os.path.basename(test_path)}'")
    print(f"    within a tolerance of {tolerance}.\n")

    if all_close:
        print(f"✅ Test PASSED — All values match within tolerance {tolerance}.")
    else:
        print(f"❌ Test FAILED — Some values differ beyond tolerance {tolerance}.")


#test_aggregate_energy_types(REMIND_augmented_path, map_REMIND_energy_uses_path, test_path)

##################################################################
####################### TEST MAPPING #############################
##################################################################

map_REMIND_energy_uses=pd.read_excel(map_REMIND_energy_uses_path,header=0)
REMIND_augmented = pd.read_csv(REMIND_augmented_path,header=0)

def check_variables_presence_in_mapping(map_REMIND_energy_uses,REMIND_augmented):
    prices_mapping=pd.unique(map_REMIND_energy_uses["REMIND_price"])
    volumes_mapping=pd.unique(map_REMIND_energy_uses["REMIND_volume"])

    all_mapping_variables=set(np.concatenate([prices_mapping,volumes_mapping]))

    all_REMIND_augmented_variables=set(pd.unique(REMIND_augmented["Variable"]))


    print("in mapping but not REMIND \n",
    sorted(all_mapping_variables.difference(all_REMIND_augmented_variables)))

    print("in REMIND but not mapping \n",
    sorted(all_REMIND_augmented_variables.difference(all_mapping_variables)))

#check_variables_presence_in_mapping(map_REMIND_energy_uses,REMIND_augmented)

##################################################################
####################### TEST MAPPING #############################
##################################################################

IEA_mapping_path = "src/Hybridization/mappings/mapping_IEA.csv"
IEA_data_path ="src/Hybridization/IEA/result/aggregate_residential_agriculture_2020.csv"
priorities_path="src/Hybridization/mappings/priorities/priorities.csv"
volumes_path = "src/Hybridization/cache/volumes.csv"
region_mapping_path = "src/Hybridization/mappings/region_mapping.csv"

IEA_mapping=pd.read_csv(IEA_mapping_path, header=0)
IEA_data=pd.read_csv(IEA_data_path, header=0)
volumes=pd.read_csv(volumes_path, header=0)

region_mapping=pd.read_csv(region_mapping_path, header=0)

aggregated_IEA=aggregate_by_consumer_and_use(IEA_data,IEA_mapping)

aggregated_IEA = rename_regions(region_mapping, aggregated_IEA)



model= 'REMIND-MAgPIE 3.2-4.6'
scenario="Net Zero 2050"

calibration_year="2020"

#out_df=build_availabilities_df(volumes, aggregated_IEA, model, scenario, calibration_year)


##############################
### TEST SMR SUM TO ZERO #####
##############################


def test_specific_margin_rates(smr_df, apv_df):
    """
    Verifica che per ogni Model–Scenario–Region la somma di:
        SMR_consumer_j * Volume_consumer_j
    sia uguale a zero per ogni anno.

    Parametri:
        smr_df : DataFrame
            Output di compute_specific_margin_rates
        apv_df : DataFrame
            Output di aggregate_prices_volumes_append (contiene prezzi e volumi)

    Restituisce:
        DataFrame con le discrepanze (dovrebbero essere tutti zeri).
        Se completamente zero → test passato.
    """

    # Identifica colonne anno
    year_cols = [c for c in smr_df.columns if str(c).isdigit()]

    # --- 1. Filtra i volumi dal df APV ---
    vol_df = apv_df[apv_df["Variable"] == "Volume"].copy()

    # --- 2. Allinea gli indici di SMR e volumi ---
    keys = ["Model", "Scenario", "Region", "Energy consumers"]

    smr_idx = smr_df.set_index(keys)[year_cols]
    vol_idx = vol_df.set_index(keys)[year_cols]

    # Assicuriamoci che l'ordine corrisponda perfettamente
    if not smr_idx.index.equals(vol_idx.index):
        vol_idx = vol_idx.reindex(smr_idx.index)

    # --- 3. Calcola SMR * Volume per ogni consumer ---
    margin_contrib = smr_idx.values * vol_idx.values
    contrib_df = pd.DataFrame(
        margin_contrib, index=smr_idx.index, columns=year_cols
    )

    # --- 4. Aggrega per Model–Scenario–Region ---
    contrib_df_grouped = (
        contrib_df
        .groupby(level=["Model", "Scenario", "Region"])
        .sum()
    )

    return contrib_df_grouped



prices_volumes_df_path = "src/Hybridization/cache/aggregated_output.csv"

prices_volumes_df = pd.read_csv(prices_volumes_df_path, header=0)

smr_df_path = "src/Hybridization/cache/sm_rates.csv"

smr_df = pd.read_csv(smr_df_path, header=0)


res = test_specific_margin_rates(smr_df, prices_volumes_df)

# Controllo diretto:
print("Do specific margins sum to zero: "+str((res.abs() < 1e-9).all().all()))


def test_specific_margin_rates_lower_bound(smr_df):
    """
    Test that no specific margin rate value is lower than -1.

    Parameters
    ----------
    smr_df : pd.DataFrame
        DataFrame containing Specific Margin Rates with columns:
        Model, Scenario, Region, Variable='Specific margin rate',
        Energy consumers, Unit, <years...>

    Raises
    ------
    ValueError
        If any specific margin rate is < -1
    """
    # Identify year columns
    year_cols = [c for c in smr_df.columns if str(c).isdigit()]

    # Filter only SMR rows
    smr_only = smr_df[smr_df["Variable"] == "Specific margin rate"]

    # Check for any value < -1
    mask = (smr_only[year_cols] < -1).any().any()
    if mask:
        raise ValueError("❌ There are specific margin rate values lower than -1!")
    else:
        print("✅ All specific margin rates are >= -1")

test_specific_margin_rates_lower_bound(smr_df)


##################################################
##### TEST filter_REMIND ###########################
##################################################

def test_filter_REMIND():
    """
    Test that filter_REMIND:
    - drops year columns before 2020
    - drops rows where Region == 'World'
    - saves the result to the specified output path
    """
    import tempfile

    data = {
        "Model":    ["REMIND", "REMIND", "REMIND", "REMIND"],
        "Scenario": ["SSP2-NPi2025"] * 4,
        "Region":   ["CAZ", "EUR", "World", "CHA"],
        "Variable": ["Final Energy|Industry"] * 4,
        "Unit":     ["EJ/yr"] * 4,
        "2005":     [1.0, 2.0, 3.0, 4.0],
        "2010":     [1.1, 2.1, 3.1, 4.1],
        "2015":     [1.2, 2.2, 3.2, 4.2],
        "2020":     [1.3, 2.3, 3.3, 4.3],
        "2025":     [1.4, 2.4, 3.4, 4.4],
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_path = f.name

    try:
        result = filter_REMIND(df, tmp_path)

        # No pre-2020 year columns
        year_cols = [c for c in result.columns if str(c).isdigit()]
        assert all(int(c) >= 2020 for c in year_cols), \
            f"Pre-2020 year columns still present: {[c for c in year_cols if int(c) < 2020]}"

        # No World rows
        assert "World" not in result["Region"].values, \
            "Region='World' rows were not removed"

        # Correct number of rows (3 non-World rows)
        assert len(result) == 3, f"Expected 3 rows, got {len(result)}"

        # Output CSV was created
        assert os.path.exists(tmp_path), "Output CSV was not created"

        print("test_filter_REMIND passed ✅")
    finally:
        os.remove(tmp_path)


test_filter_REMIND()