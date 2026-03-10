from pathlib import Path
import pandas as pd

def filter_NGFS(NGFS_filtered_path, NGFS_path, variables_list):
    if Path(NGFS_filtered_path).exists():
        NGFS = pd.read_csv(NGFS_filtered_path)
    
    else:
        NGFS_total = pd.read_excel(
            NGFS_path, sheet_name='data', header=0
        )


        NGFS = NGFS_total[
            (NGFS_total["Model"] == 'REMIND-MAgPIE 3.2-4.6') &
            (NGFS_total["Region"].str.startswith('REMIND')) &
            (NGFS_total["Variable"].isin(variables_list))
        ]

        NGFS.to_csv(NGFS_filtered_path, index=False)
        del NGFS_total

    return NGFS

def compute_growth_rates(df):
    # Identifica le colonne-anno (numeri)
    year_cols = [col for col in df.columns if str(col).isdigit()]
    year_cols = sorted(year_cols)  # Ordine crescente
    
    # Primo anno = calibration_year
    calibration_year = year_cols[0]
    
    # Copia il dataset originale
    df_growth = df.copy()
    
    # Calcolo growth rate: -1 + valore_t / valore_2020
    for year in year_cols:
        df_growth[year] = -1 + (df[year] / df[calibration_year])
    
    # Modifica Unit
    df_growth["Unit"] = ""
    
    # Modifica Variable
    df_growth["Variable"] = df_growth["Variable"] + "|growth rate"
    
    return df_growth