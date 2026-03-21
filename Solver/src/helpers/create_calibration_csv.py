import pandas as pd
import numpy as np

def create_calibration_csv(calibration_obj, output_file="Solver/data/calibration_2020.csv",
                           energy_consumers_csv="Solver/preprocessed_data/energy_consumers.csv"):
    """
    Create a CSV file with the same format as data/input_format.csv,
    but with calibration data for the year 2020 only, with other years left blank.
    
    Mapping:
    - YE_Ej → PE (Primary Energy) for ENERGY sector only
    - YE_Pj → P (Process Energy) for all sectors with non-zero values
    - YE_Tj → T (Transport Energy) for all sectors
    - YE_Bj → B (Buildings Energy) for all sectors with non-zero values
    - C_EB → B (Buildings Energy) for HOUSEHOLDS
    - C_ET → T (Transport Energy) for HOUSEHOLDS
    
    Parameters
    ----------
    calibration_obj : calibrationVariables
        An instance of calibrationVariables containing the calibration results
    output_file : str
        Path to the output CSV file
    energy_consumers_csv : str
        Path to the energy_consumers CSV (column ``"energy_consumer"``).
    """
    
    # Read the template file to get the structure
    template_df = pd.read_csv("Solver/data/input_format.csv")
    
    # Get unique values for each column (except the year columns)
    all_years = [str(year) for year in range(2020, 2101, 5)]
    
    # Create a list to store all rows
    rows = []
    
    # Get the sector mappings from the calibration object
    # These are the actual indices used in the model
    from calibration import A, M, SE, E, ST, CH, T
    
    # Map actual indices to sector names
    index_to_sector = {
        A: "AGRICULTURE",
        M: "MANUFACTURE", 
        SE: "SERVICES",
        E: "ENERGY",
        ST: "STEEL",
        CH: "CHEMICAL",
        T: "TRANSPORTATION",
    }
    
    all_sector_names = pd.read_csv(energy_consumers_csv)["energy_consumer"].tolist()
    
    # Get model, scenario, and region from template
    model = template_df["Model"].iloc[0]
    scenario = template_df["Scenario"].iloc[0]
    region = template_df["Region"].iloc[0]
    
    # Create rows for Volume data
    for sector_name in all_sector_names:
        
        # Find the sector index
        sector_idx = None
        for idx, name in index_to_sector.items():
            if name == sector_name:
                sector_idx = idx
                break
        
        # PE (Primary Energy)
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Volume",
            "Energy consumers": sector_name,
            "Energy uses": "PE",
            "Unit": "EJ"
        }
        for year in all_years:
            if year == "2020":
                if sector_name == "ENERGY" and sector_idx is not None:
                    row[year] = calibration_obj.YE_Ej[sector_idx]
                else:
                    row[year] = 0
            else:
                row[year] = 0
        rows.append(row)
        
        # P (Process Energy)
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Volume",
            "Energy consumers": sector_name,
            "Energy uses": "P",
            "Unit": "EJ"
        }
        for year in all_years:
            if year == "2020":
                if sector_idx is not None:
                    row[year] = calibration_obj.YE_Pj[sector_idx]
                else:
                    row[year] = 0
            else:
                row[year] = 0
        rows.append(row)
        
        # T (Transport Energy)
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Volume",
            "Energy consumers": sector_name,
            "Energy uses": "T",
            "Unit": "EJ"
        }
        for year in all_years:
            if year == "2020":
                if sector_name == "HOUSEHOLDS":
                    # C_ET for HOUSEHOLDS transport
                    row[year] = calibration_obj.C_ET
                elif sector_idx is not None:
                    row[year] = calibration_obj.YE_Tj[sector_idx]
                else:
                    row[year] = 0
            else:
                row[year] = 0
        rows.append(row)
        
        # B (Buildings Energy)
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Volume",
            "Energy consumers": sector_name,
            "Energy uses": "B",
            "Unit": "EJ"
        }
        for year in all_years:
            if year == "2020":
                if sector_name == "HOUSEHOLDS":
                    # C_EB for HOUSEHOLDS buildings
                    row[year] = calibration_obj.C_EB
                elif sector_idx is not None:
                    row[year] = calibration_obj.YE_Bj[sector_idx]
                else:
                    row[year] = 0
            else:
                row[year] = 0
        rows.append(row)
    
    # Create rows for Price data
    # Mapping:
    # - pE_B: all sectors, B
    # - pE_TT: TRANSPORTATION, T
    # - pE_TnT: all sectors except TRANSPORTATION, T
    # - pE_Pj: all sectors (by position), P
    # - pE_Ej: all sectors, PE
    
    for sector_name in all_sector_names:
        
        # Find the sector index
        sector_idx = None
        for idx, name in index_to_sector.items():
            if name == sector_name:
                sector_idx = idx
                break
        
        # PE (Primary Energy) - pE_Ej
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Price",
            "Energy consumers": sector_name,
            "Energy uses": "PE",
            "Unit": "EJ"
        }
        for year in all_years:
            if year == "2020":
                if sector_idx is not None:
                    row[year] = calibration_obj.pE_Ej[sector_idx]
                else:
                    row[year] = 0
            else:
                row[year] = 0
        rows.append(row)
        
        # P (Process Energy) - pE_Pj
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Price",
            "Energy consumers": sector_name,
            "Energy uses": "P",
            "Unit": "EJ"
        }
        for year in all_years:
            if year == "2020":
                if sector_idx is not None:
                    row[year] = calibration_obj.pE_Pj[sector_idx]
                else:
                    row[year] = 0
            else:
                row[year] = 0
        rows.append(row)
        
        # T (Transport Energy) - pE_TT or pE_TnT depending on sector
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Price",
            "Energy consumers": sector_name,
            "Energy uses": "T",
            "Unit": "EJ"
        }
        for year in all_years:
            if year == "2020":
                if sector_name == "TRANSPORTATION":
                    row[year] = calibration_obj.pE_TT
                else:
                    # All other sectors (including HOUSEHOLDS) get pE_TnT
                    row[year] = calibration_obj.pE_TnT
            else:
                row[year] = 0
        rows.append(row)
        
        # B (Buildings Energy) - pE_B (same for all sectors)
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Price",
            "Energy consumers": sector_name,
            "Energy uses": "B",
            "Unit": "EJ"
        }
        for year in all_years:
            if year == "2020":
                row[year] = calibration_obj.pE_B
            else:
                row[year] = 0
        rows.append(row)
    
    # Create rows for Rho data
    # Mapping:
    # - rhoB: all sectors, B
    # - rhoTT: TRANSPORTATION, T
    # - rhoTnT: all sectors except TRANSPORTATION, T
    # - rhoPj: all sectors (by position), P
    # Note: No PE (Primary Energy) rows for Rho
    
    for sector_name in all_sector_names:
        
        # Find the sector index
        sector_idx = None
        for idx, name in index_to_sector.items():
            if name == sector_name:
                sector_idx = idx
                break
        
        # P (Process Energy) - rhoPj
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Rho",
            "Energy consumers": sector_name,
            "Energy uses": "P",
            "Unit": ""
        }
        for year in all_years:
            if year == "2020":
                if sector_idx is not None:
                    row[year] = calibration_obj.rhoPj[sector_idx]
                else:
                    row[year] = 0
            else:
                row[year] = 0
        rows.append(row)
        
        # T (Transport Energy) - rhoTT or rhoTnT depending on sector
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Rho",
            "Energy consumers": sector_name,
            "Energy uses": "T",
            "Unit": ""
        }
        for year in all_years:
            if year == "2020":
                if sector_name == "TRANSPORTATION":
                    row[year] = calibration_obj.rhoTT
                else:
                    # All other sectors (including HOUSEHOLDS) get rhoTnT
                    row[year] = calibration_obj.rhoTnT
            else:
                row[year] = 0
        rows.append(row)
        
        # B (Buildings Energy) - rhoB (same for all sectors)
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Rho",
            "Energy consumers": sector_name,
            "Energy uses": "B",
            "Unit": ""
        }
        for year in all_years:
            if year == "2020":
                row[year] = calibration_obj.rhoB
            else:
                row[year] = 0
        rows.append(row)
    
    # Create rows for Technical Coefficients (a)
    # Mapping:
    # - aYE_Bj: all sectors, B
    # - aYE_Pj: all sectors, P
    # - aYE_Tj: all sectors, T
    # - aYE_Ej: all sectors, only non-zero for ENERGY; E
    # Note: No PE rows for technical coefficients
    
    # Calculate aYE_Ej if not already in calibration_obj
    aYE_Ej = calibration_obj.YE_Ej / calibration_obj.Yj0
    
    for sector_name in all_sector_names:
        # Skip HOUSEHOLDS for technical coefficients
        if sector_name == "HOUSEHOLDS":
            continue
        
        # Find the sector index
        sector_idx = None
        for idx, name in index_to_sector.items():
            if name == sector_name:
                sector_idx = idx
                break
        
        # P (Process Energy) - aYE_Pj
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Technical_coefficient",
            "Energy consumers": sector_name,
            "Energy uses": "P",
            "Unit": ""
        }
        for year in all_years:
            if year == "2020":
                if sector_idx is not None:
                    row[year] = calibration_obj.aYE_Pj[sector_idx]
                else:
                    row[year] = 0
            else:
                row[year] = 0
        rows.append(row)
        
        # T (Transport Energy) - aYE_Tj
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Technical_coefficient",
            "Energy consumers": sector_name,
            "Energy uses": "T",
            "Unit": ""
        }
        for year in all_years:
            if year == "2020":
                if sector_idx is not None:
                    row[year] = calibration_obj.aYE_Tj[sector_idx]
                else:
                    row[year] = 0
            else:
                row[year] = 0
        rows.append(row)
        
        # B (Buildings Energy) - aYE_Bj
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Technical_coefficient",
            "Energy consumers": sector_name,
            "Energy uses": "B",
            "Unit": ""
        }
        for year in all_years:
            if year == "2020":
                if sector_idx is not None:
                    row[year] = calibration_obj.aYE_Bj[sector_idx]
                else:
                    row[year] = 0
            else:
                row[year] = 0
        rows.append(row)
        
        # E (Primary Energy) - aYE_Ej (only non-zero for ENERGY sector)
        row = {
            "Model": model,
            "Scenario": scenario,
            "Region": region,
            "Variable": "Technical_coefficient",
            "Energy consumers": sector_name,
            "Energy uses": "E",
            "Unit": ""
        }
        for year in all_years:
            if year == "2020":
                if sector_idx is not None:
                    row[year] = aYE_Ej[sector_idx]
                else:
                    row[year] = 0
            else:
                row[year] = 0
        rows.append(row)
    
    # Convert to DataFrame and save
    output_df = pd.DataFrame(rows)
    
    # Reorder columns to match template
    column_order = ["Model", "Scenario", "Region", "Variable", "Energy consumers", "Energy uses", "Unit"] + all_years
    output_df = output_df[column_order]
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Calibration CSV saved to {output_file}")
    
    return output_df


