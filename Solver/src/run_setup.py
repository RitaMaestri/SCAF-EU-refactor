from datetime import datetime
import numpy as np
import pandas as pd


#############################################################
#######################  SETUP  #############################
#############################################################

#run name (used to build output paths)
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H-%M")
run_name = f"results_{dt_string}"

#growth ratios df
growth_ratios_df = pd.read_csv("Solver/preprocessed_data/growth_factors.csv")

#energy calibration data
energy_calibration_data = pd.read_csv("Solver/preprocessed_data/calibration/hybridization_df.csv")
population_calibration_data = pd.read_csv("Solver/preprocessed_data/calibration/population.csv")

#assumed variables
assumed_variables_df = pd.read_csv("Solver/assumed_data/assumed_variables.csv", index_col="variable_name")

#elasticity data
armington_elasticities_df = pd.read_csv("Solver/preprocessed_data/elasticities/armington_elasticities.csv", index_col="commodity")
export_elasticities_df = pd.read_csv("Solver/assumed_data/GTAP_export_elasticities7.csv", index_col="code")
kl_elasticities_df = pd.read_csv("Solver/preprocessed_data/elasticities/kl_elasticities.csv", index_col="commodity")
income_elasticities_df = pd.read_csv("Solver/preprocessed_data/elasticities/income_elasticities.csv", index_col="commodity")
compensated_price_elasticities_df = pd.read_csv("Solver/preprocessed_data/elasticities/compensated_own_price_elasticities.csv", index_col="commodity")

#years
years_int = np.array([eval(i) for i in growth_ratios_df.columns[3:]])
years = [str(y) for y in years_int]
calibration_year = years[0]