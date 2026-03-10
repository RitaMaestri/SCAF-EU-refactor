import numpy as np
import pandas as pd
import sys
sys.path.append('/home/rita/Documents/Tesi/Projects/SCAF-EU-refactor/Solver/src')
import warnings
warnings.filterwarnings("ignore")



from import_GTAP_data import N
from Variables_specs import VARIABLES_SPECS
from helpers.time_series_df_functions import build_and_fill_timeseries_df




########################################################################
########################## BUILD TIMESERIES DF #########################
########################################################################

from run_setup import years

input_file_name = "Solver/data/REMIND_exogenous_data_reformatted.csv" #temporary, to be replaced with the one from run_setup
growth_ratios_df = pd.read_csv(input_file_name)
output_file_name = "Solver/results/template_df_evolution.csv" #temporary, to be replaced with the one from run_setup

timeseries_df=build_and_fill_timeseries_df(VARIABLES_SPECS,growth_ratios_df,years)
timeseries_df.to_csv(output_file_name, index=False)      
