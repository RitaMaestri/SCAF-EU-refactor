import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")



from import_GTAP_data import N
from Variables_specs import VARIABLES_SPECS
from time_series_df_functions import build_timeseries_df




########################################################################
########################## BUILD TIMESERIES DF #########################
########################################################################

from run_setup import years

output_file_name = "results/test_template_df.csv" #temporary, to be replaced with the one from run_setup

timeseries_df=build_timeseries_df(VARIABLES_SPECS,years)

timeseries_df.to_csv(output_file_name, index=False)      
