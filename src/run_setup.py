from datetime import datetime
import numpy as np
import pandas as pd


#############################################################
#######################  SETUP  #############################
#############################################################

#outoutput file name
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")
add_string = "test"
output_file_name = str().join(["results/", add_string, "(", dt_string, ")", ".csv"])

#growth ratios df
input_file =  "REMIND_exogenous_data_reformatted" 
growth_ratios_df = pd.read_csv("data/"+input_file+".csv") 

#energy calibration data
energy_calibration_data = pd.read_csv("data/calibration_2020.csv")

#years
years_int = np.array([eval(i) for i in growth_ratios_df.columns[3:]])
years = [str(y) for y in years_int]