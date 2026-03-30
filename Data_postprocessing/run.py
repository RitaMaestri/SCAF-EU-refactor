#DATA ANALYSIS
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
import sys
import matplotlib.colors as mcolors
from scipy import stats
from lib import extract_var_df, plot_varj_evol, plot_variable_1D, plot_KL_GDP_evolution, plot_VA_share_vs_log_gdp_per_capita, plot_energy_volumes_comparison, plot_Yj_vs_REMIND_output



results_path = "Solver/results/drafts/results_2026-03-30_12-40.csv"

REMIND_E_volumes_path = "Solver/preprocessed_data/calibration/hybridization_df.csv"
REMIND_E_volumes = pd.read_csv(REMIND_E_volumes_path)
REMIND_E_volumes = REMIND_E_volumes[(REMIND_E_volumes["Region"]=="EUR") & (REMIND_E_volumes["Variable"]=="Volume")]

csv_stem = os.path.splitext(os.path.basename(results_path))[0]
output_dir = os.path.join("Data_postprocessing", "plots", csv_stem)
os.makedirs(output_dir, exist_ok=True)

REMIND_output_path = "Data_preprocessing/data_calibration_evolution/Technical_coefficients/REMIND_activities_outputs.csv"
REMIND_output = pd.read_csv(REMIND_output_path)

SCAF_results = pd.read_csv(results_path)
meta_cols = ['variable_name', 'row_label', 'col_label', 'status']
year_cols = [c for c in SCAF_results.columns if c not in meta_cols and int(c) <= 2050]
SCAF_results = SCAF_results[meta_cols + year_cols]



plot_varj_evol(df=SCAF_results, var="KLj", pq="pq", diff=False,display_top_names=7, mytitle="Normalised evolution of the value added per sector (value)", output_dir=output_dir)
plot_varj_evol(df=SCAF_results, var="KLj", pq="q", diff=False,display_top_names=7, mytitle="Normalised evolution of the value added per sector (volume)", output_dir=output_dir)
#plot_varj_evol(df=df, var="KLj", pq="p", diff=False,display_top_names=7, mytitle="Normalised evolution of the value added per sector (price)", output_dir=output_dir)

plot_varj_evol(df=SCAF_results, var="Kj", pq="q", diff=False,display_top_names=7, mytitle="Normalised evolution of capital per sector (volume)", output_dir=output_dir)
plot_varj_evol(df=SCAF_results, var="Lj", pq="q", diff=False,display_top_names=7, mytitle="Normalised evolution of labour per sector (volume)", output_dir=output_dir)

plot_varj_evol(df=SCAF_results, var="Yj", pq="q", diff=False,display_top_names=7, output_dir=output_dir)



plot_variable_1D(SCAF_results, "bKL", "q", diff=False, output_dir=output_dir)
###############################################################################
plot_variable_1D(SCAF_results, "K", "p", diff=False, output_dir=output_dir)



################# plot GDP growth, capital and labour growth #######################
plot_KL_GDP_evolution(SCAF_results, year_cols, output_dir=output_dir)
###############################################################################

################# plot sectoral VA share vs log GDP per capita #################
plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_consumption=False, output_dir=output_dir)
plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_consumption=True,  output_dir=output_dir)
###############################################################################

################# compare SCAF vs REMIND energy volumes #######################
plot_energy_volumes_comparison(SCAF_results, REMIND_E_volumes, year_cols, output_dir=output_dir)
###############################################################################

################# compare Yj growth vs REMIND output by sector ################
plot_Yj_vs_REMIND_output(SCAF_results, REMIND_output, year_cols, output_dir=output_dir)
###############################################################################







