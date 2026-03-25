#DATA ANALYSIS
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
import sys
import matplotlib.colors as mcolors
from scipy import stats
from lib import extract_var_df, plot_varj_evol, plot_variable_1D, plot_KL_GDP_evolution



csv_path = "Solver/results/results(25-03-2026_17:59).csv"
csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
output_dir = os.path.join("Data_postprocessing", "plots", csv_stem)
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
meta_cols = ['variable_name', 'row_label', 'col_label', 'status']
year_cols = [c for c in df.columns if c not in meta_cols and int(c) <= 2050]
df = df[meta_cols + year_cols]



plot_varj_evol(df=df, var="KLj", pq="pq", diff=False,display_top_names=7, mytitle="Normalised evolution of the value added per sector (value)", output_dir=output_dir)
plot_varj_evol(df=df, var="KLj", pq="q", diff=False,display_top_names=7, mytitle="Normalised evolution of the value added per sector (volume)", output_dir=output_dir)
#plot_varj_evol(df=df, var="KLj", pq="p", diff=False,display_top_names=7, mytitle="Normalised evolution of the value added per sector (price)", output_dir=output_dir)

plot_varj_evol(df=df, var="Kj", pq="q", diff=False,display_top_names=7, mytitle="Normalised evolution of capital per sector (volume)", output_dir=output_dir)
plot_varj_evol(df=df, var="Lj", pq="q", diff=False,display_top_names=7, mytitle="Normalised evolution of labour per sector (volume)", output_dir=output_dir)

plot_varj_evol(df=df, var="Yj", pq="q", diff=False,display_top_names=7, output_dir=output_dir)



plot_variable_1D(df, "bKL", "q", diff=False, output_dir=output_dir)
###############################################################################
plot_variable_1D(df, "K", "p", diff=False, output_dir=output_dir)



################# plot GDP growth, capital and labour growth #######################
plot_KL_GDP_evolution(df, year_cols, output_dir=output_dir)
###############################################################################







