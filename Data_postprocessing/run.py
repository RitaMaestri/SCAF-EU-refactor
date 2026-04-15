#DATA ANALYSIS
import matplotlib
matplotlib.use('Agg')
import os
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
import sys
import matplotlib.colors as mcolors
from scipy import stats
from lib import extract_var_df, plot_varj_evol, plot_varj_evol_absolute, plot_variable_1D, plot_KL_GDP_evolution, plot_VA_share_vs_log_gdp_per_capita, plot_energy_volumes_comparison, plot_energy_volumes_by_consumer, plot_Yj_vs_REMIND_output, plot_sector_Sj_Yj, sectors_names_eng, plot_energy_expenditure_by_sector, plot_energy_sector_inputs, plot_pY_Ej, plot_demand_components_stacked


# Set to a specific CSV path to plot a single run, or None to plot all tagged results
results_path = "Solver/results/tagged/results_2026-03-30_19-13/results_2026-03-30_19-13.csv"
#results_path = None

# Load shared data once
REMIND_E_volumes_path = "Solver/preprocessed_data/calibration/hybridization_df.csv"
REMIND_E_volumes = pd.read_csv(REMIND_E_volumes_path)
REMIND_E_volumes = REMIND_E_volumes[(REMIND_E_volumes["Region"]=="EUR") & (REMIND_E_volumes["Variable"]=="Volume")]

REMIND_output_path = "Data_preprocessing/data_calibration_evolution/Technical_coefficients/REMIND_activities_outputs.csv"
REMIND_output = pd.read_csv(REMIND_output_path)


def plot_run(results_path):
    csv_stem = os.path.splitext(os.path.basename(results_path))[0]
    output_dir = os.path.join("Data_postprocessing", "plots", csv_stem)
    os.makedirs(output_dir, exist_ok=True)

    yaml_src = os.path.join(os.path.dirname(results_path), f"{csv_stem}_metadata.yaml")
    if os.path.exists(yaml_src):
        metadata_dir = os.path.join(output_dir, "_metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        shutil.copy2(yaml_src, metadata_dir)

    SCAF_results = pd.read_csv(results_path)
    meta_cols = ['variable_name', 'row_label', 'col_label', 'status']
    year_cols = [c for c in SCAF_results.columns if c not in meta_cols and int(c) <= 2050]
    SCAF_results = SCAF_results[meta_cols + year_cols]
    
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_Mj=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_Mj=True, exclude_energy=True, fix_ylim=False, output_dir=output_dir)


    ################# demand components stacked bar charts ########################
    plot_demand_components_stacked(SCAF_results, year_cols, output_dir=output_dir)
    ###############################################################################



    plot_varj_evol_absolute(df=SCAF_results, var="KLj", pq="pq", display_top_names=7, mytitle="Absolute evolution of the value added per sector (value)", output_dir=output_dir)
    plot_energy_volumes_by_consumer(SCAF_results, year_cols, output_dir=output_dir)
    plot_varj_evol_absolute(df=SCAF_results, var="Xj", pq="pq", display_top_names=7, mytitle="Absolute evolution of exports per sector (value)", output_dir=output_dir)


    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_Sj=True, exclude_energy=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_Sj=True, exclude_energy=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_Cj=True, exclude_energy=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_consumption=True, exclude_energy=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_Ij=True, exclude_energy=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_Ij=True, exclude_energy=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_Gj=True, exclude_energy=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_Gj=True, exclude_energy=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_Xj=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_Xj=True, exclude_energy=True, fix_ylim=False, output_dir=output_dir)


    plot_energy_sector_inputs(SCAF_results, year_cols, output_dir=output_dir)

    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_Yj=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_Yj=True, fix_ylim=False, output_dir=output_dir)
    # Sj, Cj, Ij, Gj shares excluding energy sector from denominator


    # capital and labour share plots
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_capital=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_capital=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_capital=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_capital=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_labour=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_labour=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_labour=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_labour=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_Yj=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_Yj=True, fix_ylim=False, output_dir=output_dir)
    ###############################################################################

    ################# nominal energy expenditure by sector #########################
    plot_energy_expenditure_by_sector(SCAF_results, year_cols, output_dir=output_dir)
    plot_pY_Ej(SCAF_results, year_cols, output_dir=output_dir)
    ###############################################################################



    plot_varj_evol(df=SCAF_results, var="KLj", pq="pq", diff=False, display_top_names=7, mytitle="Normalised evolution of the value added per sector (value)", output_dir=output_dir)
    plot_varj_evol(df=SCAF_results, var="KLj", pq="q", diff=False, display_top_names=7, mytitle="Normalised evolution of the value added per sector (volume)", output_dir=output_dir)
    #plot_varj_evol(df=df, var="KLj", pq="p", diff=False, display_top_names=7, mytitle="Normalised evolution of the value added per sector (price)", output_dir=output_dir)

    plot_varj_evol(df=SCAF_results, var="Kj", pq="q", diff=False, display_top_names=7, mytitle="Normalised evolution of capital per sector (volume)", output_dir=output_dir)
    plot_varj_evol(df=SCAF_results, var="Lj", pq="q", diff=False, display_top_names=7, mytitle="Normalised evolution of labour per sector (volume)", output_dir=output_dir)

    plot_varj_evol(df=SCAF_results, var="Yj", pq="q", diff=False, display_top_names=7, output_dir=output_dir)

    plot_varj_evol(df=SCAF_results, var="Xj", pq="q",  diff=False, display_top_names=7, mytitle="Normalised evolution of exports per sector (volume)", output_dir=output_dir)

    plot_variable_1D(SCAF_results, "bKL", "q", diff=False, output_dir=output_dir)
    ###############################################################################
    plot_variable_1D(SCAF_results, "K", "p", diff=False, output_dir=output_dir)

    ################# plot GDP growth, capital and labour growth #######################
    plot_KL_GDP_evolution(SCAF_results, year_cols, output_dir=output_dir)
    ###############################################################################

    ################# plot sectoral VA share vs log GDP per capita #################
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_consumption=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_consumption=True,  output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_consumption=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_consumption=False, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_va=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_va=True, fix_ylim=False, output_dir=output_dir)
    # same plots excluding the energy sector from share computation
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_consumption=False, exclude_energy=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_consumption=True,  exclude_energy=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_consumption=True, fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_consumption=False, fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_va=True, exclude_energy=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_va=True, fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    # capital and labour share plots
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_capital=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_capital=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_capital=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_capital=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_labour=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_labour=True, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_labour=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_nominal_labour=True, fix_ylim=False, output_dir=output_dir)
###############################################################################

    ################# compare SCAF vs REMIND energy volumes #######################
    plot_energy_volumes_comparison(SCAF_results, REMIND_E_volumes, year_cols, output_dir=output_dir)
    ###############################################################################

    ################# compare Yj growth vs REMIND output by sector ################
    plot_Yj_vs_REMIND_output(SCAF_results, REMIND_output, year_cols, output_dir=output_dir)
    ###############################################################################

    ################# Sj, pSj, Yj, pYj per sector #################################
    for sector in sectors_names_eng:
        plot_sector_Sj_Yj(SCAF_results, year_cols, sector, output_dir=output_dir)
    ###############################################################################



if results_path:
    plot_run(results_path)
else:
    tagged_csvs = sorted(glob.glob("Solver/results/tagged/**/*.csv", recursive=True))
    for path in tagged_csvs:
        print(f"Plotting {path}...")
        plot_run(path)


