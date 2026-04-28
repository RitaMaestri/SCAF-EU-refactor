#DATA ANALYSIS
import matplotlib
matplotlib.use('Agg')
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
import sys
import matplotlib.colors as mcolors
from scipy import stats
from lib import extract_var_df, plot_varj_evol, plot_varj_evol_absolute, plot_variable_1D, plot_KL_GDP_evolution, plot_VA_share_vs_log_gdp_per_capita, plot_variable_share_vs_log_gdp_per_capita, plot_variable_diff_by_sector, plot_aggregate_diff, plot_structural_change_panel, plot_structural_change_panel_diff, plot_energy_volumes_comparison_by_use, plot_total_energy_volume_comparison, plot_energy_volumes_diverging_stacked, plot_energy_volumes_diverging_stacked_scaf_diff, plot_energy_volumes_by_consumer, plot_Yj_vs_REMIND_output, plot_sector_Sj_Yj, plot_sector_Sj_Yj_diff, plot_real_va_output_diff, plot_real_va_vs_gdp, sectors_names_eng, plot_energy_expenditure_by_sector, plot_energy_expenditure_share, plot_export_share_of_output, plot_export_share_of_output_diff, plot_real_export_share_of_output, plot_nominal_demand_evolutions, plot_energy_sector_inputs, plot_pY_Ej, plot_demand_components_stacked, plot_KL_shares_by_sector, check_GDP_decomposed_vs_GDPreal

# =============================================================================
# SETUP
# =============================================================================
PLOT_FUNCTION = "presentation_plots"  # "presentation_plots" | "exploratory_plots" | "narrow_presentation_plots"
DATA_SOURCE   = "mapping"             # "single" | "mapping"

# --- single mode: paths, subtitle and key set manually ---
SINGLE_RESULTS_PATH = "Solver/results/tagged/results_2026-04-28_14-49/results_2026-04-28_14-49.csv"
SINGLE_NO_SC_PATH   = "Solver/results/tagged/results_2026-04-14_17-45/results_2026-04-14_17-45.csv"  # or None
SINGLE_SUBTITLE     = "No structural change"
SINGLE_KEY          = "no_SC"

# --- mapping mode: filter by run_set ---
ACTIVE_RUN_SETS = ["corrected_GDPPI"]  # or None to process all run sets
# =============================================================================

# Load shared data once
REMIND_E_volumes_path = "Solver/preprocessed_data/calibration/hybridization_df.csv"
REMIND_E_volumes = pd.read_csv(REMIND_E_volumes_path)
REMIND_E_volumes = REMIND_E_volumes[(REMIND_E_volumes["Region"]=="EUR") & (REMIND_E_volumes["Variable"]=="Volume")]

REMIND_output_path = "Data_preprocessing/data_calibration_evolution/Technical_coefficients/REMIND_activities_outputs.csv"
REMIND_output = pd.read_csv(REMIND_output_path)


def load_no_sc_df(path):
    if not path:
        return None
    raw = pd.read_csv(path)
    meta = ['variable_name', 'row_label', 'col_label', 'status']
    ycols = [c for c in raw.columns if c not in meta and int(c) <= 2050]
    return raw[meta + ycols]


def presentation_plots(results_path, subtitle="", output_name=None, no_sc_df=None):
    csv_stem = os.path.splitext(os.path.basename(results_path))[0]
    output_dir = os.path.join("Data_postprocessing", "plots", output_name if output_name else csv_stem)
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

    ################# structural change panel #######################################
    plot_structural_change_panel(SCAF_results, year_cols, fix_ylim=True,  subtitle=subtitle, output_dir=output_dir)
    plot_structural_change_panel(SCAF_results, year_cols, fix_ylim=False, subtitle=subtitle, output_dir=output_dir)
    plot_structural_change_panel(SCAF_results, year_cols, fix_ylim=True,  subtitle=subtitle, include_real_va=False, output_dir=output_dir)
    plot_structural_change_panel(SCAF_results, year_cols, fix_ylim=False, subtitle=subtitle, include_real_va=False, output_dir=output_dir)
    if no_sc_df is not None:
        plot_structural_change_panel_diff(SCAF_results, no_sc_df, year_cols, subtitle=subtitle, output_dir=output_dir)
        plot_structural_change_panel_diff(SCAF_results, no_sc_df, year_cols, subtitle=subtitle, include_real_va=False, output_dir=output_dir)
    ###############################################################################

    if no_sc_df is not None:
        plot_energy_volumes_diverging_stacked_scaf_diff(SCAF_results, no_sc_df, year_cols, output_dir=output_dir)

    plot_energy_volumes_diverging_stacked(SCAF_results, REMIND_E_volumes, year_cols,
                                          scaf_label=subtitle, output_dir=output_dir)

    plot_total_energy_volume_comparison(
        SCAF_results, REMIND_E_volumes, year_cols,
        include_PE=True,
        scaf_label=subtitle,
        df_no_sc=no_sc_df,
        output_dir=output_dir,
    )


def narrow_presentation_plots(results_path, subtitle="", key="", no_sc_df=None):
    output_dir = os.path.join("Data_postprocessing", "plots", "presentation", key)
    os.makedirs(output_dir, exist_ok=True)

    SCAF_results = pd.read_csv(results_path)
    meta_cols = ['variable_name', 'row_label', 'col_label', 'status']
    year_cols = [c for c in SCAF_results.columns if c not in meta_cols and int(c) <= 2050]
    SCAF_results = SCAF_results[meta_cols + year_cols]

    if no_sc_df is not None:
        plot_energy_volumes_diverging_stacked_scaf_diff(
            SCAF_results, no_sc_df, year_cols,
            output_path=os.path.join(output_dir, "E_bar.png"))

    plot_total_energy_volume_comparison(
        SCAF_results, REMIND_E_volumes, year_cols,
        include_PE=True, scaf_label=subtitle, df_no_sc=no_sc_df,
        output_path=os.path.join(output_dir, "E_tot.png"))

    plot_structural_change_panel(
        SCAF_results, year_cols, fix_ylim=True, subtitle=subtitle,
        include_real_va=False,
        output_path=os.path.join(output_dir, "SC_ind.png"))

    if no_sc_df is not None:
        plot_structural_change_panel_diff(
            SCAF_results, no_sc_df, year_cols, subtitle=subtitle,
            include_real_va=False,
            output_path=os.path.join(output_dir, "SC_diff.png"))


def exploratory_plots(results_path, subtitle="", no_sc_df=None):
    csv_stem = os.path.splitext(os.path.basename(results_path))[0]
    output_dir = os.path.join("Data_postprocessing", "plots", "exploratory_analysis", csv_stem)
    os.makedirs(output_dir, exist_ok=True)

    SCAF_results = pd.read_csv(results_path)
    meta_cols = ['variable_name', 'row_label', 'col_label', 'status']
    year_cols = [c for c in SCAF_results.columns if c not in meta_cols and int(c) <= 2050]
    SCAF_results = SCAF_results[meta_cols + year_cols]

    check_GDP_decomposed_vs_GDPreal(SCAF_results, year_cols)

    plot_variable_1D(SCAF_results, "GDPPI", "q", diff=False, output_dir=output_dir)

    ################# capital/labour income shares (year 0) ########################
    plot_KL_shares_by_sector(SCAF_results, year_cols, output_dir=output_dir)
    ###############################################################################

    if no_sc_df is not None:
        plot_variable_diff_by_sector(SCAF_results, no_sc_df, year_cols, "Yj", "pYj", is_nominal=False, y_label="Δ real output (pYj₀·Yj)", output_dir=output_dir)
        plot_variable_diff_by_sector(SCAF_results, no_sc_df, year_cols, "Yj", "pYj", is_nominal=True, y_label="Δ nominal output (pYj₀·Yj)", output_dir=output_dir)
        plot_variable_diff_by_sector(SCAF_results, no_sc_df, year_cols, "Xj", "pXj", is_nominal=False, y_label="Δ real export (pXj₀·Xj)", output_dir=output_dir)
        plot_variable_diff_by_sector(SCAF_results, no_sc_df, year_cols, "Mj", "pMj", is_nominal=False, y_label="Δ real import (pMj₀·Mj)", output_dir=output_dir)
        plot_variable_diff_by_sector(SCAF_results, no_sc_df, year_cols, "Sj", "pSj", is_nominal=False, y_label="Δ real total supply (pSj₀·Sj)", output_dir=output_dir)

        plot_aggregate_diff(SCAF_results, no_sc_df, year_cols, "GDP",   y_label="Δ GDP (%)",   output_dir=output_dir)
        plot_aggregate_diff(SCAF_results, no_sc_df, year_cols, "GDPPI", y_label="Δ GDPPI (%)", output_dir=output_dir)
        plot_aggregate_diff(SCAF_results, no_sc_df, year_cols, "pL",    y_label="Δ pL (%)",    output_dir=output_dir)
        plot_aggregate_diff(SCAF_results, no_sc_df, year_cols, "pK",    y_label="Δ pK (%)",    output_dir=output_dir)
        plot_real_va_output_diff(SCAF_results, no_sc_df, year_cols, output_dir=output_dir)

    ################# plot GDP growth, capital and labour growth #######################
    plot_KL_GDP_evolution(SCAF_results, year_cols, output_dir=output_dir)
    plot_real_va_vs_gdp(SCAF_results, year_cols, output_dir=output_dir)
    ###############################################################################

    ################# compare SCAF vs REMIND energy volumes #######################
    plot_energy_volumes_comparison_by_use(SCAF_results, REMIND_E_volumes, year_cols, scaf_label=subtitle, output_dir=output_dir)
    plot_total_energy_volume_comparison(SCAF_results, REMIND_E_volumes, year_cols, scaf_label=subtitle, output_dir=output_dir)
    plot_energy_volumes_diverging_stacked(SCAF_results, REMIND_E_volumes, year_cols, output_dir=output_dir)
    ###############################################################################


    ################# nominal energy expenditure by sector #########################
    plot_energy_expenditure_by_sector(SCAF_results, year_cols, output_dir=output_dir)
    plot_energy_expenditure_by_sector(SCAF_results, year_cols, normalise=False, output_dir=output_dir)
    plot_energy_expenditure_share(SCAF_results, year_cols, output_dir=output_dir)
    plot_export_share_of_output(SCAF_results, year_cols, output_dir=output_dir)
    plot_real_export_share_of_output(SCAF_results, year_cols, output_dir=output_dir)
    if no_sc_df is not None:
        plot_export_share_of_output_diff(SCAF_results, no_sc_df, year_cols, output_dir=output_dir)
    plot_nominal_demand_evolutions(SCAF_results, year_cols, output_dir=output_dir)
    plot_pY_Ej(SCAF_results, year_cols, output_dir=output_dir)
    ###############################################################################

    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Mj", "pMj", True,  "Share of nominal imports", fix_ylim=False, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Mj", "pMj", True,  "Share of nominal imports", fix_ylim=False, exclude_energy=True, output_dir=output_dir)

    ################# demand components stacked bar charts ########################
    plot_demand_components_stacked(SCAF_results, year_cols, output_dir=output_dir)
    ###############################################################################

    plot_varj_evol_absolute(df=SCAF_results, var="KLj", pq="pq", display_top_names=7, mytitle="Absolute evolution of the value added per sector (value)", output_dir=output_dir)
    plot_energy_volumes_by_consumer(SCAF_results, year_cols, output_dir=output_dir)
    plot_varj_evol_absolute(df=SCAF_results, var="Xj", pq="pq", display_top_names=7, mytitle="Absolute evolution of exports per sector (value)", output_dir=output_dir)

    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Sj", "pSj", False, "Share of real sales",                    fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Sj", "pSj", True,  "Share of nominal sales",                 fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Cj", "pCj", False, "Share of real household consumption",    fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Cj", "pCj", True,  "Share of nominal household consumption", fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Ij", "pCj", False, "Share of real investment",               fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Ij", "pCj", True,  "Share of nominal investment",            fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Gj", "pCj", False, "Share of real government expenditure",   fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Gj", "pCj", True,  "Share of nominal government expenditure",fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Xj", "pXj", True,  "Share of nominal exports",              fix_ylim=False, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Xj", "pXj", True,  "Share of nominal exports",              fix_ylim=False, exclude_energy=True, output_dir=output_dir)

    plot_energy_sector_inputs(SCAF_results, year_cols, output_dir=output_dir)

    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Yj", "pYj", False, "Share of real output",   fix_ylim=False, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Yj", "pYj", True,  "Share of nominal output",fix_ylim=False, output_dir=output_dir)

    # Sj, Cj, Ij, Gj shares excluding energy sector from denominator

    # capital and labour share plots
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Kj", "pK", False, "Share of real capital",   output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Kj", "pK", False, "Share of real capital",   fix_ylim=False, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Kj", "pK", True,  "Share of nominal capital",output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Kj", "pK", True,  "Share of nominal capital",fix_ylim=False, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Lj", "pL", False, "Share of real labour",    output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Lj", "pL", False, "Share of real labour",    fix_ylim=False, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Lj", "pL", True,  "Share of nominal labour", output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Lj", "pL", True,  "Share of nominal labour", fix_ylim=False, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Yj", "pYj", False, "Share of real output",   fix_ylim=False, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Yj", "pYj", True,  "Share of nominal output",fix_ylim=False, output_dir=output_dir)
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

    ################# plot sectoral VA share vs log GDP per capita #################
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Cj", "pCj", True,  "Share of nominal household consumption", output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Cj", "pCj", True,  "Share of nominal household consumption", fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, fix_ylim=False, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_va=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_va=True, fix_ylim=False, output_dir=output_dir)
    # same plots excluding the energy sector from share computation
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, exclude_energy=True, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Cj", "pCj", True,  "Share of nominal household consumption", exclude_energy=True, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Cj", "pCj", True,  "Share of nominal household consumption", fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_va=True, exclude_energy=True, output_dir=output_dir)
    plot_VA_share_vs_log_gdp_per_capita(SCAF_results, year_cols, use_real_va=True, fix_ylim=False, exclude_energy=True, output_dir=output_dir)
    # capital and labour share plots
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Kj", "pK", False, "Share of real capital",   output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Kj", "pK", False, "Share of real capital",   fix_ylim=False, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Kj", "pK", True,  "Share of nominal capital",output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Kj", "pK", True,  "Share of nominal capital",fix_ylim=False, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Lj", "pL", False, "Share of real labour",    output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Lj", "pL", False, "Share of real labour",    fix_ylim=False, output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Lj", "pL", True,  "Share of nominal labour", output_dir=output_dir)
    plot_variable_share_vs_log_gdp_per_capita(SCAF_results, year_cols, "Lj", "pL", True,  "Share of nominal labour", fix_ylim=False, output_dir=output_dir)
    ###############################################################################

    ################# compare Yj growth vs REMIND output by sector ################
    plot_Yj_vs_REMIND_output(SCAF_results, REMIND_output, year_cols, output_dir=output_dir)
    ###############################################################################

    ################# Sj, pSj, Yj, pYj per sector #################################
    for sector in sectors_names_eng:
        plot_sector_Sj_Yj(SCAF_results, year_cols, sector, output_dir=output_dir)
        if no_sc_df is not None:
            plot_sector_Sj_Yj_diff(SCAF_results, no_sc_df, year_cols, sector, output_dir=output_dir)
    ###############################################################################


def run_plots(results_path, subtitle="", output_name=None, no_sc_df=None):
    if PLOT_FUNCTION == "presentation_plots":
        presentation_plots(results_path, subtitle=subtitle, output_name=output_name, no_sc_df=no_sc_df)
    elif PLOT_FUNCTION == "exploratory_plots":
        exploratory_plots(results_path, subtitle=subtitle, no_sc_df=no_sc_df)
    elif PLOT_FUNCTION == "narrow_presentation_plots":
        narrow_presentation_plots(results_path, subtitle=subtitle, key=output_name or "", no_sc_df=no_sc_df)


# =============================================================================
# MAIN
# =============================================================================
if DATA_SOURCE == "single":
    no_sc_df = load_no_sc_df(SINGLE_NO_SC_PATH)
    run_plots(SINGLE_RESULTS_PATH, subtitle=SINGLE_SUBTITLE, output_name=SINGLE_KEY, no_sc_df=no_sc_df)

elif DATA_SOURCE == "mapping":
    mapping = pd.read_csv("Data_postprocessing/results_mapping.csv", skipinitialspace=True)
    if ACTIVE_RUN_SETS is not None:
        mapping = mapping[mapping["run_set"].str.strip().isin(ACTIVE_RUN_SETS)]

    for _, row in mapping.iterrows():
        run_id        = row["ID"].strip()
        origin_folder = row["origin_folder"].strip()
        key           = row["key"].strip()
        run_set       = row["run_set"].strip()
        subtitle      = str(row["plot_subtitle"]).strip() if pd.notna(row["plot_subtitle"]) else ""
        path          = os.path.join("Solver", "results", origin_folder, run_id, f"{run_id}.csv")

        no_sc_match = mapping[
            (mapping["number_of_drivers"].str.strip() == "baseline") &
            (mapping["run_set"].str.strip() == run_set)
        ]
        no_sc_path = None
        if not no_sc_match.empty:
            _r   = no_sc_match.iloc[0]
            _rid = _r["ID"].strip()
            no_sc_path = os.path.join("Solver", "results", _r["origin_folder"].strip(), _rid, f"{_rid}.csv")

        run_plots(path, subtitle=subtitle, output_name=os.path.join(run_set, key), no_sc_df=load_no_sc_df(no_sc_path))
