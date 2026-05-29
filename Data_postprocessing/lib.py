import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
import sys
import matplotlib.colors as mcolors
from scipy import stats



cmap=["#E06969",
"#B35900",
"#6CD900",
"#8FFFFF",
"#8C8CFF",
"#6300C7",
"#FF91C8"
]




# cmap=["#E06969",
# "#E80000",
# "#B30000",
# "#FFA64D",
# "#F27900",
# "#B35900",
# "#FFFF73",
# "#BFBF00",
# "#7A7A00",
# "#B9FF73",
# "#6CD900",
# "#3D7A00",
# "#54F0A2",
# "#00BF60",
# "#00572B",
# "#8FFFFF",
# "#19C6FF",
# "#007D7D",
# "#73B9FF",
# "#0060BF",
# "#003469",
# "#8C8CFF",
# "#5959FF",
# "#0000FF",
# "#000069",
# "#B973FF",
# "#6300C7",
# "#340069",
# "#FF00FF",
# "#B00CB0",
# "#5C005C",
# "#9C004E",
# "#FF1F8F",
# "#FF91C8",
# "#000000",
# "#545454",
# "#9294A1"]

my_cmap=mcolors.ListedColormap(cmap)

ENERGY_USE_PALETTE = ["#f79380","#9380f7","#2ac7a5","#bcc82a","#278699","#914060","#1a4a35"]

sectors_names_eng=[
"AGRICULTURE",
"MANUFACTURE",
"SERVICES",
"STEEL",
"CHEMICAL",
"ENERGY",
"TRANSPORTATION",
    ]

def extract_var_df(var, pq, df):
    meta_cols = ['variable_name', 'row_label', 'col_label', 'status']
    year_cols = [c for c in df.columns if c not in meta_cols]

    P_rows = df.loc[df['variable_name'] == "p"+var].reset_index(drop=True)
    Q_rows = df.loc[df['variable_name'] == var].reset_index(drop=True)

    P = P_rows[year_cols]
    Q = Q_rows[year_cols]

    if pq=="pq":
        var_df = P*Q
    elif pq == "p":
        var_df = P
    elif pq == "q":
        var_df= Q
    else:
        print("wrong pq!")
        sys.exit()

    if len(var_df.index)>1:
        source_rows = P_rows if pq == "p" else Q_rows
        var_df = var_df.reset_index(drop=True)
        var_df.insert(0, "Name", source_rows['row_label'].values, True)
    return var_df
    
    


def plot_varj_evol(df, var, pq, max_year="2050", display_top_names=4, display_bottom_names=0, diff=False, mytitle=None, output_dir=None):
    
    var_df=extract_var_df(var, pq, df)
    var_df=var_df.loc[:,:max_year]
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111)
    
    x=np.array(var_df.columns[1:]).astype('int')
    
    y0= x[0].astype(str)
    yL= x[-1].astype(str)
    
    
    relative_values=var_df[yL].values/var_df[y0].values
    nans=np.argwhere(np.isnan(relative_values)).flatten()
    #nansX=np.argwhere(np.array(df["2050"].loc[ df['variable'] == "Xj" ]==0)).flatten()
    rank= np.argsort(-relative_values)
    rank=rank[~np.in1d(rank,nans)]
    display_bottom_names=len(sectors_names_eng)-display_bottom_names
    index_side_writing=np.hstack([rank[:display_top_names],rank[display_bottom_names:]])
    
    
    for j in rank:
        y=np.array(var_df.loc[j])[1:].astype('float')
        y= y/y[0]-shift(y/y[0], 1, cval=np.NaN) if diff else y/y[0]
        lab=var_df.loc[j][0]
        ax.plot(x[0:],y[0:] , label = lab)
        if j in index_side_writing:
            ax.annotate(xy=(x[-1],y[-1]), xytext=(5,0), textcoords='offset points', text=var_df.loc[j][0], va='center')
    
    # #if I want ordered colors
    # for i,j in enumerate(ax.lines):
    #     j.set_color(my_cmap.colors[i])
    
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1.16, 0.5), prop={'size': 16})
        
    #if I want each color to be assigned to a specific sector
    for j in range(len(var_df)):
        y = np.array(var_df.iloc[j, 1:]).astype('float')
        y = y / y[0] - np.roll(y / y[0], 1) if diff else y / y[0]
        lab = var_df.iloc[j, 0]
        color = my_cmap(j)  # Assign color based on position in the dataframe
        ax.plot(x, y, label=lab, color=color)
        if j in index_side_writing:
            ax.annotate(text=lab, xy=(x[-1], y[-1]), xytext=(5, 0), textcoords='offset points', va='center')
        
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # Remove duplicates
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1.16, 0.5), prop={'size': 13})
    
    # Shrink current axis's height by 10% on the bottom
    # Shrink current axis by 20%

    if mytitle==None:
        if pq=="pq":
            plt.title("p"+var+var,fontsize = 17)
        elif pq == "p":
            plt.title("p"+var,fontsize = 17)
        elif pq == "q":
            plt.title(var,fontsize = 17)
    else:
        plt.title(mytitle,fontsize = 17)
            
    plt.xlim(2019.99,2050.01)
    
    
    # #devo dividere per year[0]
    # absolute_plot_2Darray=var_df.iloc[:,1:].to_numpy()
    # (absolute_plot_2Darray.T/absolute_plot_2Darray[]).T
    # relative_plot_2Darray=absolute_plot_2Darray
    # df_max=var_df.iloc[:,1:].to_numpy().max()
    # df_min=var_df.iloc[:,1:].to_numpy().min()
    # delta=(df_max-df_min)*0.05
    # plt.xlim(df_min-delta,df_max+delta)
    
    #plt.title("Value added by sector", fontsize = 25)
    plt.xlabel("Year",fontsize = 17)
    plt.ylabel("Relative change with respect to year 2020", fontsize = 17)

    if output_dir is not None:
        stem = (mytitle if mytitle else (pq + var)).replace(" ", "_")
        subdir = os.path.join(output_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, stem + ".png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_varj_evol_absolute(df, var, pq, max_year="2050", display_top_names=7, display_bottom_names=0, mytitle=None, output_dir=None):
    """Like plot_varj_evol but plots absolute values (not normalised to calibration year)."""
    var_df = extract_var_df(var, pq, df)
    var_df = var_df.loc[:, :max_year]
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    x = np.array(var_df.columns[1:]).astype('int')
    yL = x[-1].astype(str)

    end_values = var_df[yL].values
    nans = np.argwhere(np.isnan(end_values)).flatten()
    rank = np.argsort(-end_values)
    rank = rank[~np.in1d(rank, nans)]
    display_bottom_names_idx = len(sectors_names_eng) - display_bottom_names
    index_side_writing = np.hstack([rank[:display_top_names], rank[display_bottom_names_idx:]])

    for j in range(len(var_df)):
        y = np.array(var_df.iloc[j, 1:]).astype('float')
        lab = var_df.iloc[j, 0]
        color = my_cmap(j)
        ax.plot(x, y, label=lab, color=color)
        if j in index_side_writing:
            ax.annotate(text=lab, xy=(x[-1], y[-1]), xytext=(5, 0), textcoords='offset points', va='center')

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1.16, 0.5), prop={'size': 13})

    if mytitle is None:
        if pq == "pq":
            plt.title("p" + var + var + " (absolute)", fontsize=17)
        elif pq == "p":
            plt.title("p" + var + " (absolute)", fontsize=17)
        elif pq == "q":
            plt.title(var + " (absolute)", fontsize=17)
    else:
        plt.title(mytitle, fontsize=17)

    plt.xlim(2019.99, 2050.01)
    plt.xlabel("Year", fontsize=17)
    plt.ylabel("Absolute value", fontsize=17)

    if output_dir is not None:
        stem = (mytitle if mytitle else (pq + var + "_absolute")).replace(" ", "_")
        subdir = os.path.join(output_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, stem + ".png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_splitted_evolutions(var, pq, diff):
    
    var_df=extract_var_df(var, pq)
    for i in np.split(var_df.index, [9,18,27]):
        for j in i:
            x=np.array(var_df.columns[1:]).astype('int')
            y=np.array(var_df.loc[j])[1:].astype('float')
            y= y/y[0]-shift(y/y[0], 1, cval=np.NaN) if diff else y/y[0]
            plt.plot(x[0:],y[0:] , label = var_df.loc[j][0])
        ax = plt.subplot(111)
        
        # Shrink current axis's height by 10% on the bottom
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        if pq=="pq":
            plt.title("p"+var+var)
        elif pq == "p":
            plt.title("p"+var)
        elif pq == "q":
            plt.title(var)
        
        plt.show()


def plot_1D(var_df, title, diff=False, output_dir=None):
    fig = plt.figure(figsize=(15,10))
    x=np.array(var_df.columns).astype('int')
    y=np.array(var_df)[0].astype('float')
    y= y-shift(y, 1, cval=np.NaN) if diff else y
    plt.plot(x,y , label = title)
    plt.title(title, fontsize = 35)
    plt.xlabel("Year",fontsize = 20)
    plt.ylabel("Relative change with respect to year 2020", fontsize = 20)
    if output_dir is not None:
        stem = title.replace(" ", "_")
        subdir = os.path.join(output_dir, stem)
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, stem + ".png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
def plot_variable_1D(df, var_name, pq, diff=False, output_dir=None):
    var_df= extract_var_df(var_name, pq, df)
    plot_1D(var_df, var_name, diff=diff, output_dir=output_dir)


def plot_KL_GDP_evolution(df, year_cols, output_dir=None):
    L=df.loc[df['variable_name'] == "L", year_cols].values[0].astype("float")
    bKL=df.loc[df['variable_name'] == "bKL", year_cols].values[0].astype("float")
    LbKL=L*bKL
    K=df.loc[df['variable_name'] == "K", year_cols].values[0].astype("float")
    GDPreal=df.loc[df['variable_name'] == "GDPreal", year_cols].values[0].astype("float")
    pL=df.loc[df['variable_name'] == "pL", year_cols].values[0].astype("float")
    pK=df.loc[df['variable_name'] == "pK", year_cols].values[0].astype("float")

    fig = plt.figure(figsize=(13,7))
    ax = fig.add_subplot(111)
    x=np.array(year_cols).astype('int')
    y=  pL/pL[0]
    plt.plot(x,y,label = "$Labour \ price$", linestyle='dashed')
    y=  K/K[0]
    plt.plot(x,y,label = "$Capital$")
    y=  GDPreal/GDPreal[0]
    plt.plot(x,y,label = "$GDP$")
    y=  LbKL/LbKL[0]
    plt.plot(x,y,label = "$Labour$")
    y=  pK/pK[0]
    plt.plot(x,y,label = "$Capital \ price$", linestyle='dashed')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 13})
    plt.title("Evolution of the parameters from the coupled IAM", fontsize = 20)
    plt.xlabel("Year",fontsize = 17)
    plt.ylabel("Relative change with respect to year 2020", fontsize = 17)

    if output_dir is not None:
        subdir = os.path.join(output_dir, "IAM_parameters_evolution")
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, "IAM_parameters_evolution.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_VA_share_vs_log_gdp_per_capita(df, year_cols, use_real_va=False,
                                        fix_ylim=True, exclude_energy=False, output_dir=None):
    if use_real_va:
        lj_rows = df.loc[df['variable_name'] == "Lj"].reset_index(drop=True)
        kj_rows = df.loc[df['variable_name'] == "Kj"].reset_index(drop=True)
        pL_base = df.loc[df['variable_name'] == "pL", year_cols[0]].values[0].astype("float")
        pK_base = df.loc[df['variable_name'] == "pK", year_cols[0]].values[0].astype("float")
        lj_vals = lj_rows[year_cols].values.astype("float")
        kj_vals = kj_rows[year_cols].values.astype("float")
        sector_names = lj_rows['row_label'].values
        nominal = lj_vals * pL_base + kj_vals * pK_base
        ylabel = "Share of real value added"
        fname_prefix = "real_VA_share"
    else:
        p_rows = df.loc[df['variable_name'] == "pKLj"].reset_index(drop=True)
        q_rows = df.loc[df['variable_name'] == "KLj"].reset_index(drop=True)
        ylabel = "Share of value added"
        fname_prefix = "VA_share"
        p_vals = p_rows[year_cols].values.astype("float")
        q_vals = q_rows[year_cols].values.astype("float")
        sector_names = p_rows['row_label'].values
        nominal = p_vals * q_vals

    if exclude_energy:
        mask = np.array([s.upper() != "ENERGY" for s in sector_names])
        nominal = nominal[mask]
        sector_names = sector_names[mask]
        fname_prefix += "_excl_energy"

    shares = nominal / nominal.sum(axis=0)               # (n_sectors, n_years)

    GDPreal = df.loc[df['variable_name'] == "GDPreal", year_cols].values[0].astype("float")
    L       = df.loc[df['variable_name'] == "L",       year_cols].values[0].astype("float")
    log_gdp_pc = np.log(GDPreal / L)

    if fix_ylim:
        fname_prefix += "_not_normalised"

    subdir = os.path.join(output_dir, fname_prefix) if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    for j, sector in enumerate(sector_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(log_gdp_pc, shares[j], marker='o', markersize=4, linewidth=1.5)
        ax.set_title(sector, fontsize=17)
        ax.set_xlabel("log(GDP per capita)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        if fix_ylim:
            ax.set_ylim(0, 1)

        if output_dir is not None:
            fname = f"{fname_prefix}_{sector.replace(' ', '_')}.png"
            plt.savefig(os.path.join(subdir, fname), bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_variable_share_vs_log_gdp_per_capita(df, year_cols, quantity, price, is_nominal, y_label,
                                               fix_ylim=True, exclude_energy=False, output_dir=None):
    q_rows = df.loc[df['variable_name'] == quantity].reset_index(drop=True)
    sector_names = q_rows['row_label'].values
    if is_nominal:
        p_vals = df.loc[df['variable_name'] == price, year_cols].values.astype("float")
        nominal = p_vals * q_rows[year_cols].values.astype("float")
        fname_prefix = f"nominal_{quantity}_shares"
    else:
        p_base = df.loc[df['variable_name'] == price, year_cols[0]].values[0].astype("float")
        nominal = q_rows[year_cols].values.astype("float") * p_base
        fname_prefix = f"real_{quantity}_shares"

    if exclude_energy:
        mask = np.array([s.upper() != "ENERGY" for s in sector_names])
        nominal = nominal[mask]
        sector_names = sector_names[mask]
        fname_prefix += "_excl_energy"

    shares = nominal / nominal.sum(axis=0)

    GDPreal = df.loc[df['variable_name'] == "GDPreal", year_cols].values[0].astype("float")
    L       = df.loc[df['variable_name'] == "L",       year_cols].values[0].astype("float")
    log_gdp_pc = np.log(GDPreal / L)

    if fix_ylim:
        fname_prefix += "_not_normalised"

    subdir = os.path.join(output_dir, fname_prefix) if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    for j, sector in enumerate(sector_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(log_gdp_pc, shares[j], marker='o', markersize=4, linewidth=1.5)
        ax.set_title(sector, fontsize=17)
        ax.set_xlabel("log(GDP per capita)", fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        if fix_ylim:
            ax.set_ylim(0, 1)

        if output_dir is not None:
            fname = f"{fname_prefix}_{sector.replace(' ', '_')}.png"
            plt.savefig(os.path.join(subdir, fname), bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_variable_diff_by_sector(df, df_ref, year_cols, quantity, price, is_nominal, y_label,
                                 output_dir=None):
    q_df  = df.loc[df['variable_name']         == quantity].reset_index(drop=True)
    q_ref = df_ref.loc[df_ref['variable_name'] == quantity].reset_index(drop=True)
    sector_names = q_df['row_label'].values

    if is_nominal:
        p_df  = df.loc[df['variable_name']         == price, year_cols].values.astype("float")
        p_ref = df_ref.loc[df_ref['variable_name'] == price, year_cols].values.astype("float")
        val_df  = p_df  * q_df[year_cols].values.astype("float")
        val_ref = p_ref * q_ref[year_cols].values.astype("float")
        fname_prefix = f"diff_nominal_{quantity}"
    else:
        p_base = df_ref.loc[df_ref['variable_name'] == price, year_cols[0]].values[0].astype("float")
        val_df  = q_df[year_cols].values.astype("float")  * p_base
        val_ref = q_ref[year_cols].values.astype("float") * p_base
        fname_prefix = f"diff_real_{quantity}"

    diff = val_df - val_ref
    years = [int(c) for c in year_cols]

    subdir = os.path.join(output_dir, fname_prefix) if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    for j, sector in enumerate(sector_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(years, diff[j], marker='o', markersize=4, linewidth=1.5)
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_title(sector, fontsize=17)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)

        if output_dir is not None:
            fname = f"{fname_prefix}_{sector.replace(' ', '_')}.png"
            plt.savefig(os.path.join(subdir, fname), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    total_diff = diff.sum(axis=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(years, total_diff, marker='o', markersize=4, linewidth=1.5)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title("Total (sum over sectors)", fontsize=17)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    if output_dir is not None:
        plt.savefig(os.path.join(subdir, f"{fname_prefix}_TOTAL.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


SECTOR_DIFF_PALETTE = [
   "#b4bec5",
   "#ffd7d6",
   "#fa7f69",
   "#2f4370",
   "#3fa2ba",
   "#c7b5ac",
   "#947b85"
]


def plot_Yj_diff_diverging_stacked(df, df_ref, year_cols, subtitle="", output_dir=None, output_path=None):
    """Diverging stacked bar chart of real output (Yj) difference by sector vs baseline.

    X axis: year. Y axis: Δ real output (pYj₀·Yj). Positive sector contributions stack
    upward, negative downward. One bar per year, stacked by sector.
    """
    q_df  = df.loc[df['variable_name']         == "Yj"].reset_index(drop=True)
    q_ref = df_ref.loc[df_ref['variable_name'] == "Yj"].reset_index(drop=True)
    p_base = df_ref.loc[df_ref['variable_name'] == "pYj", year_cols[0]].values[0].astype("float")

    val_df  = q_df[year_cols].values.astype("float")  * p_base
    val_ref = q_ref[year_cols].values.astype("float") * p_base
    diff = val_df - val_ref  # shape: (n_sectors, n_years)

    sector_names = q_df['row_label'].values
    x = np.array(year_cols).astype("int")
    bar_width = (x[1] - x[0]) * 0.8 if len(x) > 1 else 4

    colors = [SECTOR_DIFF_PALETTE[i % len(SECTOR_DIFF_PALETTE)] for i in range(len(sector_names))]

    fig, ax = plt.subplots(figsize=(14, 7))

    pos_bottoms = np.zeros(len(year_cols))
    neg_bottoms = np.zeros(len(year_cols))

    for i, (sector, d) in enumerate(zip(sector_names, diff)):
        pos = np.where(d > 0, d, 0.0)
        neg = np.where(d < 0, d, 0.0)
        ax.bar(x, pos, bottom=pos_bottoms, color=colors[i], label=sector,
               edgecolor='white', linewidth=0.4, width=bar_width)
        ax.bar(x, neg, bottom=neg_bottoms, color=colors[i],
               edgecolor='white', linewidth=0.4, width=bar_width)
        pos_bottoms += pos
        neg_bottoms += neg

    total_diff = diff.sum(axis=0)
    ax.plot(x, total_diff, color='black', linestyle='dashed', linewidth=1.2,
            marker='o', markersize=4, markerfacecolor='black', markeredgecolor='black',
            label='Total', zorder=5)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(year_cols, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Δ Real output (EUR million, 2020 prices)", fontsize=13)
    fig.suptitle("Real output difference between structural change and baseline scenarios, decomposed by sector", fontsize=15)
    if subtitle:
        fig.text(0.5, 0.92, subtitle, ha='center', va='top', fontsize=13)
    ax.legend(loc='upper left', fontsize=11)
    top_margin = 0.88 if subtitle else 0.93
    plt.tight_layout(rect=[0, 0, 1, top_margin])

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
    elif output_dir is not None:
        subdir = os.path.join(output_dir, "Yj_diff_stacked")
        os.makedirs(subdir, exist_ok=True)
        fig.savefig(os.path.join(subdir, "Yj_diff_diverging_stacked.png"), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_aggregate_diff(df, df_ref, year_cols, variable_name, y_label, output_dir=None):
    val_df  = df.loc[df['variable_name']         == variable_name, year_cols].values[0].astype("float")
    val_ref = df_ref.loc[df_ref['variable_name'] == variable_name, year_cols].values[0].astype("float")
    diff  = (val_df - val_ref) / val_ref * 100
    years = [int(c) for c in year_cols]

    fname_prefix = f"diff_{variable_name}"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(years, diff, marker='o', markersize=4, linewidth=1.5)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title(f"Δ {variable_name} vs no-SC", fontsize=17)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    if output_dir is not None:
        subdir = os.path.join(output_dir, fname_prefix)
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, f"{fname_prefix}.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def align_L_to_population(L_array, year_cols,
                           data_dir="Data_postprocessing", base_year="2020"):
    """Scale SCAF L so that L[base_year] equals aggregate WB population[base_year].

    A single multiplicative coefficient  coeff = pop[base_year] / L[base_year]
    is applied to every element of L_array, preserving the growth profile.

    Parameters
    ----------
    L_array   : np.ndarray  — L values for each year in year_cols (model units)
    year_cols : list[str]   — year strings aligned with L_array
    data_dir  : str         — folder containing WB_data/ (default "Data_postprocessing")
    base_year : str         — year in which L and population must match (default "2020")

    Returns
    -------
    np.ndarray — L_array * (pop[base_year] / L[base_year])
    """
    import json

    wb_dir = os.path.join(data_dir, "WB_data")

    with open(os.path.join(wb_dir, "mappings.json"), "r") as fh:
        mappings = json.load(fh)
    agg_codes = mappings["EUR region"]

    pop_df = pd.read_csv(
        os.path.join(wb_dir, "pop.csv"), skiprows=4, index_col="Country Code",
        encoding="utf-8-sig"
    )
    pop_df.columns = [str(c).strip() for c in pop_df.columns]

    pop_base = float(pop_df.loc[agg_codes, base_year].sum())
    L_base   = float(L_array[list(year_cols).index(base_year)])

    return L_array * (pop_base / L_base)


def compute_structural_change_panel(df, year_cols, data_dir="Data_postprocessing"):
    """Compute all structural-change share time series from raw SCAF output.

    Parameters
    ----------
    df        : raw SCAF results DataFrame (variable_name / row_label / year columns)
    year_cols : list of year column strings

    Returns
    -------
    pd.DataFrame  with columns:  indicator | sector | year1 | year2 | …
        indicator values:
            "log_gdp_pc"               — aggregate x-axis (sector = "")
            "Real Value Added Share"
            "Nominal Value Added Share"
            "Nominal Consumption Share"
            "Nominal Output Share"
            "Real Output Share"
    """
    # ---- x-axis ----
    GDPreal    = df.loc[df['variable_name'] == "GDPreal", year_cols].values[0].astype("float")
    L          = df.loc[df['variable_name'] == "L",       year_cols].values[0].astype("float")
    L          = align_L_to_population(L, year_cols, data_dir=data_dir)
    log_gdp_pc = np.log(GDPreal * 1e6 / L)   # GDPreal in M EUR → convert to EUR

    # ---- sector → aggregate mapping ----
    _map_df    = pd.read_csv(os.path.join(data_dir, "mapping", "mapping_sectors.csv"),
                              skipinitialspace=True)
    sec_to_agg = dict(zip(_map_df["sector"].str.strip(), _map_df["aggregate"].str.strip()))
    agg_order  = list(dict.fromkeys(_map_df["aggregate"].str.strip()))  # preserves CSV order

    def _aggregate(levels, sector_names):
        """Sum rows of levels (n_sectors × n_years) by aggregate group.
        Returns (agg_levels shape n_groups × n_years, agg_names list)."""
        result = np.zeros((len(agg_order), levels.shape[1]))
        for i, agg in enumerate(agg_order):
            for j, sec in enumerate(sector_names):
                if sec_to_agg.get(sec) == agg:
                    result[i] += levels[j]
        return result, list(agg_order)

    # ---- share helpers ----
    def _shares_va(src):
        pKLj = src.loc[src['variable_name'] == "pKLj"].reset_index(drop=True)
        KLj  = src.loc[src['variable_name'] == "KLj"].reset_index(drop=True)
        levels = pKLj[year_cols].values.astype("float") * KLj[year_cols].values.astype("float")
        agg_levels, agg_names = _aggregate(levels, pKLj['row_label'].values)
        return agg_levels / agg_levels.sum(axis=0), agg_names

    def _shares_rva(src, pL_base, pK_base):
        Lj = src.loc[src['variable_name'] == "Lj"].reset_index(drop=True)
        Kj = src.loc[src['variable_name'] == "Kj"].reset_index(drop=True)
        levels = (Lj[year_cols].values.astype("float") * pL_base
                + Kj[year_cols].values.astype("float") * pK_base)
        agg_levels, agg_names = _aggregate(levels, Lj['row_label'].values)
        return agg_levels / agg_levels.sum(axis=0), agg_names

    def _shares_cj(src):
        pCj = src.loc[src['variable_name'] == "pCj"].reset_index(drop=True)
        Cj  = src.loc[src['variable_name'] == "Cj"].reset_index(drop=True)
        levels = pCj[year_cols].values.astype("float") * Cj[year_cols].values.astype("float")
        agg_levels, agg_names = _aggregate(levels, pCj['row_label'].values)
        return agg_levels / agg_levels.sum(axis=0), agg_names

    def _shares_out_nom(src):
        pYj = src.loc[src['variable_name'] == "pYj"].reset_index(drop=True)
        Yj  = src.loc[src['variable_name'] == "Yj"].reset_index(drop=True)
        levels = pYj[year_cols].values.astype("float") * Yj[year_cols].values.astype("float")
        agg_levels, agg_names = _aggregate(levels, pYj['row_label'].values)
        return agg_levels / agg_levels.sum(axis=0), agg_names

    def _shares_out_real(src, pYj0):
        Yj   = src.loc[src['variable_name'] == "Yj"].reset_index(drop=True)
        levels = pYj0 * Yj[year_cols].values.astype("float")   # pYj0 shape (n_sectors, 1)
        agg_levels, agg_names = _aggregate(levels, Yj['row_label'].values)
        return agg_levels / agg_levels.sum(axis=0), agg_names

    # base prices (calibration year)
    pL_base = df.loc[df['variable_name'] == "pL", year_cols[0]].values[0].astype("float")
    pK_base = df.loc[df['variable_name'] == "pK", year_cols[0]].values[0].astype("float")
    pYj0    = df.loc[df['variable_name'] == "pYj"].reset_index(drop=True)[year_cols[0]].values.astype("float")[:, np.newaxis]

    va_shares,  va_names  = _shares_va(df)
    rva_shares, rva_names = _shares_rva(df, pL_base, pK_base)
    cj_shares,  cj_names  = _shares_cj(df)
    out_nom,    out_names  = _shares_out_nom(df)
    out_real,   _          = _shares_out_real(df, pYj0)

    rows = [{"indicator": "log_gdp_pc", "sector": "",
              **dict(zip(year_cols, log_gdp_pc))}]

    for shares, names, label in [
        (rva_shares, rva_names, "Real Value Added Share"),
        (va_shares,  va_names,  "Nominal Value Added Share"),
        (cj_shares,  cj_names,  "Nominal Consumption Share"),
        (out_nom,    out_names, "Nominal Output Share"),
        (out_real,   out_names, "Real Output Share"),
    ]:
        for i, sector in enumerate(names):
            rows.append({"indicator": label, "sector": sector,
                         **dict(zip(year_cols, shares[i]))})

    return pd.DataFrame(rows)


def build_joint_structural_change_timeseries(panel_df, wb_nominal_df, wb_real_df,
                                              model_year="2020"):
    """Splice World Bank historical VA-share series (1995–2019) with SCAF model output (2020–2050).

    WB data supplies years strictly before ``model_year``; the full SCAF panel
    (from :func:`compute_structural_change_panel`) supplies all years from
    ``model_year`` onwards.  Log real GDP per capita follows the same split: WB
    values for the historical period, the panel's ``log_gdp_pc`` for the model
    period.

    Parameters
    ----------
    panel_df      : pd.DataFrame — output of ``compute_structural_change_panel``
    wb_nominal_df : pd.DataFrame — first element returned by ``construct_wb_timeseries``
    wb_real_df    : pd.DataFrame — second element returned by ``construct_wb_timeseries``
    model_year    : str, default ``"2020"`` — first year taken from ``panel_df``;
                    WB years strictly below this value form the historical segment

    Returns
    -------
    pd.DataFrame with columns: variable | sector | unit | 1995 | … | 2050
        rows:
            ``value_added_share_nominal``  (agriculture / industry / services)
            ``value_added_share_real``     (agriculture / industry / services)
            ``log_real_gdp_per_capita``    (sector = "all")
    """
    # panel sector label (now lowercase aggregates) → WB sector name (identity)
    _SECTOR_MAP = {
        "agriculture":   "agriculture",
        "industry": "industry",
        "services":      "services",
    }

    # ---- Year column detection ----
    wb_years    = sorted([c for c in wb_nominal_df.columns if c.isdigit() and len(c) == 4])
    pre_years   = [y for y in wb_years if y < model_year]          # WB: 1995–2019
    panel_years = sorted([c for c in panel_df.columns
                           if c.isdigit() and len(c) == 4 and c >= model_year])  # SCAF: 2020–2050

    if not pre_years:
        raise ValueError(
            f"build_joint_structural_change_timeseries: "
            f"no WB years found before model_year={model_year!r}."
        )
    if not panel_years:
        raise ValueError(
            f"build_joint_structural_change_timeseries: "
            f"no panel years found >= model_year={model_year!r}."
        )

    nom_unit  = "fraction of nominal three-sector VA"
    real_unit = "fraction of real three-sector VA"
    gdp_unit  = "log EUR 2020 per person"

    rows = []

    # ---- Nominal VA shares ----
    for scaf_sector, wb_sector in _SECTOR_MAP.items():
        wb_row    = wb_nominal_df.loc[
            (wb_nominal_df["variable"] == "value_added_share_nominal") &
            (wb_nominal_df["sector"]   == wb_sector)
        ]
        panel_row = panel_df.loc[
            (panel_df["indicator"] == "Nominal Value Added Share") &
            (panel_df["sector"]    == scaf_sector)
        ]
        row = {"variable": "value_added_share_nominal", "sector": wb_sector, "unit": nom_unit}
        row.update({y: float(wb_row[y].values[0])    for y in pre_years})
        row.update({y: float(panel_row[y].values[0]) for y in panel_years})
        rows.append(row)

    # ---- Real VA shares ----
    for scaf_sector, wb_sector in _SECTOR_MAP.items():
        wb_row    = wb_real_df.loc[
            (wb_real_df["variable"] == "value_added_share_real") &
            (wb_real_df["sector"]   == wb_sector)
        ]
        panel_row = panel_df.loc[
            (panel_df["indicator"] == "Real Value Added Share") &
            (panel_df["sector"]    == scaf_sector)
        ]
        row = {"variable": "value_added_share_real", "sector": wb_sector, "unit": real_unit}
        row.update({y: float(wb_row[y].values[0])    for y in pre_years})
        row.update({y: float(panel_row[y].values[0]) for y in panel_years})
        rows.append(row)

    # ---- Log real GDP per capita — WB for pre_years, panel for panel_years ----
    gdp_wb    = wb_nominal_df.loc[wb_nominal_df["variable"] == "log_real_gdp_per_capita"]
    gdp_panel = panel_df.loc[panel_df["indicator"] == "log_gdp_pc"]

    # Rescale WB GDP so that WB[model_year] == SCAF[model_year].
    # On log scale this is a constant additive shift = log(SCAF_GDP / WB_GDP) at model_year,
    # equivalent to multiplying every WB GDP level by coeff = SCAF_GDP_2020 / WB_GDP_2020.
    log_shift = (float(gdp_panel[model_year].values[0])
                 - float(gdp_wb[model_year].values[0]))

    gdp_row   = {"variable": "log_real_gdp_per_capita", "sector": "all", "unit": gdp_unit}
    gdp_row.update({y: float(gdp_wb[y].values[0]) + log_shift for y in pre_years})
    gdp_row.update({y: float(gdp_panel[y].values[0]) for y in panel_years})
    rows.append(gdp_row)

    return pd.DataFrame(rows).reset_index(drop=True)


def plot_joint_structural_change_timeseries(joint_df, model_year="2020",
                                             subtitle="", output_dir=None):
    """Two 1×3 panels plotting the spliced WB + SCAF structural-change time series.

    Produces one figure for nominal VA shares and one for real VA shares.
    Each figure has three subplots (agriculture | industry | services) sharing
    a common y-axis.  X-axis is log real GDP per capita (EUR 2020).

    Visual encoding
    ---------------
    Grey dashed line  — all years connected in chronological order
    Blue filled dots  — years before ``model_year`` (World Bank historical data)
    Red filled dots   — ``model_year`` and beyond (SCAF model output)

    Parameters
    ----------
    joint_df   : pd.DataFrame — output of ``build_joint_structural_change_timeseries``
    model_year : str, default ``"2020"`` — first year coloured red
    subtitle   : str — optional subtitle rendered below the main title
    output_dir : str or None — directory to save ``VA_nom_joint.png`` /
                 ``VA_real_joint.png``; if None, calls ``plt.show()``
    """
    SECTORS       = ["agriculture", "industry", "services"]
    SECTOR_LABELS = {"agriculture": "Agriculture",
                     "industry": "Industry",
                     "services":    "Services"}

    # ---- Detect year columns ----
    year_cols   = sorted([c for c in joint_df.columns if c.isdigit() and len(c) == 4])
    pre_years   = [y for y in year_cols if y <  model_year]
    model_years = [y for y in year_cols if y >= model_year]

    # ---- X-axis (log real GDP per capita) ----
    gdp_row  = joint_df.loc[joint_df["variable"] == "log_real_gdp_per_capita"]
    x_all    = gdp_row[year_cols].values[0].astype("float")
    x_pre    = gdp_row[pre_years].values[0].astype("float")   if pre_years   else np.array([])
    x_model  = gdp_row[model_years].values[0].astype("float") if model_years else np.array([])

    for var_name, fig_title in [
        ("value_added_share_nominal", "Nominal Value Added Share"),
        ("value_added_share_real",    "Real Value Added Share"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        top_margin = 0.88 if subtitle else 0.93
        fig.subplots_adjust(top=top_margin, wspace=0.08)
        fig.suptitle(fig_title, fontsize=16, fontweight="bold", y=0.99)
        if subtitle:
            fig.text(0.5, 0.93, subtitle, ha="center", va="top", fontsize=12)

        # Legend label for the SCAF segment
        model_years_str = (f"{model_years[0]}–{model_years[-1]}"
                           if len(model_years) > 1 else str(model_years[0]))

        # Collect handles/labels once for the shared legend
        legend_handles = []

        for col_idx, sector in enumerate(SECTORS):
            ax    = axes[col_idx]
            s_row = joint_df.loc[
                (joint_df["variable"] == var_name) &
                (joint_df["sector"]   == sector)
            ]
            y_all   = s_row[year_cols].values[0].astype("float")
            y_pre   = s_row[pre_years].values[0].astype("float")   if pre_years   else np.array([])
            y_model = s_row[model_years].values[0].astype("float") if model_years else np.array([])

            # Grey connecting line (all years, chronological)
            ax.plot(x_all, y_all, color="grey", linestyle="--", linewidth=1, zorder=1)
            # Blue dots — historical WB data
            h_blue = ax.scatter(x_pre,   y_pre,   color="steelblue", s=25, zorder=2,
                                label=f"1995–{int(model_year)-1} (WB)")
            # Red dots — SCAF model year(s)
            h_red  = ax.scatter(x_model, y_model, color="tomato",    s=50, zorder=3,
                                label=f"{model_years_str} (SCAF)")

            ax.set_title(SECTOR_LABELS[sector], fontsize=13)
            ax.set_xlabel("log real GDP per capita (EUR 2020)", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("share", fontsize=11)

            if col_idx == 0:
                legend_handles = [h_blue, h_red]

        # Shared legend on the last subplot
        axes[-1].legend(handles=legend_handles, fontsize=10, loc="upper left")

        if output_dir is not None:
            fname = "VA_nom_joint.png" if "nominal" in var_name else "VA_real_joint.png"
            plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


def plot_wb_va_shares(wb_nominal_df, wb_real_df,
                      subtitle="", output_dir=None, pad=0.15, regression=True):
    """Plot World Bank historical VA shares vs log real GDP per capita (two 1×3 figures).

    Y-axis: same height across all three sector panels (largest data range wins),
    each panel centred on its own data; lower bound clipped at 0.

    Parameters
    ----------
    wb_nominal_df : first  element of construct_wb_timeseries()
    wb_real_df    : second element of construct_wb_timeseries()
    subtitle      : optional figure subtitle string
    output_dir    : if given, saves "WB_VA_nom.png" and "WB_VA_real.png" there
    pad           : fractional padding added to max_half (default 0.15)
    regression    : if True, overlay an OLS regression line on each panel (default True)
    """
    SECTORS       = ["agriculture", "industry", "services"]
    SECTOR_LABELS = {"agriculture": "Agriculture",
                     "industry": "Industry",
                     "services": "Services"}

    # x-axis: log real GDP per capita from the nominal df
    year_cols = sorted([c for c in wb_nominal_df.columns if c.isdigit() and len(c) == 4])
    gdp_row   = wb_nominal_df.loc[wb_nominal_df["variable"] == "log_real_gdp_per_capita"]
    x         = gdp_row[year_cols].values[0].astype("float")

    for df, var_name, fig_title, fname in [
        (wb_nominal_df, "value_added_share_nominal",
         "Nominal Value Added Share (World Bank historical)", "WB_VA_nom.png"),
        (wb_real_df,    "value_added_share_real",
         "Real Value Added Share (World Bank historical)",    "WB_VA_real.png"),
    ]:
        # ---- compute y-axis limits ----
        series_by_sector = {}
        for sec in SECTORS:
            row = df.loc[(df["variable"] == var_name) & (df["sector"] == sec)]
            series_by_sector[sec] = row[year_cols].values[0].astype("float")

        half_ranges = {sec: (s.max() - s.min()) / 2 for sec, s in series_by_sector.items()}
        centers     = {sec: (s.min() + s.max()) / 2 for sec, s in series_by_sector.items()}
        max_half    = max(half_ranges.values()) * (1 + pad)

        ylims = {}
        for sec in SECTORS:
            y_lo = max(0.0, centers[sec] - max_half)
            y_hi = centers[sec] + max_half
            ylims[sec] = (y_lo, y_hi)

        # ---- plot ----
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))   # no sharey — each sector has its own limits
        top_margin = 0.80 if subtitle else 0.88
        fig.subplots_adjust(top=top_margin, bottom=0.18, wspace=0.30)
        fig.suptitle(fig_title, fontsize=16, fontweight="bold", y=0.99)
        if subtitle:
            fig.text(0.5, 0.90, subtitle, ha="center", va="top", fontsize=12)

        for col_idx, sec in enumerate(SECTORS):
            ax = axes[col_idx]
            y  = series_by_sector[sec]

            ax.scatter(x, y, s=25, color="steelblue")

            if regression:
                coeffs = np.polyfit(x, y, 1)
                x_line = np.array([x.min(), x.max()])
                y_line = np.polyval(coeffs, x_line)
                ax.plot(x_line, y_line, color="firebrick", linewidth=1.5, zorder=1)

            ax.set_title(SECTOR_LABELS[sec], fontsize=13)
            ax.set_xlabel("log real GDP (EUR2020) per capita", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("share over the value added sum", fontsize=10)
            ax.set_ylim(*ylims[sec])

        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


def plot_scaf_va_shares(panel_df, wb_nominal_df, wb_real_df,
                        subtitle="", output_dir=None, pad=0.15, regression=True):
    """Plot SCAF model VA and consumption shares vs log GDP per capita (three 1×3 figures).

    Y-axis interval height is borrowed from the corresponding WB figure so the two sets
    of plots are visually comparable; each panel is centred on the SCAF data.

    Parameters
    ----------
    panel_df      : output of compute_structural_change_panel()
    wb_nominal_df : first  element of construct_wb_timeseries()
    wb_real_df    : second element of construct_wb_timeseries()
    subtitle      : optional figure subtitle string
    output_dir    : if given, saves SCAF_VA_nom.png / SCAF_VA_real.png / SCAF_VA_cons.png
    pad           : fractional padding (default 0.15, must match plot_wb_va_shares call)
    regression    : if True, overlay an OLS regression line on each panel (default True)
    """
    SECTORS       = ["agriculture", "industry", "services"]
    SECTOR_LABELS = {"agriculture": "Agriculture",
                     "industry": "Industry",
                     "services": "Services"}

    # ---- x-axis from SCAF panel ----
    panel_year_cols = [c for c in panel_df.columns if c not in ("indicator", "sector")]
    panel_year_cols = sorted(panel_year_cols, key=int)
    gdp_row = panel_df.loc[panel_df["indicator"] == "log_gdp_pc"]
    x       = gdp_row[panel_year_cols].values[0].astype("float")

    # ---- WB max_half helper ----
    wb_year_cols = sorted([c for c in wb_nominal_df.columns
                           if c.isdigit() and len(c) == 4])

    def _wb_max_half(df, var_name):
        half_ranges = []
        for sec in SECTORS:
            s = df.loc[(df["variable"] == var_name) & (df["sector"] == sec),
                       wb_year_cols].values[0].astype("float")
            half_ranges.append((s.max() - s.min()) / 2)
        return max(half_ranges) * (1 + pad)

    wb_nom_mh  = _wb_max_half(wb_nominal_df, "value_added_share_nominal")
    wb_real_mh = _wb_max_half(wb_real_df,    "value_added_share_real")

    for indicator, fig_title, fname, max_half in [
        ("Nominal Value Added Share",
         "Nominal Value Added Share (SCAF model)", "SCAF_VA_nom.png",  wb_nom_mh),
        ("Real Value Added Share",
         "Real Value Added Share (SCAF model)",    "SCAF_VA_real.png", wb_real_mh),
        ("Nominal Consumption Share",
         "Nominal Consumption Share (SCAF model)", "SCAF_VA_cons.png", wb_nom_mh),
    ]:
        # ---- y-axis limits: SCAF-centred, WB height ----
        series_by_sector = {}
        for sec in SECTORS:
            row = panel_df.loc[(panel_df["indicator"] == indicator)
                               & (panel_df["sector"] == sec)]
            series_by_sector[sec] = row[panel_year_cols].values[0].astype("float")

        ylims = {}
        for sec in SECTORS:
            s        = series_by_sector[sec]
            center_j = (s.min() + s.max()) / 2
            ylims[sec] = (max(0.0, center_j - max_half), center_j + max_half)

        # ---- plot ----
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        top_margin = 0.80 if subtitle else 0.88
        fig.subplots_adjust(top=top_margin, bottom=0.18, wspace=0.30)
        fig.suptitle(fig_title, fontsize=16, fontweight="bold", y=0.99)
        if subtitle:
            fig.text(0.5, 0.90, subtitle, ha="center", va="top", fontsize=12)

        for col_idx, sec in enumerate(SECTORS):
            ax = axes[col_idx]
            y  = series_by_sector[sec]

            ax.scatter(x, y, s=25, color="steelblue")

            if regression:
                coeffs = np.polyfit(x, y, 1)
                x_line = np.array([x.min(), x.max()])
                y_line = np.polyval(coeffs, x_line)
                ax.plot(x_line, y_line, color="firebrick", linewidth=1.5, zorder=1)

            ax.set_title(SECTOR_LABELS[sec], fontsize=13)
            ax.set_xlabel("log real GDP (EUR2020) per capita", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("share over the value added sum", fontsize=10)
            ax.set_ylim(*ylims[sec])

        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


def plot_structural_change_panel(result_df, year_cols, fix_ylim=True, subtitle="", include_real_va=True, nominal_consumption=True, include_output=True, output_dir=None, output_path=None):
    """3×3 panel: VA share, real VA share, Cj share for AGRICULTURE / MANUFACTURE / SERVICES.

    Parameters
    ----------
    result_df : DataFrame returned by compute_structural_change_panel
    year_cols : list of year column strings

    Rows (controlled by flags):
      0 — real VA share      (Lj·pL₀ + Kj·pK₀ / total)  [skipped if include_real_va=False]
      1 — nominal VA share   (pKLj·KLj / total)
      2 — nominal Cj share   (pCj·Cj / total)             [only if nominal_consumption=True]
      3 — nominal output share (pYj·Yj)                   [only if include_output=True]
      4 — real output share  (pYj₀·Yj)                    [only if include_output=True]

    fix_ylim=True  → each subplot y-axis clamped to [0, 0.85]
    fix_ylim=False → y-axis auto-scales
    subtitle       → user-provided subtitle displayed below the main title
    """
    target_sectors = ["AGRICULTURE", "MANUFACTURE", "SERVICES"]

    # ---- x-axis ----
    gdp_row    = result_df[result_df["indicator"] == "log_gdp_pc"]
    log_gdp_pc = gdp_row[year_cols].values[0].astype("float")

    def _get_shares(indicator_label):
        sub    = result_df[result_df["indicator"] == indicator_label]
        names  = sub["sector"].values
        shares = sub[year_cols].values.astype("float")
        return shares, names

    row_data = [
        (*_get_shares("Real Value Added Share"),    "Real Value Added Share"),
        (*_get_shares("Nominal Value Added Share"), "Nominal Value Added Share"),
    ]
    if not include_real_va:
        row_data = [row_data[1]]
    if nominal_consumption:
        row_data.append((*_get_shares("Nominal Consumption Share"), "Nominal Consumption Share"))
    if include_output:
        row_data += [
            (*_get_shares("Nominal Output Share"), "Nominal output share\n(pYj·Yj)"),
            (*_get_shares("Real Output Share"),    "Real output share\n(pYj₀·Yj)"),
        ]

    n_rows = len(row_data)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    top_margin = 0.91 if subtitle else 0.94
    fig.subplots_adjust(top=top_margin, hspace=0.35, wspace=0.3)
    fig.suptitle("Structural change indicators", fontsize=18, fontweight='bold', y=0.99)
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha='center', va='top', fontsize=13)

    for row_idx, (shares, names, row_label) in enumerate(row_data):
        for col_idx, sector in enumerate(target_sectors):
            ax = axes[row_idx, col_idx]
            idx = np.where(names == sector)[0]
            if idx.size > 0:
                ax.plot(log_gdp_pc, shares[idx[0]], marker='o', markersize=4, linewidth=1.5)
            if fix_ylim:
                ax.set_ylim(0, 0.85)
            # column title on top row only
            if row_idx == 0:
                ax.set_title(sector, fontsize=14)
            # row label on leftmost column only
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=12)
            # x-label on bottom row only
            if row_idx == n_rows - 1:
                ax.set_xlabel("log(GDP per capita)", fontsize=11)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
    elif output_dir is not None:
        subdir = os.path.join(output_dir, "structural_change_panel")
        os.makedirs(subdir, exist_ok=True)
        suffix = "" if include_real_va else "_no_real_va"
        fname = f"structural_change_panel{suffix}.png" if fix_ylim else f"structural_change_panel_free_ylim{suffix}.png"
        plt.savefig(os.path.join(subdir, fname), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_structural_change_panel_diff(df, df_ref, year_cols, subtitle="", nominal_consumption=False, include_output=True, output_dir=None, output_path=None):
    """3×3 panel of *differences* (df minus df_ref) for the three structural-change
    indicators across AGRICULTURE / MANUFACTURE / SERVICES.

    Rows:
      0 — Δ nominal VA share  (pKLj·KLj / total)
      1 — Δ real VA share     (Lj·pL₀ + Kj·pK₀ / total)  — uses df's base prices
      2 — Δ nominal Cj share  (pCj·Cj / total)

    x-axis: log(GDP per capita) from df (the scenario run).
    y-axis: auto-scales; a dashed zero line is drawn on each subplot.
    """
    target_sectors = ["AGRICULTURE", "MANUFACTURE", "SERVICES"]

    # ---- shared x-axis from df ----
    GDPreal    = df.loc[df['variable_name'] == "GDPreal", year_cols].values[0].astype("float")
    L          = df.loc[df['variable_name'] == "L",       year_cols].values[0].astype("float")
    log_gdp_pc = np.log(GDPreal / L)

    def _shares_va(src):
        pKLj = src.loc[src['variable_name'] == "pKLj"].reset_index(drop=True)
        KLj  = src.loc[src['variable_name'] == "KLj"].reset_index(drop=True)
        nom  = pKLj[year_cols].values.astype("float") * KLj[year_cols].values.astype("float")
        return nom / nom.sum(axis=0), pKLj['row_label'].values

    def _shares_rva(src, pL_base, pK_base):
        Lj = src.loc[src['variable_name'] == "Lj"].reset_index(drop=True)
        Kj = src.loc[src['variable_name'] == "Kj"].reset_index(drop=True)
        nom = (Lj[year_cols].values.astype("float") * pL_base
             + Kj[year_cols].values.astype("float") * pK_base)
        return nom / nom.sum(axis=0), Lj['row_label'].values

    def _shares_cj(src):
        pCj = src.loc[src['variable_name'] == "pCj"].reset_index(drop=True)
        Cj  = src.loc[src['variable_name'] == "Cj"].reset_index(drop=True)
        nom = pCj[year_cols].values.astype("float") * Cj[year_cols].values.astype("float")
        return nom / nom.sum(axis=0), pCj['row_label'].values

    def _shares_out_nom(src):
        pYj = src.loc[src['variable_name'] == "pYj"].reset_index(drop=True)
        Yj  = src.loc[src['variable_name'] == "Yj"].reset_index(drop=True)
        nom = pYj[year_cols].values.astype("float") * Yj[year_cols].values.astype("float")
        return nom / nom.sum(axis=0), pYj['row_label'].values

    def _shares_out_real(src, pYj0):
        Yj   = src.loc[src['variable_name'] == "Yj"].reset_index(drop=True)
        real = pYj0 * Yj[year_cols].values.astype("float")
        return real / real.sum(axis=0), Yj['row_label'].values

    # base prices from df (calibration year)
    pL_base = df.loc[df['variable_name'] == "pL", year_cols[0]].values[0].astype("float")
    pK_base = df.loc[df['variable_name'] == "pK", year_cols[0]].values[0].astype("float")
    pYj0    = df.loc[df['variable_name'] == "pYj"].reset_index(drop=True)[year_cols[0]].values.astype("float")[:, np.newaxis]

    va_shares_df,  va_names   = _shares_va(df)
    va_shares_ref, _          = _shares_va(df_ref)
    rva_shares_df,  rva_names = _shares_rva(df,     pL_base, pK_base)
    rva_shares_ref, _         = _shares_rva(df_ref, pL_base, pK_base)
    cj_shares_df,  cj_names   = _shares_cj(df)
    cj_shares_ref, _          = _shares_cj(df_ref)
    out_nom_df,  out_names    = _shares_out_nom(df)
    out_nom_ref, _            = _shares_out_nom(df_ref)
    out_real_df, _            = _shares_out_real(df,     pYj0)
    out_real_ref, _           = _shares_out_real(df_ref, pYj0)

    row_data = [
        (rva_shares_df - rva_shares_ref, rva_names, "Δ Real Value Added Share"),
        (va_shares_df  - va_shares_ref,  va_names,  "Δ Nominal Value Added Share"),
    ]
    if nominal_consumption:
        row_data.append(
            (cj_shares_df - cj_shares_ref, cj_names, "Δ Nominal Consumption Share")
        )
    if include_output:
        row_data += [
            (out_nom_df  - out_nom_ref,  out_names, "Δ nominal output share"),
            (out_real_df - out_real_ref, out_names, "Δ real output share"),
        ]

    n_rows = len(row_data)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    top_margin = 0.91 if subtitle else 0.94
    fig.subplots_adjust(top=top_margin, hspace=0.35, wspace=0.3)
    fig.suptitle(
        "Difference in structural change indicators between structural change scenario and baseline",
        fontsize=16, fontweight='bold', y=0.99)
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha='center', va='top', fontsize=13)

    for row_idx, (diff_shares, names, row_label) in enumerate(row_data):
        for col_idx, sector in enumerate(target_sectors):
            ax = axes[row_idx, col_idx]
            idx = np.where(names == sector)[0]
            if idx.size > 0:
                ax.plot(log_gdp_pc, diff_shares[idx[0]], marker='o', markersize=4, linewidth=1.5)
            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(min(ylo, -0.001), max(yhi, 0.001))
            if row_idx == 0:
                ax.set_title(sector, fontsize=14)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=12)
            if row_idx == n_rows - 1:
                ax.set_xlabel("log(GDP per capita)", fontsize=11)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
    elif output_dir is not None:
        subdir = os.path.join(output_dir, "structural_change_panel")
        os.makedirs(subdir, exist_ok=True)
        suffix = "" if include_real_va else "_no_real_va"
        plt.savefig(os.path.join(subdir, f"structural_change_panel_diff{suffix}.png"), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_structural_change_panels_split(df, df_ref, year_cols, subtitle="", output_dir=None):
    """Saves 5 separate 1×3 figures of share differences (df minus df_ref):
    SC_diff_real_VA.png, SC_diff_nom_VA.png, SC_diff_nom_C.png,
    SC_diff_nom_L.png, SC_diff_real_L.png
    """
    target_sectors = ["AGRICULTURE", "MANUFACTURE", "SERVICES"]

    GDPreal    = df.loc[df['variable_name'] == "GDPreal", year_cols].values[0].astype("float")
    L_agg      = df.loc[df['variable_name'] == "L",       year_cols].values[0].astype("float")
    log_gdp_pc = np.log(GDPreal / L_agg)

    pL_base = df.loc[df['variable_name'] == "pL", year_cols[0]].values[0].astype("float")
    pK_base = df.loc[df['variable_name'] == "pK", year_cols[0]].values[0].astype("float")

    def _va(src):
        p = src.loc[src['variable_name'] == "pKLj"].reset_index(drop=True)
        q = src.loc[src['variable_name'] == "KLj"].reset_index(drop=True)
        nom = p[year_cols].values.astype("float") * q[year_cols].values.astype("float")
        return nom / nom.sum(axis=0), p['row_label'].values

    def _rva(src):
        Lj = src.loc[src['variable_name'] == "Lj"].reset_index(drop=True)
        Kj = src.loc[src['variable_name'] == "Kj"].reset_index(drop=True)
        nom = (Lj[year_cols].values.astype("float") * pL_base
             + Kj[year_cols].values.astype("float") * pK_base)
        return nom / nom.sum(axis=0), Lj['row_label'].values

    def _cj(src):
        p = src.loc[src['variable_name'] == "pCj"].reset_index(drop=True)
        q = src.loc[src['variable_name'] == "Cj"].reset_index(drop=True)
        nom = p[year_cols].values.astype("float") * q[year_cols].values.astype("float")
        return nom / nom.sum(axis=0), p['row_label'].values

    def _empl_nom(src):
        Lj = src.loc[src['variable_name'] == "Lj"].reset_index(drop=True)
        pL = src.loc[src['variable_name'] == "pL", year_cols].values.astype("float")
        nom = Lj[year_cols].values.astype("float") * pL
        return nom / nom.sum(axis=0), Lj['row_label'].values

    def _empl_real(src):
        Lj = src.loc[src['variable_name'] == "Lj"].reset_index(drop=True)
        nom = Lj[year_cols].values.astype("float") * pL_base
        return nom / nom.sum(axis=0), Lj['row_label'].values

    rva_df,  rva_names  = _rva(df);       rva_ref,  _ = _rva(df_ref)
    va_df,   va_names   = _va(df);        va_ref,   _ = _va(df_ref)
    cj_df,   cj_names   = _cj(df);        cj_ref,   _ = _cj(df_ref)
    enl_df,  enl_names  = _empl_nom(df);  enl_ref,  _ = _empl_nom(df_ref)
    erl_df,  erl_names  = _empl_real(df); erl_ref,  _ = _empl_real(df_ref)

    plots = [
        (rva_df - rva_ref, rva_names, "real value added",    "SC_diff_real_VA.png"),
        (va_df  - va_ref,  va_names,  "nominal value added", "SC_diff_nom_VA.png"),
        (cj_df  - cj_ref,  cj_names,  "nominal consumption", "SC_diff_nom_C.png"),
        (enl_df - enl_ref, enl_names, "nominal employment",  "SC_diff_nom_L.png"),
        (erl_df - erl_ref, erl_names, "real employment",     "SC_diff_real_L.png"),
    ]

    escaped = subtitle.replace(' ', r'\ ')
    for diff_shares, names, label, fname in plots:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.subplots_adjust(top=0.85, wspace=0.3)
        if subtitle:
            title = rf"Difference of {label} shares between $\mathit{{{escaped}}}$ scenario and baseline"
        else:
            title = f"Difference of {label} shares between scenario and baseline"
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)

        for col_idx, sector in enumerate(target_sectors):
            ax = axes[col_idx]
            idx = np.where(names == sector)[0]
            if idx.size > 0:
                ax.plot(log_gdp_pc, diff_shares[idx[0]], marker='o', markersize=4, linewidth=1.5)
            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(min(ylo, -0.001), max(yhi, 0.001))
            ax.set_title(sector, fontsize=14)
            ax.set_xlabel("log(GDP per capita)", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"Δ {label} share", fontsize=12)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_energy_volumes_comparison_by_use(df, REMIND_E_volumes, year_cols, scaf_label="SCAF", df_no_sc=None, output_dir=None):
    evol_rows = df.loc[df['variable_name'] == "E_vol"].reset_index(drop=True)
    remind_year_cols = [c for c in year_cols if c in REMIND_E_volumes.columns]

    scaf_by_use = (
        evol_rows.groupby('col_label')[year_cols]
        .sum()
        .astype("float")
    )
    remind_by_use = (
        REMIND_E_volumes.groupby('Energy uses')[remind_year_cols]
        .sum()
        .astype("float")
    )
    if df_no_sc is not None:
        no_sc_by_use = (
            df_no_sc.loc[df_no_sc['variable_name'] == "E_vol"]
            .groupby('col_label')[year_cols]
            .sum()
            .astype("float")
        )

    x = np.array(remind_year_cols).astype('int')

    evol_subdir = os.path.join(output_dir, "E_vol") if output_dir is not None else None
    if evol_subdir is not None:
        os.makedirs(evol_subdir, exist_ok=True)

    def _save_or_show(fig, fname):
        if output_dir is not None:
            plt.savefig(os.path.join(evol_subdir, fname), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    for use in scaf_by_use.index:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, scaf_by_use.loc[use, remind_year_cols].values,
                color='red', linewidth=2, label=scaf_label)
        if df_no_sc is not None and use in no_sc_by_use.index:
            ax.plot(x, no_sc_by_use.loc[use, remind_year_cols].values,
                    color='blue', linewidth=2, label='Baseline')
        ax.plot(x, remind_by_use.loc[use].values,
                color='grey', linestyle='--', linewidth=2, label='REMIND')
        ax.set_title(f"Energy demand volume: {use}", fontsize=17)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Energy demand volume (EJ)", fontsize=14)
        ax.legend(loc='upper right', fontsize=13)
        safe_use = use.replace('&', 'and').replace(' ', '_')
        _save_or_show(fig, f"E_vol_{safe_use}.png")


def plot_total_energy_volume_comparison(df, REMIND_E_volumes, year_cols, include_PE=False, scaf_label="SCAF", df_no_sc=None, output_dir=None, output_path=None):
    evol_rows = df.loc[df['variable_name'] == "E_vol"].reset_index(drop=True)
    remind_year_cols = [c for c in year_cols if c in REMIND_E_volumes.columns]

    scaf_by_use = (
        evol_rows.groupby('col_label')[year_cols]
        .sum()
        .astype("float")
    )
    remind_by_use = (
        REMIND_E_volumes.groupby('Energy uses')[remind_year_cols]
        .sum()
        .astype("float")
    )
    if df_no_sc is not None:
        no_sc_by_use = (
            df_no_sc.loc[df_no_sc['variable_name'] == "E_vol"]
            .groupby('col_label')[year_cols]
            .sum()
            .astype("float")
        )

    x = np.array(remind_year_cols).astype('int')

    evol_subdir = os.path.join(output_dir, "E_vol") if output_dir is not None else None
    if evol_subdir is not None:
        os.makedirs(evol_subdir, exist_ok=True)

    def _save_or_show(fig, fname):
        if output_dir is not None:
            plt.savefig(os.path.join(evol_subdir, fname), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def _total(by_use):
        filtered = by_use if include_PE else by_use.drop(index="PE", errors="ignore")
        return filtered[remind_year_cols].sum(axis=0).values
    
    scaf_total   = _total(scaf_by_use)
    remind_total = _total(remind_by_use)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, scaf_total, color='red', linewidth=2, label=scaf_label)
    if df_no_sc is not None:
        no_sc_total = _total(no_sc_by_use)
        ax.plot(x, no_sc_total, color='blue', linewidth=2, label='Baseline')
    ax.plot(x, remind_total, color='grey', linestyle='--', linewidth=2, label='REMIND')
    ax.set_title("Evolution of total energy demand by scenario", fontsize=17)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Energy demand (EJ)", fontsize=14)
    ax.legend(loc='upper right', fontsize=13)
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
    else:
        _save_or_show(fig, "E_vol_total.png")


def export_E_Demand_tot_csv(df, year_cols, scaf_label="SCAF", df_no_sc=None, output_dir=None):
    evol_rows = df.loc[df['variable_name'] == "E_vol"].reset_index(drop=True)
    scaf_by_use = evol_rows.groupby('col_label')[year_cols].sum().astype("float")
    scaf_total = scaf_by_use.sum(axis=0)

    no_sc_by_use = (
        df_no_sc.loc[df_no_sc['variable_name'] == "E_vol"]
        .groupby('col_label')[year_cols]
        .sum()
        .astype("float")
    )
    no_sc_total = no_sc_by_use.sum(axis=0)

    diff = scaf_total - no_sc_total
    ratio = diff / no_sc_total

    result = pd.DataFrame({
        "baseline": no_sc_total,
        scaf_label: scaf_total,
        "difference": diff,
        "percentage difference": ratio,
    }).T
    result.index.name = None

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        result.to_csv(os.path.join(output_dir, "E_Demand_tot.csv"))

    return result


def plot_energy_volumes_diverging_stacked(df, REMIND_E_volumes, year_cols, scaf_label="SCAF", output_dir=None, output_path=None):
    """Diverging stacked bar chart of the SCAF − REMIND energy volume gap, decomposed by energy type.

    One bar per year in the time series. For each year, positive contributions (SCAF > REMIND)
    are stacked upward from zero and negative contributions (SCAF < REMIND) downward.
    All energy uses (including PE) are included.
    """
    evol_rows = df.loc[df['variable_name'] == "E_vol"].reset_index(drop=True)
    remind_year_cols = [c for c in year_cols if c in REMIND_E_volumes.columns]

    scaf_by_use = evol_rows.groupby('col_label')[year_cols].sum().astype("float")
    remind_by_use = (
        REMIND_E_volumes.groupby('Energy uses')[remind_year_cols]
        .sum()
        .astype("float")
    )

    uses = [u for u in scaf_by_use.index if u in remind_by_use.index]

    diff = np.array([
        scaf_by_use.loc[u, remind_year_cols].values - remind_by_use.loc[u].values
        for u in uses
    ])

    colors = [ENERGY_USE_PALETTE[i % len(ENERGY_USE_PALETTE)] for i in range(len(uses))]

    x = np.array(remind_year_cols).astype('int')
    bar_width = (x[1] - x[0]) * 0.8 if len(x) > 1 else 4

    fig, ax = plt.subplots(figsize=(14, 7))

    pos_bottoms = np.zeros(len(remind_year_cols))
    neg_bottoms = np.zeros(len(remind_year_cols))

    for i, (use, d) in enumerate(zip(uses, diff)):
        pos = np.where(d > 0, d, 0.0)
        neg = np.where(d < 0, d, 0.0)
        ax.bar(x, pos, bottom=pos_bottoms, color=colors[i], label=use,
               edgecolor='white', linewidth=0.4, width=bar_width)
        ax.bar(x, neg, bottom=neg_bottoms, color=colors[i],
               edgecolor='white', linewidth=0.4, width=bar_width)
        pos_bottoms += pos
        neg_bottoms += neg

    total_diff = diff.sum(axis=0)
    ax.plot(x, total_diff, color='black', linestyle='dashed', linewidth=1.5,
            marker='o', markersize=4, label='Total')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(remind_year_cols, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel(f"Δ Energy volume (EJ)", fontsize=13)
    ax.set_title(f"Energy volume difference between SCAF baseline and REMIND, decomposed by energy use", fontsize=15)
    ax.legend(loc='upper left', fontsize=11)
    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
    elif output_dir is not None:
        subdir = output_dir
        os.makedirs(subdir, exist_ok=True)
        fig.savefig(os.path.join(subdir, "E_vol_diverging_stacked.png"), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_energy_volumes_diverging_stacked_scaf_diff(df, df_baseline, year_cols, subtitle="", output_dir=None, output_path=None):
    """Diverging stacked bar chart of the SCAF scenario − SCAF baseline energy volume gap, decomposed by energy type.

    One bar per year. Positive contributions (scenario > baseline) stack upward from zero,
    negative contributions (scenario < baseline) stack downward. All energy uses included.
    """
    def _by_use(data):
        return (
            data.loc[data['variable_name'] == "E_vol"]
            .groupby('col_label')[year_cols]
            .sum()
            .astype("float")
        )

    scaf_by_use      = _by_use(df)
    baseline_by_use  = _by_use(df_baseline)

    uses = [u for u in scaf_by_use.index if u in baseline_by_use.index]

    diff = np.array([
        scaf_by_use.loc[u].values - baseline_by_use.loc[u].values
        for u in uses
    ])

    colors = [ENERGY_USE_PALETTE[i % len(ENERGY_USE_PALETTE)] for i in range(len(uses))]

    x = np.array(year_cols).astype('int')
    bar_width = (x[1] - x[0]) * 0.8 if len(x) > 1 else 4

    fig, ax = plt.subplots(figsize=(14, 7))

    pos_bottoms = np.zeros(len(year_cols))
    neg_bottoms = np.zeros(len(year_cols))

    for i, (use, d) in enumerate(zip(uses, diff)):
        pos = np.where(d > 0, d, 0.0)
        neg = np.where(d < 0, d, 0.0)
        ax.bar(x, pos, bottom=pos_bottoms, color=colors[i], label=use,
               edgecolor='white', linewidth=0.4, width=bar_width)
        ax.bar(x, neg, bottom=neg_bottoms, color=colors[i],
               edgecolor='white', linewidth=0.4, width=bar_width)
        pos_bottoms += pos
        neg_bottoms += neg

    total_diff = diff.sum(axis=0)
    ax.plot(x, total_diff, color='black', linestyle='dashed', linewidth=1.2,
            marker='o', markersize=4, markerfacecolor='black', markeredgecolor='black',
            label='Total', zorder=5)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(year_cols, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Δ Energy volume (EJ)", fontsize=13)
    fig.suptitle("Energy volume difference between structural change scenario and baseline, decomposed by energy use", fontsize=15)
    if subtitle:
        fig.text(0.5, 0.92, subtitle, ha='center', va='top', fontsize=13)
    ax.legend(loc='upper left', fontsize=11)
    top_margin = 0.88 if subtitle else 0.93
    plt.tight_layout(rect=[0, 0, 1, top_margin])

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
    elif output_dir is not None:
        subdir = os.path.join(output_dir, "E_vol")
        os.makedirs(subdir, exist_ok=True)
        fig.savefig(os.path.join(subdir, "E_vol_diverging_stacked_scaf_diff.png"), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_energy_volumes_by_consumer(df, year_cols, output_dir=None):
    """One plot per energy consumer: total E_vol summed over all energy uses."""
    evol_rows = df.loc[df['variable_name'] == "E_vol"].reset_index(drop=True)

    by_consumer = (
        evol_rows.groupby('row_label')[year_cols]
        .sum()
        .astype("float")
    )

    x = np.array(year_cols).astype('int')

    subdir = os.path.join(output_dir, "E_vol_by_consumer") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    for consumer in by_consumer.index:
        vals = by_consumer.loc[consumer].values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, vals, linewidth=2)
        ax.set_title(f"Energy volume: {consumer}", fontsize=17)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Energy volume (EJ)", fontsize=14)
        plt.xlim(x[0] - 0.01, x[-1] + 0.01)
        if output_dir is not None:
            safe = consumer.replace('&', 'and').replace(' ', '_')
            plt.savefig(os.path.join(subdir, f"E_vol_consumer_{safe}.png"), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_Yj_vs_REMIND_output(df, REMIND_output, year_cols, output_dir=None):
    # Mapping: SCAF sector row_label -> REMIND Variable string
    sector_remind_map = {
        "STEEL":       "Production|Industry|Steel",
        "CHEMICAL":    "Value Added|Industry|Chemicals",
        "MANUFACTURE": "Value Added|Industry|Other Industry",
    }

    yj_rows = df.loc[df['variable_name'] == "Yj"].reset_index(drop=True)

    # Restrict to years present in both datasets
    common_years = [c for c in year_cols if c in REMIND_output.columns]
    x = np.array(common_years).astype('int')

    for sector, remind_var in sector_remind_map.items():
        scaf_row = yj_rows.loc[yj_rows['row_label'] == sector, common_years]
        if scaf_row.empty:
            continue
        scaf_vals = scaf_row.values[0].astype("float")
        scaf_vals = scaf_vals / scaf_vals[0]

        remind_row = REMIND_output.loc[REMIND_output['Variable'] == remind_var, common_years]
        if remind_row.empty:
            continue
        remind_vals = remind_row.values[0].astype("float")
        remind_vals = remind_vals / remind_vals[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, scaf_vals,   color='red',  linewidth=2, label='SCAF')
        ax.plot(x, remind_vals, color='blue', linewidth=2, label='REMIND')
        ax.set_title(f"Output growth: {sector}", fontsize=17)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Normalised output (2020 = 1)", fontsize=14)
        ax.legend(loc='upper right', fontsize=13)

        if output_dir is not None:
            yj_subdir = os.path.join(output_dir, "Yj_vs_REMIND")
            os.makedirs(yj_subdir, exist_ok=True)
            plt.savefig(os.path.join(yj_subdir, f"Yj_vs_REMIND_{sector}.png"), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_sector_Sj_Yj(df, year_cols, sector, output_dir=None):
    x = np.array(year_cols).astype('int')

    def _get(var_name):
        row = df.loc[
            (df['variable_name'] == var_name) & (df['row_label'] == sector),
            year_cols
        ]
        return row.values[0].astype("float") if not row.empty else None

    data = {label: _get(label) for label in ["Sj", "pSj", "Yj", "pYj"]}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(sector, fontsize=18)

    for ax, (label, vals) in zip(axes.flat, data.items()):
        if vals is not None:
            ax.plot(x, vals / vals[0], marker='o', markersize=3, linewidth=1.5)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Relative change (2020 = 1)", fontsize=12)

    plt.tight_layout()

    if output_dir is not None:
        subdir = os.path.join(output_dir, "Sj_pSj_Yj_pYj")
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, f"Sj_pSj_Yj_pYj_{sector}.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_sector_Sj_Yj_diff(df, df_ref, year_cols, sector, output_dir=None):
    x = np.array(year_cols).astype('int')

    def _get(src, var_name):
        row = src.loc[
            (src['variable_name'] == var_name) & (src['row_label'] == sector),
            year_cols
        ]
        return row.values[0].astype("float") if not row.empty else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{sector} — Δ vs no-SC (relative change)", fontsize=18)

    for ax, label in zip(axes.flat, ["Sj", "pSj", "Yj", "pYj"]):
        v_sc = _get(df, label)
        v_ref = _get(df_ref, label)
        if v_sc is not None and v_ref is not None:
            diff = v_sc / v_sc[0] - v_ref / v_ref[0]
            ax.plot(x, diff, marker='o', markersize=3, linewidth=1.5)
            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Δ relative change (2020 = 1)", fontsize=12)

    plt.tight_layout()

    if output_dir is not None:
        subdir = os.path.join(output_dir, "Sj_pSj_Yj_pYj_diff")
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, f"Sj_pSj_Yj_pYj_diff_{sector}.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_real_va_output_diff(df, df_ref, year_cols, output_dir=None):
    x = [int(c) for c in year_cols]

    pL_base = df.loc[df['variable_name'] == "pL", year_cols[0]].values[0].astype("float")
    pK_base = df.loc[df['variable_name'] == "pK", year_cols[0]].values[0].astype("float")

    Lj_sc  = df.loc[df['variable_name']     == "Lj"].reset_index(drop=True)
    Kj_sc  = df.loc[df['variable_name']     == "Kj"].reset_index(drop=True)
    Yj_sc  = df.loc[df['variable_name']     == "Yj"].reset_index(drop=True)
    pYj_sc = df.loc[df['variable_name']     == "pYj"].reset_index(drop=True)
    Lj_ref = df_ref.loc[df_ref['variable_name'] == "Lj"].reset_index(drop=True)
    Kj_ref = df_ref.loc[df_ref['variable_name'] == "Kj"].reset_index(drop=True)
    Yj_ref = df_ref.loc[df_ref['variable_name'] == "Yj"].reset_index(drop=True)

    sector_names = Lj_sc['row_label'].values

    real_va_sc  = (Lj_sc[year_cols].values.astype("float")  * pL_base
                 + Kj_sc[year_cols].values.astype("float")  * pK_base)
    real_va_ref = (Lj_ref[year_cols].values.astype("float") * pL_base
                 + Kj_ref[year_cols].values.astype("float") * pK_base)
    diff_va = real_va_sc - real_va_ref

    pYj0 = pYj_sc[year_cols[0]].values.astype("float")[:, np.newaxis]
    real_out_sc  = Yj_sc[year_cols].values.astype("float")  * pYj0
    real_out_ref = Yj_ref[year_cols].values.astype("float") * pYj0
    diff_out = real_out_sc - real_out_ref

    subdir = os.path.join(output_dir, "diff_real_va_output") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    for j, sector in enumerate(sector_names):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{sector} — Δ real VA and output vs no-SC", fontsize=18)

        ax = axes[0]
        ax.plot(x, diff_va[j], marker='o', markersize=4, linewidth=1.5)
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_title("Real VA (pL₀·Lj + pK₀·Kj)", fontsize=14)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Δ real value added", fontsize=14)

        ax = axes[1]
        ax.plot(x, diff_out[j], marker='o', markersize=4, linewidth=1.5)
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_title("Real output (pYj₀·Yj)", fontsize=14)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Δ real output", fontsize=14)

        plt.tight_layout()
        if subdir is not None:
            fname = f"diff_real_va_output_{sector.replace(' ', '_')}.png"
            plt.savefig(os.path.join(subdir, fname), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Total economy — Δ real VA and output vs no-SC", fontsize=18)

    ax = axes[0]
    ax.plot(x, diff_va.sum(axis=0), marker='o', markersize=4, linewidth=1.5)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title("Σ Real VA (Σ pL₀·Lj + pK₀·Kj)", fontsize=14)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Δ total real value added", fontsize=14)

    ax = axes[1]
    ax.plot(x, diff_out.sum(axis=0), marker='o', markersize=4, linewidth=1.5)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title("Σ Real output (Σ pYj₀·Yj)", fontsize=14)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Δ total real output", fontsize=14)

    plt.tight_layout()
    if subdir is not None:
        plt.savefig(os.path.join(subdir, "diff_real_va_output_total.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_real_va_vs_gdp(df, year_cols, output_dir=None):
    x = [int(c) for c in year_cols]

    pL_base = df.loc[df['variable_name'] == "pL", year_cols[0]].values[0].astype("float")
    pK_base = df.loc[df['variable_name'] == "pK", year_cols[0]].values[0].astype("float")

    Lj = df.loc[df['variable_name'] == "Lj"][year_cols].values.astype("float")
    Kj = df.loc[df['variable_name'] == "Kj"][year_cols].values.astype("float")
    real_va  = (pL_base * Lj + pK_base * Kj).sum(axis=0)

    gdp_real = df.loc[df['variable_name'] == "GDPreal", year_cols].values[0].astype("float")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Total real VA vs real GDP", fontsize=18)

    ax = axes[0]
    ax.plot(x, real_va  / real_va[0],  marker='o', markersize=4, linewidth=1.5, label="Total real VA")
    ax.plot(x, gdp_real / gdp_real[0], marker='o', markersize=4, linewidth=1.5, linestyle='--', label="Real GDP")
    ax.set_title("Normalized trajectories (year₀ = 1)", fontsize=14)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Index (year₀ = 1)", fontsize=14)
    ax.legend()

    ax = axes[1]
    ax.plot(x, real_va - gdp_real, marker='o', markersize=4, linewidth=1.5)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title("Difference (real VA − real GDP)", fontsize=14)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Real VA − real GDP", fontsize=14)

    plt.tight_layout()

    if output_dir is not None:
        subdir = os.path.join(output_dir, "real_va_vs_gdp")
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, "real_va_vs_gdp.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_energy_expenditure_by_sector(df, year_cols, normalise=True, output_dir=None):
    """Plot pY_Ej * Yij(ENERGY -> j) for each sector j.

    normalise=True  → each series divided by its calibration-year value (relative change).
    normalise=False → absolute nominal values.
    """
    x = np.array(year_cols).astype('int')

    pY_Ej_rows = df.loc[df['variable_name'] == "pY_Ej"].reset_index(drop=True)
    Yij_energy_rows = df.loc[
        (df['variable_name'] == "Yij") & (df['row_label'] == "ENERGY")
    ].reset_index(drop=True)

    # Align on col_label (consuming sector) for Yij and row_label for pY_Ej
    sectors = pY_Ej_rows['row_label'].values
    expenditure = []
    for sector in sectors:
        pY_E = pY_Ej_rows.loc[pY_Ej_rows['row_label'] == sector, year_cols].values
        Y_E  = Yij_energy_rows.loc[Yij_energy_rows['col_label'] == sector, year_cols].values
        if pY_E.size == 0 or Y_E.size == 0:
            expenditure.append(None)
        else:
            expenditure.append((pY_E[0] * Y_E[0]).astype("float"))

    fig, ax = plt.subplots(figsize=(14, 8))

    for j, (sector, vals) in enumerate(zip(sectors, expenditure)):
        if vals is None:
            continue
        y = vals / vals[0] if normalise else vals
        ax.plot(x, y, label=sector, color=my_cmap(j), linewidth=1.5, marker='o', markersize=3)
        ax.annotate(text=sector, xy=(x[-1], y[-1]), xytext=(5, 0),
                    textcoords='offset points', va='center', fontsize=10)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 12})
    ax.set_title("Nominal energy expenditure by sector (pY_Ej × Yij[ENERGY→j])", fontsize=16)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Relative change (2020 = 1)" if normalise else "Nominal value", fontsize=14)
    plt.xlim(int(year_cols[0]) - 0.01, int(year_cols[-1]) + 0.01)

    if output_dir is not None:
        subdir = os.path.join(output_dir, "energy_expenditure_by_sector")
        os.makedirs(subdir, exist_ok=True)
        fname = "energy_expenditure_by_sector.png" if normalise else "energy_expenditure_by_sector_absolute.png"
        plt.savefig(os.path.join(subdir, fname), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_energy_expenditure_share(df, year_cols, output_dir=None):
    """One plot per sector: (pY_Ej * Yij[ENERGY→j]) / (pYj * Yj) over time."""
    x = np.array(year_cols).astype('int')

    pY_Ej_rows      = df.loc[df['variable_name'] == "pY_Ej"].reset_index(drop=True)
    Yij_energy_rows = df.loc[
        (df['variable_name'] == "Yij") & (df['row_label'] == "ENERGY")
    ].reset_index(drop=True)
    Yj_rows  = df.loc[df['variable_name'] == "Yj"].reset_index(drop=True)
    pYj_rows = df.loc[df['variable_name'] == "pYj"].reset_index(drop=True)

    subdir = os.path.join(output_dir, "energy_expenditure_share") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    sectors = pY_Ej_rows['row_label'].values
    for j, sector in enumerate(sectors):
        pY_E = pY_Ej_rows.loc[pY_Ej_rows['row_label'] == sector, year_cols].values
        Y_E  = Yij_energy_rows.loc[Yij_energy_rows['col_label'] == sector, year_cols].values
        Yj   = Yj_rows.loc[Yj_rows['row_label'] == sector, year_cols].values
        pYj  = pYj_rows.loc[pYj_rows['row_label'] == sector, year_cols].values

        if any(arr.size == 0 for arr in (pY_E, Y_E, Yj, pYj)):
            continue

        numerator   = (pY_E[0] * Y_E[0]).astype("float")
        denominator = (pYj[0]  * Yj[0]).astype("float")
        ratio = numerator / denominator

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, ratio, color=my_cmap(j), linewidth=1.5, marker='o', markersize=3)
        ax.set_title(f"Energy expenditure share — {sector}", fontsize=16)
        ax.set_xlabel("Year", fontsize=13)
        ax.set_ylabel("pY_Ej · Yij[ENERGY→j]  /  pYj · Yj", fontsize=12)
        plt.xlim(x[0] - 0.01, x[-1] + 0.01)
        plt.tight_layout()

        if subdir is not None:
            plt.savefig(os.path.join(subdir, f"energy_expenditure_share_{sector}.png"), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_nominal_demand_evolutions(df, year_cols, max_year="2035", output_dir=None):
    """Normalised evolution of pXj*Xj, pMj*Mj, pXj*Xj-pMj*Mj, pCj*Cj, pIj*Ij, pGj*Gj.
    One plot per quantity, all non-energy sectors, up to max_year.
    Falls back to pCj for pIj/pGj if those variables are absent.
    """
    filtered_years = [y for y in year_cols if int(y) <= int(max_year)]
    x = np.array(filtered_years).astype('int')

    def _rows(var_name):
        r = df.loc[df['variable_name'] == var_name].reset_index(drop=True)
        return r if not r.empty else None

    def _rows_or_fallback(primary, fallback):
        r = _rows(primary)
        return r if r is not None else _rows(fallback)

    def _nominal(p_rows, q_rows):
        """Return OrderedDict sector -> array[filtered_years], energy excluded."""
        if p_rows is None or q_rows is None:
            return {}
        out = {}
        for sector in sectors_names_eng:
            if sector.upper() == "ENERGY":
                continue
            p = p_rows.loc[p_rows['row_label'] == sector, filtered_years].values
            q = q_rows.loc[q_rows['row_label'] == sector, filtered_years].values
            if p.size == 0 or q.size == 0:
                continue
            out[sector] = (p[0] * q[0]).astype("float")
        return out

    pXj = _rows("pXj");          Xj  = _rows("Xj")
    pMj = _rows("pMj");          Mj  = _rows("Mj")
    pCj = _rows("pCj");          Cj  = _rows("Cj")
    pIj = _rows_or_fallback("pIj", "pCj");  Ij  = _rows("Ij")
    pGj = _rows_or_fallback("pGj", "pCj");  Gj  = _rows("Gj")

    nom_X = _nominal(pXj, Xj)
    nom_M = _nominal(pMj, Mj)
    nom_C = _nominal(pCj, Cj)
    nom_I = _nominal(pIj, Ij)
    nom_G = _nominal(pGj, Gj)

    # Net exports: compute for sectors present in both X and M
    nom_NX = {}
    for sector in sectors_names_eng:
        if sector.upper() == "ENERGY":
            continue
        if sector in nom_X and sector in nom_M:
            nom_NX[sector] = nom_X[sector] - nom_M[sector]

    subdir = os.path.join(output_dir, "nominal_demand_evolutions") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    def _plot_one(data_dict, title, fname):
        fig, ax = plt.subplots(figsize=(14, 8))
        for sector, vals in data_dict.items():
            if vals[0] == 0:
                continue
            j = sectors_names_eng.index(sector) if sector in sectors_names_eng else 0
            y = vals / vals[0]
            ax.plot(x, y, label=sector, color=my_cmap(j), linewidth=1.5)
            ax.annotate(text=sector, xy=(x[-1], y[-1]), xytext=(5, 0),
                        textcoords='offset points', va='center', fontsize=10)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 12})
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Year", fontsize=13)
        ax.set_ylabel("Relative change (calibration year = 1)", fontsize=12)
        plt.xlim(x[0] - 0.01, x[-1] + 0.01)
        plt.tight_layout()
        if subdir is not None:
            plt.savefig(os.path.join(subdir, fname), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    _plot_one(nom_X,  "Nominal exports pXj·Xj (excl. energy)",              "pXj_Xj.png")
    _plot_one(nom_M,  "Nominal imports pMj·Mj (excl. energy)",              "pMj_Mj.png")
    _plot_one(nom_NX, "Nominal net exports pXj·Xj − pMj·Mj (excl. energy)", "pXj_Xj_minus_pMj_Mj.png")
    _plot_one(nom_C,  "Nominal consumption pCj·Cj (excl. energy)",          "pCj_Cj.png")
    _plot_one(nom_I,  "Nominal investment pIj·Ij (excl. energy)",           "pIj_Ij.png")
    _plot_one(nom_G,  "Nominal government expenditure pGj·Gj (excl. energy)", "pGj_Gj.png")


def plot_net_exports(df, year_cols, df_ref=None, output_dir=None):
    """Per-sector time series of real and nominal net exports (or their diff vs df_ref).

    When df_ref is None: plots pXj·(Xj−Mj) and pXj0·(Xj−Mj) in levels.
    When df_ref is provided: plots Δ[pXj·(Xj−Mj)] and Δ[pXj0·(Xj−Mj)] vs the reference.
    pXj0 is the base-year (first-period) export price taken from df_ref when available, else from df.
    """
    x = np.array(year_cols).astype('int')

    def _rows(src, var):
        return src.loc[src['variable_name'] == var].reset_index(drop=True)

    pXj_rows = _rows(df, 'pXj');  Xj_rows = _rows(df, 'Xj');  Mj_rows = _rows(df, 'Mj')

    is_diff = df_ref is not None
    if is_diff:
        pXj_ref = _rows(df_ref, 'pXj')
        Xj_ref  = _rows(df_ref, 'Xj')
        Mj_ref  = _rows(df_ref, 'Mj')

    subdir = os.path.join(output_dir, "net_exports") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    def _save_or_show(fig, fname):
        if subdir is not None:
            plt.savefig(os.path.join(subdir, fname), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    total_nominal = np.zeros(len(year_cols))
    total_real    = np.zeros(len(year_cols))

    for sector in sectors_names_eng:
        if sector.upper() == "ENERGY":
            continue

        p_row = pXj_rows.loc[pXj_rows['row_label'] == sector, year_cols]
        q_row = Xj_rows.loc[Xj_rows['row_label']   == sector, year_cols]
        m_row = Mj_rows.loc[Mj_rows['row_label']   == sector, year_cols]

        if p_row.empty or q_row.empty or m_row.empty:
            continue

        pXj_ts = p_row.values[0].astype('float')
        Xj_ts  = q_row.values[0].astype('float')
        Mj_ts  = m_row.values[0].astype('float')

        if is_diff:
            p_ref_row = pXj_ref.loc[pXj_ref['row_label'] == sector, year_cols]
            q_ref_row = Xj_ref.loc[Xj_ref['row_label']   == sector, year_cols]
            m_ref_row = Mj_ref.loc[Mj_ref['row_label']   == sector, year_cols]
            if p_ref_row.empty or q_ref_row.empty or m_ref_row.empty:
                continue
            pXj_ref_ts = p_ref_row.values[0].astype('float')
            Xj_ref_ts  = q_ref_row.values[0].astype('float')
            Mj_ref_ts  = m_ref_row.values[0].astype('float')
            pXj0       = pXj_ref_ts[0]
            nominal_NX = pXj_ts * (Xj_ts - Mj_ts) - pXj_ref_ts * (Xj_ref_ts - Mj_ref_ts)
            real_NX    = pXj0 * ((Xj_ts - Mj_ts) - (Xj_ref_ts - Mj_ref_ts))
        else:
            pXj0       = pXj_ts[0]
            nominal_NX = pXj_ts * (Xj_ts - Mj_ts)
            real_NX    = pXj0   * (Xj_ts - Mj_ts)

        total_nominal += nominal_NX
        total_real    += real_NX

        ylabel = ("Δ net exports (model units)" if is_diff else "Net exports (model units)")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, nominal_NX, marker='o', markersize=4, linewidth=1.5,
                label='Nominal pXj·(Xj−Mj)')
        ax.plot(x, real_NX,    marker='o', markersize=4, linewidth=1.5, linestyle='dashed',
                label='Real pXj₀·(Xj−Mj)')
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_title(sector, fontsize=17)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=12)
        _save_or_show(fig, f"net_exports_{sector.replace(' ', '_')}.png")

    ylabel_total = ("Δ net exports (model units)" if is_diff else "Net exports (model units)")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, total_nominal, marker='o', markersize=4, linewidth=1.5,
            label='Nominal pXj·(Xj−Mj)')
    ax.plot(x, total_real,    marker='o', markersize=4, linewidth=1.5, linestyle='dashed',
            label='Real pXj₀·(Xj−Mj)')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title("Total (sum over sectors, excl. energy)", fontsize=17)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel(ylabel_total, fontsize=14)
    ax.legend(fontsize=12)
    _save_or_show(fig, "net_exports_TOTAL.png")


def plot_export_share_of_output(df, year_cols, output_dir=None):
    """One plot per sector: (pXj * Xj) / (pYj * Yj) over time."""
    x = np.array(year_cols).astype('int')

    Xj_rows  = df.loc[df['variable_name'] == "Xj"].reset_index(drop=True)
    pXj_rows = df.loc[df['variable_name'] == "pXj"].reset_index(drop=True)
    Yj_rows  = df.loc[df['variable_name'] == "Yj"].reset_index(drop=True)
    pYj_rows = df.loc[df['variable_name'] == "pYj"].reset_index(drop=True)

    subdir = os.path.join(output_dir, "export_share_of_output") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    sectors = Xj_rows['row_label'].values
    for j, sector in enumerate(sectors):
        Xj  = Xj_rows.loc[Xj_rows['row_label']   == sector, year_cols].values
        pXj = pXj_rows.loc[pXj_rows['row_label']  == sector, year_cols].values
        Yj  = Yj_rows.loc[Yj_rows['row_label']    == sector, year_cols].values
        pYj = pYj_rows.loc[pYj_rows['row_label']  == sector, year_cols].values

        if any(arr.size == 0 for arr in (Xj, pXj, Yj, pYj)):
            continue

        ratio = (pXj[0] * Xj[0]).astype("float") / (pYj[0] * Yj[0]).astype("float")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, ratio, color=my_cmap(j), linewidth=1.5, marker='o', markersize=3)
        ax.set_title(f"Export share of nominal output — {sector}", fontsize=16)
        ax.set_xlabel("Year", fontsize=13)
        ax.set_ylabel("pXj · Xj  /  pYj · Yj", fontsize=12)
        plt.xlim(x[0] - 0.01, x[-1] + 0.01)
        plt.tight_layout()

        if subdir is not None:
            plt.savefig(os.path.join(subdir, f"export_share_of_output_{sector}.png"), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_real_export_share_of_output(df, year_cols, output_dir=None):
    """One plot per sector: Xj / Yj (real exports over real output) over time."""
    x = np.array(year_cols).astype('int')

    Xj_rows = df.loc[df['variable_name'] == "Xj"].reset_index(drop=True)
    Yj_rows = df.loc[df['variable_name'] == "Yj"].reset_index(drop=True)

    subdir = os.path.join(output_dir, "real_export_share_of_output") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    sectors = Xj_rows['row_label'].values
    for j, sector in enumerate(sectors):
        Xj = Xj_rows.loc[Xj_rows['row_label'] == sector, year_cols].values
        Yj = Yj_rows.loc[Yj_rows['row_label'] == sector, year_cols].values

        if any(arr.size == 0 for arr in (Xj, Yj)):
            continue

        ratio = Xj[0].astype("float") / Yj[0].astype("float")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, ratio, color=my_cmap(j), linewidth=1.5, marker='o', markersize=3)
        ax.set_title(f"Real export share of output — {sector}", fontsize=16)
        ax.set_xlabel("Year", fontsize=13)
        ax.set_ylabel("Xj / Yj", fontsize=12)
        plt.xlim(x[0] - 0.01, x[-1] + 0.01)
        plt.tight_layout()

        if subdir is not None:
            plt.savefig(os.path.join(subdir, f"real_export_share_of_output_{sector}.png"), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_export_share_of_output_diff(df, df_ref, year_cols, output_dir=None):
    """One plot per sector: Δ(pXj₀·Xj_t / pYj₀·Yj_t) between df and df_ref, using base-year prices from df_ref."""
    x = np.array(year_cols).astype('int')

    Xj_rows   = df.loc[df['variable_name']         == "Xj"].reset_index(drop=True)
    Yj_rows   = df.loc[df['variable_name']         == "Yj"].reset_index(drop=True)
    Xj_rows_r = df_ref.loc[df_ref['variable_name'] == "Xj"].reset_index(drop=True)
    Yj_rows_r = df_ref.loc[df_ref['variable_name'] == "Yj"].reset_index(drop=True)
    pXj_rows_r = df_ref.loc[df_ref['variable_name'] == "pXj"].reset_index(drop=True)
    pYj_rows_r = df_ref.loc[df_ref['variable_name'] == "pYj"].reset_index(drop=True)

    subdir = os.path.join(output_dir, "export_share_of_output_diff") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    sectors = Xj_rows['row_label'].values
    for j, sector in enumerate(sectors):
        Xj    = Xj_rows.loc[Xj_rows['row_label']       == sector, year_cols].values
        Yj    = Yj_rows.loc[Yj_rows['row_label']       == sector, year_cols].values
        Xj_r  = Xj_rows_r.loc[Xj_rows_r['row_label']  == sector, year_cols].values
        Yj_r  = Yj_rows_r.loc[Yj_rows_r['row_label']  == sector, year_cols].values
        pXj0  = pXj_rows_r.loc[pXj_rows_r['row_label'] == sector, year_cols[0]].values
        pYj0  = pYj_rows_r.loc[pYj_rows_r['row_label'] == sector, year_cols[0]].values

        if any(arr.size == 0 for arr in (Xj, Yj, Xj_r, Yj_r, pXj0, pYj0)):
            continue

        pXj0 = float(pXj0[0])
        pYj0 = float(pYj0[0])
        ratio_df  = (pXj0 * Xj[0].astype("float"))  / (pYj0 * Yj[0].astype("float"))
        ratio_ref = (pXj0 * Xj_r[0].astype("float")) / (pYj0 * Yj_r[0].astype("float"))
        diff = ratio_df - ratio_ref

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, diff, color=my_cmap(j), linewidth=1.5, marker='o', markersize=3)
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_title(f"Δ real export share of output — {sector}", fontsize=16)
        ax.set_xlabel("Year", fontsize=13)
        ax.set_ylabel("Δ(pXj₀·Xj_t / pYj₀·Yj_t)", fontsize=12)
        plt.xlim(x[0] - 0.01, x[-1] + 0.01)
        plt.tight_layout()

        if subdir is not None:
            plt.savefig(os.path.join(subdir, f"export_share_of_output_diff_{sector}.png"), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_pYj_over_pXj_services_diff(df, df_ref, year_cols, output_dir=None):
    sector = "SERVICES"
    x = np.array(year_cols).astype('int')

    pYj_df  = df.loc[(df['variable_name']  == "pYj") & (df['row_label']  == sector), year_cols].values.astype("float").flatten()
    pXj_df  = df.loc[(df['variable_name']  == "pXj") & (df['row_label']  == sector), year_cols].values.astype("float").flatten()
    pYj_ref = df_ref.loc[(df_ref['variable_name'] == "pYj") & (df_ref['row_label'] == sector), year_cols].values.astype("float").flatten()
    pXj_ref = df_ref.loc[(df_ref['variable_name'] == "pXj") & (df_ref['row_label'] == sector), year_cols].values.astype("float").flatten()

    ratio_df  = pYj_df  / pXj_df
    ratio_ref = pYj_ref / pXj_ref
    diff_pct  = (ratio_df - ratio_ref) / ratio_ref * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, diff_pct, marker='o', markersize=4, linewidth=1.5)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title(f"Δ pYj/pXj ({sector}) vs no-SC", fontsize=16)
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("% difference", fontsize=13)
    plt.tight_layout()

    if output_dir is not None:
        subdir = os.path.join(output_dir, "pYj_over_pXj_services_diff")
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, "pYj_over_pXj_services_diff.png"), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_real_aggregate_components_stacked(df, year_cols, by_sector=False, normalized=False, output_dir=None):
    x  = np.array(year_cols).astype('int')
    t0 = year_cols[0]

    # Per-sector real values: shape (n_sectors, n_years)
    # Gj and Ij are deflated by pCj in this model (no separate pGj/pIj deflator)
    Cj_rows = df.loc[df['variable_name'] == "Cj"].reset_index(drop=True)
    Gj_rows = df.loc[df['variable_name'] == "Gj"].reset_index(drop=True)
    Ij_rows = df.loc[df['variable_name'] == "Ij"].reset_index(drop=True)
    sectors = Cj_rows['row_label'].values

    pCj0 = df.loc[df['variable_name'] == "pCj", t0].values.astype("float")

    real_C_ps = pCj0[:, None] * Cj_rows[year_cols].values.astype("float")
    real_G_ps = pCj0[:, None] * Gj_rows[year_cols].values.astype("float")
    real_I_ps = pCj0[:, None] * Ij_rows[year_cols].values.astype("float")

    io_real = compute_IO_monetary(df, year_cols, nominal=False)
    io_ps = np.array([
        io_real.loc[io_real['row_label'] == s, year_cols].values.astype("float").sum(axis=0)
        for s in sectors
    ])

    comp_labels = ["real Cj", "real Gj", "real Ij", "real IO"]
    colors    = plt.cm.tab10(np.linspace(0, 0.4, 4))
    bar_width = 0.6

    def _plot_one(components_4, title, fname):
        stacked = np.stack(components_4, axis=0)  # (4, n_years)
        ylabel  = "Real value (base-year prices)"
        if normalized:
            total   = stacked.sum(axis=0, keepdims=True)
            stacked = stacked / total * 100
            ylabel  = "Share (%)"

        fig, ax = plt.subplots(figsize=(12, 7))
        bottoms = np.zeros(len(year_cols))
        for vals, label, color in zip(stacked, comp_labels, colors):
            ax.bar(x, vals, bottom=bottoms, color=color, label=label,
                   edgecolor='white', linewidth=0.5, width=bar_width)
            bottoms += vals

        if normalized:
            ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(year_cols, rotation=45, ha='right', fontsize=11)
        ax.set_xlabel("Year", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=15)
        ax.legend(loc='upper left', fontsize=11)
        plt.tight_layout()

        if output_dir is not None:
            subdir = os.path.join(output_dir, "real_aggregate_components")
            os.makedirs(subdir, exist_ok=True)
            plt.savefig(os.path.join(subdir, fname), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    norm_tag = "_normalized" if normalized else ""
    if by_sector:
        for j, sector in enumerate(sectors):
            _plot_one(
                [real_C_ps[j], real_G_ps[j], real_I_ps[j], io_ps[j]],
                title=f"Real aggregate components — {sector}",
                fname=f"by_sector_{sector}{norm_tag}.png",
            )
    else:
        _plot_one(
            [real_C_ps.sum(axis=0), real_G_ps.sum(axis=0), real_I_ps.sum(axis=0), io_ps.sum(axis=0)],
            title="Real aggregate components over time",
            fname=f"aggregate{norm_tag}.png",
        )


def plot_real_aggregate_components_diff(df, df_ref, year_cols, by_sector=False, output_dir=None):
    x  = np.array(year_cols).astype('int')
    t0 = year_cols[0]

    # Use df_ref's base-year pCj prices for both (consistent with other diff functions)
    pCj0_ref = df_ref.loc[df_ref['variable_name'] == "pCj", t0].values.astype("float")

    def _real_ps(src, var):
        rows = src.loc[src['variable_name'] == var].reset_index(drop=True)
        return pCj0_ref[:, None] * rows[year_cols].values.astype("float")

    sectors = df.loc[df['variable_name'] == "Cj", 'row_label'].values

    # Per-sector real components for df and df_ref: (n_sectors, n_years)
    diff_C = _real_ps(df, "Cj") - _real_ps(df_ref, "Cj")
    diff_G = _real_ps(df, "Gj") - _real_ps(df_ref, "Gj")
    diff_I = _real_ps(df, "Ij") - _real_ps(df_ref, "Ij")

    io_df  = compute_IO_monetary(df,     year_cols, nominal=False)
    io_ref = compute_IO_monetary(df_ref, year_cols, nominal=False)
    diff_IO_ps = np.array([
        (io_df.loc[io_df['row_label']   == s, year_cols].values.astype("float").sum(axis=0) -
         io_ref.loc[io_ref['row_label'] == s, year_cols].values.astype("float").sum(axis=0))
        for s in sectors
    ])

    comp_labels = ["real Cj", "real Gj", "real Ij", "real IO"]
    colors    = plt.cm.tab10(np.linspace(0, 0.4, 4))
    bar_width = 0.6

    def _plot_one(diffs_4, title, fname):
        fig, ax = plt.subplots(figsize=(12, 7))
        pos_bottoms = np.zeros(len(year_cols))
        neg_bottoms = np.zeros(len(year_cols))

        for d, label, color in zip(diffs_4, comp_labels, colors):
            pos = np.where(d > 0, d, 0.0)
            neg = np.where(d < 0, d, 0.0)
            ax.bar(x, pos, bottom=pos_bottoms, color=color, label=label,
                   edgecolor='white', linewidth=0.5, width=bar_width)
            ax.bar(x, neg, bottom=neg_bottoms, color=color,
                   edgecolor='white', linewidth=0.5, width=bar_width)
            pos_bottoms += pos
            neg_bottoms += neg

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(year_cols, rotation=45, ha='right', fontsize=11)
        ax.set_xlabel("Year", fontsize=13)
        ax.set_ylabel("Results − No-SC (base-year prices)", fontsize=13)
        ax.set_title(title, fontsize=15)
        ax.legend(loc='upper left', fontsize=11)
        plt.tight_layout()

        if output_dir is not None:
            subdir = os.path.join(output_dir, "real_aggregate_components")
            os.makedirs(subdir, exist_ok=True)
            plt.savefig(os.path.join(subdir, fname), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    if by_sector:
        for j, sector in enumerate(sectors):
            _plot_one(
                [diff_C[j], diff_G[j], diff_I[j], diff_IO_ps[j]],
                title=f"Δ real aggregate components vs no-SC — {sector}",
                fname=f"diff_by_sector_{sector}.png",
            )
    else:
        _plot_one(
            [diff_C.sum(axis=0), diff_G.sum(axis=0), diff_I.sum(axis=0), diff_IO_ps.sum(axis=0)],
            title="Δ real aggregate components vs no-SC",
            fname="diff_aggregate.png",
        )


def plot_pY_Ej(df, year_cols, output_dir=None):
    """One plot per sector showing the evolution of pY_Ej (energy price faced by sector j)."""
    x = np.array(year_cols).astype('int')

    pY_Ej_rows = df.loc[df['variable_name'] == "pY_Ej"].reset_index(drop=True)

    subdir = os.path.join(output_dir, "pY_Ej") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    for j, row in pY_Ej_rows.iterrows():
        sector = row['row_label']
        vals = row[year_cols].values.astype("float")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"pY_Ej — {sector}", fontsize=16)

        axes[0].plot(x, vals, color=my_cmap(j), linewidth=1.5, marker='o', markersize=3)
        axes[0].set_title("Absolute", fontsize=13)
        axes[0].set_xlabel("Year", fontsize=12)
        axes[0].set_ylabel("pY_Ej", fontsize=12)

        axes[1].plot(x, vals / vals[0], color=my_cmap(j), linewidth=1.5, marker='o', markersize=3)
        axes[1].set_title("Normalised (2020 = 1)", fontsize=13)
        axes[1].set_xlabel("Year", fontsize=12)
        axes[1].set_ylabel("Relative change (2020 = 1)", fontsize=12)

        plt.tight_layout()

        if output_dir is not None:
            plt.savefig(os.path.join(subdir, f"pY_Ej_{sector}.png"), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_energy_sector_inputs(df, year_cols, output_dir=None):
    """Plot pSj[sector] * Yij(sector -> ENERGY) for each supplying sector, normalised to 2020=1."""
    x = np.array(year_cols).astype('int')

    pSj_rows = df.loc[df['variable_name'] == "pSj"].reset_index(drop=True)
    Yij_to_energy = df.loc[
        (df['variable_name'] == "Yij") & (df['col_label'] == "ENERGY")
    ].reset_index(drop=True)

    sectors = pSj_rows['row_label'].values
    inputs = []
    for sector in sectors:
        pS  = pSj_rows.loc[pSj_rows['row_label'] == sector, year_cols].values
        Y   = Yij_to_energy.loc[Yij_to_energy['row_label'] == sector, year_cols].values
        if pS.size == 0 or Y.size == 0:
            inputs.append(None)
        else:
            inputs.append((pS[0] * Y[0]).astype("float"))

    fig, ax = plt.subplots(figsize=(14, 8))

    for j, (sector, vals) in enumerate(zip(sectors, inputs)):
        if vals is None:
            continue
        y = vals / vals[0]
        ax.plot(x, y, label=sector, color=my_cmap(j), linewidth=1.5, marker='o', markersize=3)
        ax.annotate(text=sector, xy=(x[-1], y[-1]), xytext=(5, 0),
                    textcoords='offset points', va='center', fontsize=10)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 12})
    ax.set_title("Nominal intermediate consumption of the energy sector by input (pSj × Yij[j→ENERGY])", fontsize=15)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Relative change (2020 = 1)", fontsize=14)
    plt.xlim(int(year_cols[0]) - 0.01, int(year_cols[-1]) + 0.01)

    if output_dir is not None:
        subdir = os.path.join(output_dir, "energy_sector_inputs")
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, "energy_sector_inputs.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_KL_shares_by_sector(df, year_cols, output_dir=None):
    """Stacked bar chart of capital and labour income shares for year 0.

    For each of AGRICULTURE, MANUFACTURE, SERVICES:
      capital share = pK * Kj / (pK * Kj + pL * Lj)
      labour  share = pL * Lj / (pK * Kj + pL * Lj)
    """
    year = year_cols[0]
    target_sectors = ["AGRICULTURE", "MANUFACTURE", "SERVICES"]

    pK = df.loc[df['variable_name'] == "pK", year].values[0].astype("float")
    pL = df.loc[df['variable_name'] == "pL", year].values[0].astype("float")

    Kj_rows = df.loc[df['variable_name'] == "Kj"].reset_index(drop=True)
    Lj_rows = df.loc[df['variable_name'] == "Lj"].reset_index(drop=True)

    capital_shares = []
    labour_shares  = []
    for sector in target_sectors:
        Kj = Kj_rows.loc[Kj_rows['row_label'] == sector, year].values[0].astype("float")
        Lj = Lj_rows.loc[Lj_rows['row_label'] == sector, year].values[0].astype("float")
        total = pK * Kj + pL * Lj
        capital_shares.append(pK * Kj / total)
        labour_shares.append(pL * Lj / total)

    x = np.arange(len(target_sectors))
    capital_color = "#4C72B0"
    labour_color  = "#DD8452"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, capital_shares, color=capital_color, label="Capital share  pK·Kj / (pK·Kj + pL·Lj)")
    ax.bar(x, labour_shares,  bottom=capital_shares, color=labour_color,
           label="Labour share  pL·Lj / (pK·Kj + pL·Lj)")

    ax.set_xticks(x)
    ax.set_xticklabels(target_sectors, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share", fontsize=13)
    ax.set_title(f"Capital and labour income shares by sector (year {year})", fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()

    if output_dir is not None:
        subdir = os.path.join(output_dir, "KL_shares_by_sector")
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, f"KL_shares_by_sector_{year}.png"), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_demand_components_stacked(df, year_cols, year=None, output_dir=None):
    """Two stacked bar charts (absolute and normalised) of demand components per sector.

    Components: pXj*Xj (exports), pCj*Cj (consumption), pIj*Ij (investment), pGj*Gj (government).
    Falls back to pCj when pIj or pGj are not present in the data.
    One bar per sector (7 sectors), ordered as in sectors_names_eng.
    """
    if year is None:
        year = year_cols[0]

    def _sector_vals(var_name):
        """Return {sector: value} for the requested year, or {} if variable absent."""
        rows = df.loc[df['variable_name'] == var_name].reset_index(drop=True)
        if rows.empty:
            return {}
        return dict(zip(rows['row_label'].values, rows[year].values.astype("float")))

    def _sector_vals_with_fallback(primary, fallback):
        d = _sector_vals(primary)
        return d if d else _sector_vals(fallback)

    pXj = _sector_vals("pXj")
    Xj  = _sector_vals("Xj")
    pCj = _sector_vals("pCj")
    Cj  = _sector_vals("Cj")
    pIj = _sector_vals_with_fallback("pIj", "pCj")
    Ij  = _sector_vals("Ij")
    pGj = _sector_vals_with_fallback("pGj", "pCj")
    Gj  = _sector_vals("Gj")

    def _product(p_dict, q_dict, sector):
        return p_dict.get(sector, 0.0) * q_dict.get(sector, 0.0)

    comp_labels = ["pXj·Xj", "pCj·Cj", "pIj·Ij", "pGj·Gj"]
    comp_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    comp_getters = [
        (pXj, Xj),
        (pCj, Cj),
        (pIj, Ij),
        (pGj, Gj),
    ]

    n_sectors = len(sectors_names_eng)
    # shape: (4 components, 7 sectors)
    comp_values = np.array([
        [_product(p, q, s) for s in sectors_names_eng]
        for p, q in comp_getters
    ])

    x = np.arange(n_sectors)

    subdir = os.path.join(output_dir, "demand_components_stacked") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    # ---- Chart 1: Absolute ----
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    bottoms = np.zeros(n_sectors)
    for i, (label, vals) in enumerate(zip(comp_labels, comp_values)):
        ax1.bar(x, vals, bottom=bottoms, color=comp_colors[i], label=label, edgecolor='white', linewidth=0.5)
        bottoms += vals

    ax1.set_xticks(x)
    ax1.set_xticklabels(sectors_names_eng, rotation=30, ha='right', fontsize=11)
    ax1.set_ylabel("Nominal value", fontsize=13)
    ax1.set_title(f"Demand components by sector — absolute ({year})", fontsize=15)
    ax1.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    if subdir is not None:
        fig1.savefig(os.path.join(subdir, f"demand_components_absolute_{year}.png"), bbox_inches='tight')
        plt.close(fig1)
    else:
        plt.show()

    # ---- Chart 2: Normalised (100 % stacked) ----
    totals = comp_values.sum(axis=0)          # (n_sectors,)
    # guard against zero totals
    safe_totals = np.where(totals == 0, 1.0, totals)
    comp_shares = comp_values / safe_totals[np.newaxis, :] * 100  # percentage

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    bottoms = np.zeros(n_sectors)
    for i, (label, shares) in enumerate(zip(comp_labels, comp_shares)):
        ax2.bar(x, shares, bottom=bottoms, color=comp_colors[i], label=label, edgecolor='white', linewidth=0.5)
        bottoms += shares

    ax2.set_xticks(x)
    ax2.set_xticklabels(sectors_names_eng, rotation=30, ha='right', fontsize=11)
    ax2.set_ylabel("Share (%)", fontsize=13)
    ax2.set_ylim(0, 100)
    ax2.set_title(f"Demand components by sector — normalised ({year})", fontsize=15)
    ax2.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    if subdir is not None:
        fig2.savefig(os.path.join(subdir, f"demand_components_normalised_{year}.png"), bbox_inches='tight')
        plt.close(fig2)
    else:
        plt.show()


def compute_IO_monetary(df, year_cols, nominal):
    """Convert Yij volume flows to monetary flows using gross prices.

    nominal=True  → each Yij[i,j,t] multiplied by the gross price of year t.
    nominal=False → each Yij[i,j,t] multiplied by the gross price of the first year (t0).

    Gross prices:
      Non-energy rows: pSj[row_label, t] * (1 + tauSj[row_label, t])
      Energy row:      pY_Ej[col_label, t] * (1 + tauSj[ENERGY, t])

    Returns a dataframe with the same structure as SCAF_results, preserving
    row_label, col_label, and year columns. variable_name is set to
    'IO_monetary_nominal' or 'IO_monetary_real'.
    """
    var_name = "IO_monetary_nominal" if nominal else "IO_monetary_real"
    t0 = year_cols[0]

    Yij_rows    = df.loc[df['variable_name'] == "Yij"].reset_index(drop=True)
    pSj_rows    = df.loc[df['variable_name'] == "pSj"].reset_index(drop=True)
    tauSj_rows  = df.loc[df['variable_name'] == "tauSj"].reset_index(drop=True)
    pY_Ej_rows  = df.loc[df['variable_name'] == "pY_Ej"].reset_index(drop=True)

    # tauSj[ENERGY] used as the tax factor for all energy prices
    tauSj_E_match = tauSj_rows.loc[tauSj_rows['row_label'] == "ENERGY", year_cols]

    result_records = []
    for _, row in Yij_rows.iterrows():
        row_lbl = row['row_label']
        col_lbl = row['col_label']
        Y_vals  = row[year_cols].values.astype("float")

        if row_lbl != "ENERGY":
            p_match    = pSj_rows.loc[pSj_rows['row_label'] == row_lbl, year_cols]
            tau_match  = tauSj_rows.loc[tauSj_rows['row_label'] == row_lbl, year_cols]
            if p_match.empty or tau_match.empty:
                continue
            if nominal:
                price = p_match.values[0].astype("float") * (1 + tau_match.values[0].astype("float"))
            else:
                price = float(p_match[t0].values[0]) * (1 + float(tau_match[t0].values[0]))
        else:
            p_match = pY_Ej_rows.loc[pY_Ej_rows['row_label'] == col_lbl, year_cols]
            if p_match.empty or tauSj_E_match.empty:
                continue
            if nominal:
                price = p_match.values[0].astype("float") * (1 + tauSj_E_match.values[0].astype("float"))
            else:
                price = float(p_match[t0].values[0]) * (1 + float(tauSj_E_match[t0].values[0]))

        monetary_vals = Y_vals * price
        record = {
            'variable_name': var_name,
            'row_label':     row_lbl,
            'col_label':     col_lbl,
            'status':        row['status'],
        }
        for col, val in zip(year_cols, monetary_vals):
            record[col] = val
        result_records.append(record)

    meta_cols = ['variable_name', 'row_label', 'col_label', 'status']
    return pd.DataFrame(result_records, columns=meta_cols + year_cols)


def plot_real_IO_monetary_diff(df, df_no_sc, year_cols, output_dir=None):
    """Compare real and nominal pYijYij between results and no-SC databases.

    Computes both real (fixed base-year prices) and nominal (current prices) IO
    monetary matrices for both databases, takes the difference (results − no-SC),
    then produces for each valuation:
      1. A line chart of the total (scalar) difference per year.
      2. A diverging stacked bar chart decomposing the difference by row sector.
    """
    subdir = os.path.join(output_dir, "IO_monetary") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    idx = ['row_label', 'col_label']
    x = np.array(year_cols).astype('int')
    bar_width = (x[1] - x[0]) * 0.8 if len(x) > 1 else 4

    for nominal, label, prefix in [
        (False, "real",    "real"),
        (True,  "nominal", "nominal"),
    ]:
        io_sc  = compute_IO_monetary(df,       year_cols, nominal=nominal)
        io_nsc = compute_IO_monetary(df_no_sc, year_cols, nominal=nominal)

        a    = io_sc.set_index(idx)[year_cols].astype('float')
        b    = io_nsc.set_index(idx)[year_cols].astype('float')
        diff = a - b

        row_contrib = diff.groupby(level='row_label').sum()
        total_diff  = row_contrib.sum(axis=0).values

        # --- Plot 1: scalar time series ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, total_diff, marker='o', linewidth=2, color='steelblue')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(year_cols, rotation=45, ha='right', fontsize=11)
        ax.set_xlabel("Year", fontsize=13)
        ax.set_ylabel("Results − No-SC (monetary units)", fontsize=13)
        ax.set_title(f"Total difference in {label} pYijYij: Results − No-SC", fontsize=15)
        plt.tight_layout()
        if subdir is not None:
            fig.savefig(os.path.join(subdir, f"{prefix}_pYijYij_diff_total.png"), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

        # --- Plot 2: diverging stacked bar chart by row sector ---
        sectors = row_contrib.index.tolist()
        colors  = plt.cm.tab10(np.linspace(0, 0.9, len(sectors)))

        fig, ax = plt.subplots(figsize=(14, 7))
        pos_bottoms = np.zeros(len(year_cols))
        neg_bottoms = np.zeros(len(year_cols))

        for i, sector in enumerate(sectors):
            d   = row_contrib.loc[sector].values
            pos = np.where(d > 0, d, 0.0)
            neg = np.where(d < 0, d, 0.0)
            ax.bar(x, pos, bottom=pos_bottoms, color=colors[i], label=sector,
                   edgecolor='white', linewidth=0.4, width=bar_width)
            ax.bar(x, neg, bottom=neg_bottoms, color=colors[i],
                   edgecolor='white', linewidth=0.4, width=bar_width)
            pos_bottoms += pos
            neg_bottoms += neg

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(year_cols, rotation=45, ha='right', fontsize=11)
        ax.set_xlabel("Year", fontsize=13)
        ax.set_ylabel("Results − No-SC (monetary units)", fontsize=13)
        ax.set_title(f"Difference in {label} pYijYij (Results − No-SC), by row sector", fontsize=15)
        ax.legend(loc='upper right', fontsize=11)
        plt.tight_layout()
        if subdir is not None:
            fig.savefig(os.path.join(subdir, f"{prefix}_pYijYij_diff_by_sector.png"), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def real_GDP_decomposed(df, year_cols, Yijpij_nom=None):
    """Compute a real GDP aggregate for all years.

    Formula:
        (1/GDPPI) * (comp1 + comp2 - comp3)

    where:
        comp1 = sum_j pYj[j,t] * Yj[j,t]
        comp2 = sum_j tauSj[j,t] * Sj[j,t] * pSj[j,t]
        comp3 = sum_{j,i} Yijpij[j,i,t]   (total sum of all monetary IO flows)

    Parameters
    ----------
    df         : SCAF_results dataframe
    year_cols  : list of year column strings
    Yijpij_nom : optional pre-computed nominal IO monetary dataframe (from
                 compute_IO_monetary with nominal=True); computed internally if None.

    Returns a single-row dataframe with variable_name='real_GDP_decomposed'
    and one value per year column.
    """
    if Yijpij_nom is None:
        Yijpij_nom = compute_IO_monetary(df, year_cols, nominal=True)

    def _vec(var):
        rows = df.loc[df['variable_name'] == var].reset_index(drop=True)
        return rows.set_index('row_label')[year_cols].astype("float")

    pYj   = _vec("pYj")
    Yj    = _vec("Yj")
    pSj   = _vec("pSj")
    tauSj = _vec("tauSj")
    Sj    = _vec("Sj")

    GDPPI = df.loc[df['variable_name'] == "GDPPI", year_cols].values[0].astype("float")

    # -- comp1: sum_j pYj[j] * Yj[j] ----------------------------------------
    comp1 = (pYj * Yj).sum(axis=0).values  # shape (n_years,)

    # -- comp2: sum_j tauSj[j] * Sj[j] * pSj[j] -----------------------------
    comp2 = (tauSj * Sj * pSj).sum(axis=0).values

    # -- comp3: sum_{j,i} Yijpij[j, i] ---------------------------------------
    comp3 = Yijpij_nom[year_cols].values.astype("float").sum(axis=0)

    # -- final result ---------------------------------------------------------
    result = (comp1 + comp2 - comp3) / GDPPI

    record = {'variable_name': 'real_GDP_decomposed', 'row_label': '', 'col_label': '', 'status': ''}
    for col, val in zip(year_cols, result):
        record[col] = val
    meta_cols = ['variable_name', 'row_label', 'col_label', 'status']
    return pd.DataFrame([record], columns=meta_cols + year_cols)


def check_GDP_decomposed_vs_GDPreal(df, year_cols):
    """Print a comparison of real_GDP_decomposed against the GDPreal time series."""
    gdp_dec   = real_GDP_decomposed(df, year_cols)
    dec_vals  = gdp_dec[year_cols].values[0].astype("float")
    GDPreal   = df.loc[df['variable_name'] == "GDPreal", year_cols].values[0].astype("float")
    rel_diff  = (dec_vals - GDPreal) / GDPreal * 100

    print("\n=== real_GDP_decomposed vs GDPreal ===")
    print(f"  Years:        {year_cols}")
    print(f"  Decomposed:   {dec_vals}")
    print(f"  GDPreal:      {GDPreal}")
    print(f"  Difference:   {dec_vals - GDPreal}")
    print(f"  Rel diff (%): {rel_diff}")
    print("======================================\n")


def plot_GDP_diff_decomposed_2050(df_no_sc, df_results, year_cols, subtitle="", output_dir=None):
    """Decompose the no-SC minus results difference in real GDP at 2050 into four components.

    Components (computed per scenario, then differenced as no-SC minus results):
        pYj_composition_effect   = - sum_j [ pYj[j,t50]*Yj[j,t50]/prodGDPPI[t50] - pYj[j,t0]*Yj[j,t50] ]
        taxes                    = - sum_j tauSj[j,t50] * Sj[j,t50] * pSj[j,t50]
        pCIij_composition_effect = + sum_{j,i} [ nom_Yijpij[j,i,t50]/prodGDPPI[t50] - real_Yijpij[j,i,t50] ]
        CIij_volume_effect       = + sum_{j,i} real_Yijpij[j,i,t50]

    Plots a diverging stacked bar chart with one bar, segments above/below zero according
    to their sign. A horizontal line marks the total difference; its value appears
    explicitly on the y-axis. Title and labels clarify that values are no-SC minus results
    for 2050.
    """
    t0  = year_cols[0]
    t50 = '2050'

    def _compute_components(df):
        def _vec(var):
            rows = df.loc[df['variable_name'] == var].reset_index(drop=True)
            return rows.set_index('row_label')[year_cols].astype("float")

        pYj   = _vec("pYj")
        Yj    = _vec("Yj")
        pSj   = _vec("pSj")
        tauSj = _vec("tauSj")
        Sj    = _vec("Sj")

        prodGDPPI_val = float(df.loc[df['variable_name'] == "prodGDPPI", t50].values[0])

        pYj_comp = - float((pYj[t50] * Yj[t50] / prodGDPPI_val - pYj[t0] * Yj[t50]).sum())
        taxes    = - float((tauSj[t50] * Sj[t50] * pSj[t50]).sum() / prodGDPPI_val)

        Yijpij_nom  = compute_IO_monetary(df, year_cols, nominal=True)
        Yijpij_real = compute_IO_monetary(df, year_cols, nominal=False)
        nom_2050    = Yijpij_nom[t50].values.astype("float")
        real_2050   = Yijpij_real[t50].values.astype("float")

        pCIij_comp =  float((nom_2050 / prodGDPPI_val - real_2050).sum())
        CIij_vol   =  float(real_2050.sum())

        return {
            'pYj_composition_effect':   pYj_comp,
            'taxes':                    taxes,
            'pCIij_composition_effect': pCIij_comp,
            'CIij_volume_effect':       CIij_vol,
        }

    comp_no_sc   = _compute_components(df_no_sc)
    comp_results = _compute_components(df_results)

    components = ['pYj_composition_effect', 'taxes', 'pCIij_composition_effect', 'CIij_volume_effect']
    diffs = {k:  comp_results[k] - comp_no_sc[k] for k in components}
    total = sum(diffs.values())

    display_labels = {
        'pYj_composition_effect':   'pYj composition effect',
        'taxes':                    'Taxes (tauSj·Sj·pSj)',
        'pCIij_composition_effect': 'pCIij composition effect',
        'CIij_volume_effect':       'CI volume effect (real Yijpij)',
    }
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(components)))

    fig, ax = plt.subplots(figsize=(5, 7))
    pos_bottom = 0.0
    neg_bottom = 0.0

    for i, name in enumerate(components):
        val = diffs[name]
        if val >= 0:
            ax.bar([0], [val], bottom=[pos_bottom], color=colors[i],
                   label=display_labels[name], edgecolor='white', linewidth=0.4)
            pos_bottom += val
        else:
            ax.bar([0], [val], bottom=[neg_bottom], color=colors[i],
                   label=display_labels[name], edgecolor='white', linewidth=0.4)
            neg_bottom += val

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(total, color='black', linewidth=1.5, linestyle='--', label=f'Total: {total:.2f}')

    existing_ticks = list(ax.get_yticks())
    all_ticks = sorted(set(existing_ticks + [total]))
    ax.set_yticks(all_ticks)

    ax.set_xticks([0])
    ax.set_xticklabels(['No-SC − Results, 2050'], fontsize=11)
    ax.set_ylabel('Difference (monetary units)', fontsize=13)
    title = 'Decomposition of the real output differences\nbetween baseline and structural change scenario'
    if subtitle:
        title += f'\n{subtitle}'
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()

    if output_dir is not None:
        subdir = os.path.join(output_dir, "real_output_differences_decomposition")
        os.makedirs(subdir, exist_ok=True)
        fig.savefig(
            os.path.join(subdir, "decomposition_of_real_output_differences_between_baseline_and_SC_scenario.png"),
            bbox_inches='tight',
        )
        plt.close(fig)
    else:
        plt.show()


def plot_pYj_composition_effect_diff_by_sector_2050(df_sc, df_baseline, year_cols, subtitle="", output_dir=None):
    """Bar chart of the SC minus baseline pYj composition effect by sector at 2050.

    For each sector j the per-scenario quantity is:
        (pYj[j, 2050] / prodGDPPI[2050] - pYj[j, t0]) * Yj[j, 2050]

    where prodGDPPI[2050] is the cumulative product of GDPPI from t0 to 2050.
    The plotted value is SC minus baseline for each sector.
    """
    t0  = year_cols[0]
    t50 = '2050'

    def _sector_vals(df):
        def _vec(var):
            rows = df.loc[df['variable_name'] == var].reset_index(drop=True)
            return rows.set_index('row_label')[year_cols].astype("float")

        pYj = _vec("pYj")
        Yj  = _vec("Yj")
        prodGDPPI_val = float(df.loc[df['variable_name'] == "prodGDPPI", t50].values[0])
        return - (pYj[t50] / prodGDPPI_val - pYj[t0]) * Yj[t50]

    vals_sc       = _sector_vals(df_sc)
    vals_baseline = _sector_vals(df_baseline)
    diff = vals_sc - vals_baseline  # Series indexed by sector

    sectors = diff.index.tolist()
    total   = float(diff.sum())
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(sectors)))

    fig, ax = plt.subplots(figsize=(5, 7))
    pos_bottom = 0.0
    neg_bottom = 0.0

    for i, sector in enumerate(sectors):
        val = float(diff[sector])
        if val >= 0:
            ax.bar([0], [val], bottom=[pos_bottom], color=colors[i],
                   label=sector, edgecolor='white', linewidth=0.4)
            pos_bottom += val
        else:
            ax.bar([0], [val], bottom=[neg_bottom], color=colors[i],
                   label=sector, edgecolor='white', linewidth=0.4)
            neg_bottom += val

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(total, color='black', linewidth=1.5, linestyle='--', label=f'Total: {total:.2f}')

    existing_ticks = list(ax.get_yticks())
    ax.set_yticks(sorted(set(existing_ticks + [total])))

    ax.set_xticks([0])
    ax.set_xticklabels(['SC − Baseline, 2050'], fontsize=11)
    ax.set_ylabel('SC − Baseline (monetary units)', fontsize=13)
    title = 'pYj composition effect difference by sector (SC − Baseline), 2050'
    if subtitle:
        title += f'\n{subtitle}'
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()

    if output_dir is not None:
        subdir = os.path.join(output_dir, "real_output_differences_decomposition")
        os.makedirs(subdir, exist_ok=True)
        fig.savefig(
            os.path.join(subdir, "pYj_composition_effect_diff_by_sector_2050.png"),
            bbox_inches='tight',
        )
        plt.close(fig)
    else:
        plt.show()


def plot_calibration_energy_consumer_shares(hybridization_df, output_dir=None):
    _palette = ENERGY_USE_PALETTE
    df = hybridization_df[
        (hybridization_df["Region"] == "EUR") &
        (hybridization_df["Variable"] == "Volume")
    ]
    pivot = (
        df.pivot_table(index="Energy consumers", columns="Energy uses",
                       values="2020", aggfunc="sum")
        .fillna(0)
    )
    consumers = pivot.index.tolist()
    energy_uses = pivot.columns.tolist()
    n = len(consumers)
    x = np.arange(n)
    colors = [_palette[i % len(_palette)] for i in range(len(energy_uses))]

    subdir = os.path.join(output_dir, "energy_uses_shares") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    # ---- normalised ----
    totals = pivot.sum(axis=1)
    safe_totals = totals.where(totals != 0, 1.0)
    shares = pivot.div(safe_totals, axis=0)

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    bottoms = np.zeros(n)
    for i, use in enumerate(energy_uses):
        vals = shares[use].values
        ax1.bar(x, vals, bottom=bottoms, color=colors[i], label=use,
                edgecolor='white', linewidth=0.5)
        bottoms += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(consumers, rotation=30, ha='right', fontsize=11)
    ax1.set_ylabel("Share", fontsize=13)
    ax1.set_ylim(0, 1)
    ax1.set_title("energy use composition by consumer — normalised (2020)", fontsize=15)
    ax1.legend(bbox_to_anchor=(1.0, 0.5), loc='center left', fontsize=11)
    plt.tight_layout()
    if subdir is not None:
        fig1.savefig(os.path.join(subdir, "energy_consumers_shares_normalised_2020.png"), bbox_inches='tight')
        plt.close(fig1)
    else:
        plt.show()

    # ---- absolute ----
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    bottoms = np.zeros(n)
    for i, use in enumerate(energy_uses):
        vals = pivot[use].values
        ax2.bar(x, vals, bottom=bottoms, color=colors[i], label=use,
                edgecolor='white', linewidth=0.5)
        bottoms += vals
    ax2.set_xticks(x)
    ax2.set_xticklabels(consumers, rotation=30, ha='right', fontsize=11)
    ax2.set_ylabel("Energy volume (EJ)", fontsize=13)
    ax2.set_title("Energy use composition by consumer (2020)", fontsize=15)
    ax2.legend(bbox_to_anchor=(1.0, 0.5), loc='center left', fontsize=11)
    plt.tight_layout()
    if subdir is not None:
        fig2.savefig(os.path.join(subdir, "energy_consumers_shares_absolute_2020.png"), bbox_inches='tight')
        plt.close(fig2)
    else:
        plt.show()


def plot_calibration_energy_use_shares(hybridization_df, output_dir=None):
    df = hybridization_df[
        (hybridization_df["Region"] == "EUR") &
        (hybridization_df["Variable"] == "Volume")
    ]
    pivot = (
        df.pivot_table(index="Energy uses", columns="Energy consumers",
                       values="2020", aggfunc="sum")
        .fillna(0)
    )
    energy_uses = pivot.index.tolist()
    consumers   = pivot.columns.tolist()
    n = len(energy_uses)
    x = np.arange(n)
    _palette = ENERGY_USE_PALETTE
    colors = [_palette[i % len(_palette)] for i in range(len(consumers))]

    subdir = os.path.join(output_dir, "energy_uses_shares") if output_dir is not None else None
    if subdir is not None:
        os.makedirs(subdir, exist_ok=True)

    # ---- Chart 1: normalised (bar height = 1) ----
    totals = pivot.sum(axis=1)
    safe_totals = totals.where(totals != 0, 1.0)
    shares = pivot.div(safe_totals, axis=0)

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    bottoms = np.zeros(n)
    for i, consumer in enumerate(consumers):
        vals = shares[consumer].values
        ax1.bar(x, vals, bottom=bottoms, color=colors[i], label=consumer,
                edgecolor='white', linewidth=0.5)
        bottoms += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(energy_uses, rotation=30, ha='right', fontsize=11)
    ax1.set_ylabel("Share", fontsize=13)
    ax1.set_ylim(0, 1)
    ax1.set_title("consumer composition by energy use — normalised (2020)", fontsize=15)
    ax1.legend(bbox_to_anchor=(1.0, 0.5), loc='center left', fontsize=11)
    plt.tight_layout()
    if subdir is not None:
        fig1.savefig(os.path.join(subdir, "energy_uses_shares_normalised_2020.png"), bbox_inches='tight')
        plt.close(fig1)
    else:
        plt.show()

    # ---- Chart 2: absolute (bar height = total EJ) ----
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    bottoms = np.zeros(n)
    for i, consumer in enumerate(consumers):
        vals = pivot[consumer].values
        ax2.bar(x, vals, bottom=bottoms, color=colors[i], label=consumer,
                edgecolor='white', linewidth=0.5)
        bottoms += vals
    ax2.set_xticks(x)
    ax2.set_xticklabels(energy_uses, rotation=30, ha='right', fontsize=11)
    ax2.set_ylabel("Energy volume (EJ)", fontsize=13)
    ax2.set_title("Consumers composition of energy uses (2020)", fontsize=15)
    ax2.legend(bbox_to_anchor=(1.0, 0.5), loc='center left', fontsize=11)
    plt.tight_layout()
    if subdir is not None:
        fig2.savefig(os.path.join(subdir, "energy_uses_shares_absolute_2020.png"), bbox_inches='tight')
        plt.close(fig2)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# World Bank reference time-series
# ---------------------------------------------------------------------------

def construct_wb_timeseries(data_dir: str = ".") -> tuple:
    """Read World Bank CSVs from WB_data/ and return two wide DataFrames.

    Parameters
    ----------
    data_dir : str
        Directory that contains the ``WB_data`` sub-folder.

    Returns
    -------
    nominal_df : pd.DataFrame
        Columns: variable | sector | unit | 1995 | … | 2020
        Rows: VA shares (nominal) and log real GDP per capita.
    real_df : pd.DataFrame
        Same layout, VA shares computed from constant-2015-USD values.
    """

    YEARS = [str(y) for y in range(1995, 2021)]   # "1995" … "2020"
    WB_DIR = os.path.join(data_dir, "WB_data")

    with open(os.path.join(WB_DIR, "mappings.json")) as _f:
        _mappings = json.load(_f)
    AGG_CODES  = _mappings["EUR region"]
    USD_TO_EUR = _mappings["US$2015-EUR2020"]

    def _read_wb(filename):
        """Load one World Bank CSV, clean it, and return a country-indexed DataFrame."""
        path = os.path.join(WB_DIR, filename)
        df = pd.read_csv(path, skiprows=4, encoding="utf-8-sig")
        # Replace World Bank "not available" sentinel
        df.replace("..", np.nan, inplace=True)
        df.set_index("Country Code", inplace=True)
        # Keep only the years we need and convert to numeric
        df = df[YEARS].apply(pd.to_numeric, errors="coerce")
        # Validate that all required codes are present
        missing = [c for c in AGG_CODES if c not in df.index]
        if missing:
            raise ValueError(
                f"construct_wb_timeseries: file '{filename}' is missing "
                f"the following required country codes: {missing}"
            )
        return df

    # ---- Read all nine files ----
    agr_nom  = _read_wb("Agriculture, forestry, and fishing, value added (current US$).csv")
    man_nom  = _read_wb("Industry (including construction), value added (current US$).csv")
    srv_nom  = _read_wb("Services, value added (current US$).csv")
    agr_real = _read_wb("Agriculture, forestry, and fishing, value added (constant 2015 US$).csv")
    man_real = _read_wb("Industry (including construction), value added (constant 2015 US$).csv")
    srv_real = _read_wb("Services, value added (constant 2015 US$).csv")
    gdp_real = _read_wb("GDP (constant 2015 US$).csv")
    pop      = _read_wb("pop.csv")

    # ---- Sum levels across AGG_CODES (one Series per variable, index = year) ----
    AGR_NOM  = agr_nom.loc[AGG_CODES, YEARS].sum(axis=0)
    MAN_NOM  = man_nom.loc[AGG_CODES, YEARS].sum(axis=0)
    SRV_NOM  = srv_nom.loc[AGG_CODES, YEARS].sum(axis=0)

    AGR_REAL = agr_real.loc[AGG_CODES, YEARS].sum(axis=0)
    MAN_REAL = man_real.loc[AGG_CODES, YEARS].sum(axis=0)
    SRV_REAL = srv_real.loc[AGG_CODES, YEARS].sum(axis=0)

    GDP_REAL = gdp_real.loc[AGG_CODES, YEARS].sum(axis=0)
    POP      = pop.loc[AGG_CODES, YEARS].sum(axis=0)

    # ---- Derived quantities ----
    TOTAL_NOM_VA  = AGR_NOM  + MAN_NOM  + SRV_NOM
    TOTAL_REAL_VA = AGR_REAL + MAN_REAL + SRV_REAL

    agr_nom_share  = AGR_NOM  / TOTAL_NOM_VA
    man_nom_share  = MAN_NOM  / TOTAL_NOM_VA
    srv_nom_share  = SRV_NOM  / TOTAL_NOM_VA

    agr_real_share = AGR_REAL / TOTAL_REAL_VA
    man_real_share = MAN_REAL / TOTAL_REAL_VA
    srv_real_share = SRV_REAL / TOTAL_REAL_VA

    GDP_REAL_EUR   = GDP_REAL * USD_TO_EUR          # constant 2015 US$ → EUR 2020 exchange rate
    log_real_gdp_pc = np.log(GDP_REAL_EUR / POP)

    # ---- Helper to build one row dict ----
    def _row(variable, sector, unit, series):
        return {"variable": variable, "sector": sector, "unit": unit,
                **{y: series[y] for y in YEARS}}

    # ---- Assemble DataFrames ----
    nom_unit  = "fraction of nominal three-sector VA"
    real_unit = "fraction of real three-sector VA"
    gdp_unit  = "log EUR 2020 per person"

    nominal_rows = [
        _row("value_added_share_nominal", "agriculture",   nom_unit,  agr_nom_share),
        _row("value_added_share_nominal", "industry", nom_unit,  man_nom_share),
        _row("value_added_share_nominal", "services",      nom_unit,  srv_nom_share),
        _row("log_real_gdp_per_capita",   "all",           gdp_unit,  log_real_gdp_pc),
    ]

    real_rows = [
        _row("value_added_share_real",  "agriculture",   real_unit, agr_real_share),
        _row("value_added_share_real",  "industry", real_unit, man_real_share),
        _row("value_added_share_real",  "services",      real_unit, srv_real_share),
        _row("log_real_gdp_per_capita", "all",           gdp_unit,  log_real_gdp_pc),
    ]

    nominal_df = pd.DataFrame(nominal_rows).reset_index(drop=True)
    real_df    = pd.DataFrame(real_rows).reset_index(drop=True)

    return nominal_df, real_df


def export_wb_timeseries_csv(data_dir="Data_postprocessing", output_path=None):
    """Aggregate World Bank data over the EUR region and write a wide-format CSV.

    Rows exported
    -------------
    value_added_share_nominal  — agriculture / industry / services
    value_added_share_real     — agriculture / industry / services
    log_real_gdp_per_capita    — all  (log EUR 2020 per person)
    population                 — all  (persons)
    gdp_real                   — all  (M constant 2015 USD)
    gdp_nominal                — all  (M current USD)

    All series span 1995–2020 and are summed across the EUR region defined in
    ``WB_data/mappings.json`` before being written.

    Parameters
    ----------
    data_dir    : str — folder containing ``WB_data/`` (default "Data_postprocessing")
    output_path : str or None — destination CSV path.  Defaults to
                  ``<data_dir>/WB_data/wb_timeseries.csv``

    Returns
    -------
    pd.DataFrame — the data written to disk
    """
    YEARS  = [str(y) for y in range(1995, 2021)]
    WB_DIR = os.path.join(data_dir, "WB_data")

    with open(os.path.join(WB_DIR, "mappings.json")) as fh:
        mappings = json.load(fh)
    AGG_CODES = mappings["EUR region"]

    # --- reuse construct_wb_timeseries for shares + log_real_gdp_per_capita ---
    nom_df, real_df = construct_wb_timeseries(data_dir=data_dir)

    # --- read and aggregate level variables over the EUR region ---
    def _read_wb(filename):
        path = os.path.join(WB_DIR, filename)
        df = pd.read_csv(path, skiprows=4, encoding="utf-8-sig")
        df.replace("..", np.nan, inplace=True)
        df.set_index("Country Code", inplace=True)
        return df[YEARS].apply(pd.to_numeric, errors="coerce")

    POP      = _read_wb("pop.csv").loc[AGG_CODES].sum(axis=0)
    GDP_REAL = _read_wb("GDP (constant 2015 US$).csv").loc[AGG_CODES].sum(axis=0) / 1e6
    GDP_NOM  = _read_wb("GDP (current US$).csv").loc[AGG_CODES].sum(axis=0) / 1e6

    def _row(variable, sector, unit, series):
        return {"variable": variable, "sector": sector, "unit": unit,
                **{y: series[y] for y in YEARS}}

    # --- combine into one DataFrame ---
    # nom_df:  3 VA-share rows + 1 log_gdp row
    # real_df: 3 VA-share rows + 1 log_gdp row (duplicate — skip)
    rows = list(nom_df.to_dict("records"))
    for rec in real_df.to_dict("records"):
        if rec["variable"] != "log_real_gdp_per_capita":
            rows.append(rec)
    rows.append(_row("population",  "all", "persons",             POP))
    rows.append(_row("gdp_real",    "all", "M constant 2015 USD", GDP_REAL))
    rows.append(_row("gdp_nominal", "all", "M current USD",       GDP_NOM))

    out_df = pd.DataFrame(rows).reset_index(drop=True)

    if output_path is None:
        output_path = os.path.join(WB_DIR, "wb_timeseries.csv")
    out_df.to_csv(output_path, index=False)
    return out_df