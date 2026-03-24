import os
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
        fname = (mytitle if mytitle else (pq + var)).replace(" ", "_") + ".png"
        plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
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
        fname = title.replace(" ", "_") + ".png"
        plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
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
        plt.savefig(os.path.join(output_dir, "IAM_parameters_evolution.png"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()