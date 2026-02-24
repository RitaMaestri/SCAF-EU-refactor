#DATA ANALYSIS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
import sys
from import_GTAP_data import N,sectors
import matplotlib.colors as mcolors
from scipy import stats


# sectors_names_eng=[
# "AGRICULTURE",
# "MANUFACTURE",
# "SERVICES",

#     ]

# A = sectors_names_eng.index("AGRICULTURE")
# M = sectors_names_eng.index("MANUFACTURE")
# SE = sectors_names_eng.index("SERVICES")


# cmap=["#6CD900",
# "#8C8CFF",
# "#FF91C8"
# ]




sectors_names_eng=[
"AGRICULTURE",
"MANUFACTURE",
"SERVICES",
"STEEL",
"CHEMICAL",
"ENERGY",
"TRANSPORTATION",
    ]

A = sectors_names_eng.index("AGRICULTURE")
M = sectors_names_eng.index("MANUFACTURE")
SE = sectors_names_eng.index("SERVICES")
E = sectors_names_eng.index("ENERGY")
ST = sectors_names_eng.index("STEEL")
CH = sectors_names_eng.index("CHEMICAL")
T = sectors_names_eng.index("TRANSPORTATION")

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

name = sectors

df = pd.read_csv("results/johansen2015-2150exoKnextREMIND-7sectors(24-05-2024_16:56).csv")
df.rename(columns={df.columns[0]: 'variable'},inplace=True)
df.loc[ df['variable'] == "L" ].drop("variable", axis=1)
a=df.loc[ df['variable'] == "bKL" ].drop("variable", axis=1)
df=df.loc[:, :"2050"]
#df=df.drop(columns=['2005', '2010'])
#df2 = pd.read_csv("results/7-sectors-no-energy-coupling.csv")

#pq = "p","q","pq"

def extract_var_df(var, pq):
    P=df.loc[ df['variable'] == "p"+var ].reset_index(drop=True).drop("variable", axis=1)
    Q=df.loc[ df['variable'] == var ].reset_index(drop=True).drop("variable", axis=1)

    if pq=="pq":
        var_df = P*Q
    elif pq == "p":
        var_df = P
    elif pq == "q":
        var_df= Q
    else:
        raise ValueError("wrong pq! Expected 'pq', 'p', or 'q'")
    #print(var_df)
    if len(var_df.index)>1:
        var_df.insert(0, "Name", sectors_names_eng, True) 
    return var_df 
    
    


def plot_varj_evol(var, pq, max_year="2050", display_top_names=4, display_bottom_names=0, diff=False, mytitle=None):
    
    var_df=extract_var_df(var, pq)
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
    display_bottom_names=len(sectors)-display_bottom_names
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
            
    plt.xlim(2014.99,2050.01)
    
    
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
    plt.ylabel("Relative change with respect to year 2015", fontsize = 17)
    
    plt.show()
    
    

plot_varj_evol(var="KLj", pq="pq", diff=False,display_top_names=7, mytitle="Normalised evolution of the value added per sector (value)")
plot_varj_evol(var="KLj", pq="q", diff=False,display_top_names=7, mytitle="Normalised evolution of the value added per sector (volume)")
#plot_varj_evol(var="KLj", pq="p", diff=False,display_top_names=7, mytitle="Normalised evolution of the value added per sector (price)")

plot_varj_evol(var="Kj", pq="q", diff=False,display_top_names=7, mytitle="Normalised evolution of capital per sector (volume)")
plot_varj_evol(var="Lj", pq="q", diff=False,display_top_names=7, mytitle="Normalised evolution of labour per sector (volume)")

plot_varj_evol(var="Yj", pq="q", diff=False,display_top_names=7)




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



#pharma_ind=sectors_names_eng.index("PHARMACEUTICAL IND.")
agri_ind=sectors_names_eng.index("AGRICULTURE")
    

par_keys = ['tauSj', 'tauYj', 'pXj', 'bKLj', 'alphaKj', 'alphaLj', 'aKLj', 'alphaCj', 
'alphaGj', 'alphaIj', 'alphaXj', 'alphaDj', 'betaDj', 'betaMj', 
'sigmaXj', 'sigmaSj']


# for i in par_keys:
#     array=np.array(df.loc[(df['variable'] == i)].iloc[:, 1])
#     order = array.argsort()
#     print(i,36-18-np.where(order==pharma_ind)[0].item(),"\n")


################# plot capital and labour prices ###############################



def plot_1D(var_df, title, diff=False):
    fig = plt.figure(figsize=(15,10))
    x=np.array(var_df.columns).astype('int')
    y=np.array(var_df)[0].astype('float')
    y= y-shift(y, 1, cval=np.NaN) if diff else y
    plt.plot(x,y , label = title)
    plt.title(title, fontsize = 35)
    plt.xlabel("Year",fontsize = 20)
    plt.ylabel("Relative change with respect to year 2015", fontsize = 20)
    plt.show()
    
def plot_variable_1D(var_name, pq, diff=False):
    var_df= extract_var_df(var_name, pq)
    plot_1D(var_df, var_name, diff=False)


    

plot_variable_1D("bKL", "q", diff=False)
###############################################################################
plot_variable_1D("K", "p", diff=False)





################# plot GDP growth, capital and labour growth #######################
L=np.array(df.loc[(df['variable'] == "L")].iloc[:,1:])[0].astype("float")
bKL=np.array(df.loc[(df['variable'] == "bKL")].iloc[:,1:])[0].astype("float")
LbKL=L*bKL
K=np.array(df.loc[(df['variable'] == "K")].iloc[:,1:])[0].astype("float")
GDPreal=np.array(df.loc[(df['variable'] == "GDPreal")].iloc[:,1:])[0].astype("float")
pL=np.array(df.loc[(df['variable'] == "pL")].iloc[:,1:])[0].astype("float")
pK=np.array(df.loc[(df['variable'] == "pK")].iloc[:,1:])[0].astype("float")

fig = plt.figure(figsize=(13,7))
ax = fig.add_subplot(111)
x=np.array(df.columns[1:]).astype('int')
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

# colors= ["#06646E","#0DABBD","#56C9D6"]
# for i,j in enumerate(ax.lines):
#     j.set_color(colors[i])
# Shrink current axis's height by 10% on the bottom
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 13})
plt.title("Evolution of the parameters from the coupled IAM", fontsize = 20)
plt.xlabel("Year",fontsize = 17)
plt.ylabel("Relative change with respect to year 2015", fontsize = 17)

plt.show()
###############################################################################




######################### numerical analysis of KLj2050 extremes ###############
def order_variablesj (var, relative=False):
    array2015=np.array(df["2015"].loc[ df['variable'] == var ]) 
    array2050=np.array(df["2050"].loc[ df['variable'] == var ])
    array= array2050/array2015 if relative else array2050  
    return(pd.Series(array, index =sectors_names_eng).sort_values())

def value(var, sec="ENERGY",year="2050"):
    array2050=np.array(df.loc[ df['variable'] == var ,year])
    series=pd.Series(array2050, index =sectors_names_eng).sort_values()
    return series[sec]


def calibration_value(var, sec="ENERGY"):
    array2015=np.array(df.loc[ df['variable'] == var ,"2015"])
    series=pd.Series(array2015, index =sectors_names_eng).sort_values()
    return series[sec]

def calibration_value_scalar(var):
    scalar=float(df.loc[ df['variable'] == var ,"2015"])
    return scalar

#print(value("KLj", "MACHINES"))

par_keys = ['tauSj', 'tauYj', 'pXj', 'bKLj', 'alphaKj', 'alphaLj', 'aKLj', 'alphaCj', 
'alphaGj', 'alphaIj', 'alphaXj', 'alphaDj', 'betaDj', 'betaMj', 
'sigmaXj', 'sigmaSj']

nansX=np.argwhere(np.array(df["2050"].loc[ df['variable'] == "Xj" ]==0)).flatten()


plot_varj_evol(var="Yj", pq="p", diff=False)

order_variablesj("alphaXj")#.drop([sectors_names_eng[i] for i in nans])


def function_pD(pY,sigmaY,alphaX,pX,alphaD):
    Ysigma=1-sigmaY
    pD=(((pY**Ysigma)-(alphaX**sigmaY)*(pX**Ysigma)) / (alphaD**sigmaY))**(1/Ysigma)
    return pD

value("pDj", "AGRICULTURE")

print(function_pD(pY=value("pYj"),
                  sigmaY=value("sigmaXj"),
                  alphaX=value("alphaXj"),
                  pX=value("pXj"),
                  alphaD=value("alphaDj")))


plot_varj_evol(var="Mj", pq="pq", diff=False)



##################  rebuild IO matrix  #############
def flatten_index(m, n, num_columns):
    return m * num_columns + n

indexes_E=np.array(range(E, len(sectors)**2, len(sectors) ))

EE_idx=flatten_index(E, E, len(sectors))
Yij_array=df.loc[ df['variable'] == "Yij" ]
Y_EE=Yij_array.iloc[EE_idx][1:]
pY_Ej=df.loc[ df['variable'] == "pY_Ej" ]
pY_EE=pY_Ej.iloc[E][1:]
pY_EEY_EE=Y_EE*pY_EE

pY_EEY_EE.plot(title="p_EE*Y_EE")
pY_EE.plot(title="p_EE")
Y_EE.plot(title="Y_EE")


plot_variable_1D("GDPreal", "q", diff=False)



calibration_value("pY_Ej")
calibration_value("pE_Pj")
calibration_value("pE_Ej")
calibration_value_scalar("pE_TT")
calibration_value_scalar("pE_TnT")
calibration_value_scalar("pE_B")

plot_varj_evol(var="KLj", pq="pq", diff=False)
plot_variable_1D("GDPreal", "q", diff=False)

plot_varj_evol(var="Sj", pq="pq", diff=False)

Y_iE=Yij_array.iloc[indexes_E]

