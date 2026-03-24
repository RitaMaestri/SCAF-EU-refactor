#DATA ANALYSIS
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
import sys
import matplotlib.colors as mcolors
from scipy import stats
from lib import extract_var_df, plot_varj_evol, plot_variable_1D


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




csv_path = "Data_postprocessing/data/test(23-03-2026_19:14).csv"
csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
output_dir = os.path.join("Data_postprocessing", " plots", csv_stem)
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
meta_cols = ['variable_name', 'row_label', 'col_label', 'status']
year_cols = [c for c in df.columns if c not in meta_cols and int(c) <= 2050]
df = df[meta_cols + year_cols]


#df=df.drop(columns=['2005', '2010'])
#df2 = pd.read_csv("results/7-sectors-no-energy-coupling.csv")

#pq = "p","q","pq"


    
    

plot_varj_evol(df=df, var="KLj", pq="pq", diff=False,display_top_names=7, mytitle="Normalised evolution of the value added per sector (value)", output_dir=output_dir)
plot_varj_evol(df=df, var="KLj", pq="q", diff=False,display_top_names=7, mytitle="Normalised evolution of the value added per sector (volume)", output_dir=output_dir)
#plot_varj_evol(df=df, var="KLj", pq="p", diff=False,display_top_names=7, mytitle="Normalised evolution of the value added per sector (price)", output_dir=output_dir)

plot_varj_evol(df=df, var="Kj", pq="q", diff=False,display_top_names=7, mytitle="Normalised evolution of capital per sector (volume)", output_dir=output_dir)
plot_varj_evol(df=df, var="Lj", pq="q", diff=False,display_top_names=7, mytitle="Normalised evolution of labour per sector (volume)", output_dir=output_dir)

plot_varj_evol(df=df, var="Yj", pq="q", diff=False,display_top_names=7, output_dir=output_dir)










################# plot capital and labour prices ###############################







    

plot_variable_1D(df, "bKL", "q", diff=False, output_dir=output_dir)
###############################################################################
plot_variable_1D(df, "K", "p", diff=False, output_dir=output_dir)





################# plot GDP growth, capital and labour growth #######################
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
plt.ylabel("Relative change with respect to year 2020", fontsize = 17)

plt.savefig(os.path.join(output_dir, "IAM_parameters_evolution.png"), bbox_inches='tight')
plt.close()
###############################################################################







