import numpy as np
import pandas as pd
import sys
import random
import math
import copy
import warnings
warnings.filterwarnings("ignore")



from import_GTAP_data import N
from helpers.solvers import dict_least_squares
from calibration import E
from helpers.handle_jump import conduct_solution
from Variables_specs import VARIABLES_SPECS
from helpers.time_series_df_functions import reformat_bounds_for_solver, build_and_fill_timeseries_df, timeseries_df_to_endogenous_dict, timeseries_df_to_exo_endo_dict, timeseries_df_to_unsolved_year_dict, dict_to_timeseries_df
from system_of_equations import system, joint_dict




########################################################################
########################## BUILD TIMESERIES DF #########################
########################################################################

from run_setup import growth_ratios_df, years, output_file_name

timeseries_df=build_and_fill_timeseries_df(VARIABLES_SPECS,growth_ratios_df,years)

solver_bounds = reformat_bounds_for_solver(VARIABLES_SPECS)
        


########################################################################
########################  CALIBRATION CHECK  ###########################
########################################################################

endo_var_calibration = timeseries_df_to_endogenous_dict(timeseries_df, years[0], VARIABLES_SPECS)

exo_var_calibration_unsolved = timeseries_df_to_unsolved_year_dict(timeseries_df, years[0], VARIABLES_SPECS)

max_err_cal=max(abs(system(endo_var_calibration, exo_var_calibration_unsolved)))

if max_err_cal>1e-07:
    d=joint_dict(exo_var_calibration_unsolved,endo_var_calibration)
    raise RuntimeError("the system is not correctly calibrated")


########################################################################################
##########################  FUNCTIONS FOR NON-CONVERGENCE  #############################
########################################################################################

seed = random.randrange(sys.maxsize)

def kick(variables, number_modified=20, percentage_modified=0.4, modification=0.2, seed=seed):
    
    random.seed(seed)
    kicked_variables = copy.deepcopy(variables)
    keys = random.sample(list(variables.keys()), k=number_modified)

    for v_key in keys:
        if v_key in kicked_variables:
            if hasattr(kicked_variables[v_key], "__len__"):
                v_len = len(kicked_variables[v_key])
                sec = random.sample(range(v_len), k=math.ceil(v_len * percentage_modified))
                new_values= [(1+random.choice((-1, 1))*modification)*float(kicked_variables[v_key][i]) for i in sec]
                kicked_variables[v_key][sec] = new_values
            else:
                if kicked_variables[v_key] != 0:
                    new_value = (1 + random.choice((-1, 1)) * modification) * kicked_variables[v_key]
                    kicked_variables[v_key] = new_value

    return kicked_variables


def equilibrium(pKLj, KLj, pMj, Mj, pYj, Yj, pCj, pY_Ej, pXj, Yij, Cj, Gj, Ij, Xj, tauSj, tauYj):
    # build prices matrix
    pCj_matrix = np.array([pCj] * (len(pCj))).T
    pCj_matrix[E] = pY_Ej
    total_consumption_j = pCj * Cj + pCj * Gj + pCj * Ij + (pCj_matrix * Yij).sum(axis=1)
    error = 1 - (pKLj * KLj + pYj * Yj * tauYj / (1 + tauYj) + total_consumption_j * tauSj / (1 + tauSj) + pMj * Mj + (pCj_matrix * Yij).sum(axis=0) - pXj * Xj) / total_consumption_j
    if max(abs(error)) > 10e-6:
        is_equilibrium = False
    else:
        is_equilibrium = True
    return (is_equilibrium,error)



########################################################################################
##################################  SYSTEM SOLUTION  ###################################
########################################################################################


for t in range(len(years)):
    
    print("year: ", years[t])
    
    if t==0:
        endo_vars=endo_var_calibration # kick(endo_var_calibration) 

        endo_exo_vars=exo_var_calibration_unsolved 
    else:
        endo_vars=timeseries_df_to_endogenous_dict(timeseries_df, years[t-1], VARIABLES_SPECS) #kick(System.df_to_dict(var=True, t=years[t-1]))
    
        endo_exo_vars=timeseries_df_to_exo_endo_dict(timeseries_df, years[t], VARIABLES_SPECS)
    
    sol = dict_least_squares( system, endo_vars, endo_exo_vars, solver_bounds, N, verb=1, check=True)
        
    maxerror=max(abs( system(sol.dvar, endo_exo_vars)))

    endo_solution=sol.dvar
    d=joint_dict(endo_exo_vars, endo_solution)
    
    if maxerror>1e-06:
        print("conducting solution: UNTESTESTED")
        endo_exo_origin=timeseries_df_to_exo_endo_dict(timeseries_df, years[t-1], VARIABLES_SPECS)
        endo_solution=conduct_solution(endo_exo_origin, endo_exo_vars, system, solver_bounds, N, timeseries_df,years[t-1], threshold= 0.07, growth_rate=0.01)
        d=joint_dict(endo_exo_vars, endo_solution)
        print("the system converged!")


    equilibrium_t=equilibrium(pKLj=d["pKLj"], KLj=d["KLj"], pMj=d["pMj"], Mj=d["Mj"], pYj=d["pYj"], Yj=d["Yj"], pCj=d["pCj"], pY_Ej=d["pY_Ej"], pXj=d["pXj"], Yij=d["Yij"], Cj=d["Cj"], Gj=d["Gj"], Ij=d["Ij"], Xj=d["Xj"], tauSj=d["tauSj"], tauYj=d["tauYj"])
    is_equilibrium=equilibrium_t[0]
    error=equilibrium_t[1]
    
    if not is_equilibrium:
        raise RuntimeError(f"the system is not at equilibrium: {equilibrium_t[1]}")

    timeseries_df= dict_to_timeseries_df(endo_solution, timeseries_df, years[t], VARIABLES_SPECS, years) 

       
#  SAVE CSV  
timeseries_df.to_csv(output_file_name, index=False)






