import numpy as np
import pandas as pd
import sys
#from data_closures import bounds, N, calibrationDict
from import_GTAP_data import non_zero_index_G, non_zero_index_I, non_zero_index_X, non_zero_index_M, non_zero_index_Yij,non_zero_index_L,sectors, N
import model_equations as eq
from solvers import dict_least_squares
#from time_series_data import sys_df
from datetime import datetime
import random
import math
import copy
from data_closures import bounds
from simple_calibration import A,M,SE,E,ST,CH,T
import warnings
import handle_jump as jump
from Variables_specs import VARIABLES_SPECS
from build_time_series_df import build_and_fill_timeseries_df, timeseries_df_to_endogenous_dict, timeseries_df_to_exo_endo_dict, timeseries_df_to_unsolved_year_dict, dict_to_timeseries_df
warnings.filterwarnings("ignore")

#############################################################
#######################  SETUP  #############################
#############################################################

#input file name
input_file =  "REMIND_exogenous_data_sectors"



now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")
add_string = "test"
output_file_name = str().join(["results/",
                    add_string, "(", dt_string, ")", ".csv"])




growth_ratios_df = pd.read_csv(
    "data/"+input_file+".csv") 

years_int = np.array([eval(i) for i in growth_ratios_df.columns[3:]])
years = [str(y) for y in years_int]

timeseries_df=build_and_fill_timeseries_df(VARIABLES_SPECS,growth_ratios_df,years)



########################################################################
########################## CALIBRATION #################################
########################################################################



endo_var_calibration = timeseries_df_to_endogenous_dict(timeseries_df, years[0], VARIABLES_SPECS)

exo_var_calibration_unsolved = timeseries_df_to_unsolved_year_dict(timeseries_df, years[0], VARIABLES_SPECS)



#System=sys_df(years, growth_ratios, variables_calibration, endo_exo_var_calibration)
#System.parameters_df.to_csv("results/parameters_df_data_sectors.csv")

####### check for errors ########

"""
for k in endo_exo_var_calibration.keys():
    for par in np.array([endo_exo_var_calibration[k]]).flatten() :
        if par < bounds[k][0] or par > bounds[k][1]:
            raise ValueError(f"parameter {k} out of bounds")

for k in variables_calibration.keys():
    for var in  np.array([variables_calibration[k]]).flatten():
        if var < bounds[k][0] or var > bounds[k][1]:
            raise ValueError(f"variable {k} out of bounds")
"""


########################################################################
######################### SYSTEM DEFINITION ############################
########################################################################


def fill_nans(par_value, var_value, key):
    if isinstance(par_value, np.ndarray):
        par_copy = par_value.copy()
        if par_copy.ndim == 2:
            mask_par = np.isnan(par_copy)
            row_idx, col_idx = np.where(mask_par)
            par_copy[row_idx, col_idx] = var_value.flatten()
        else:
            mask_par = np.isnan(par_copy)
            par_copy[mask_par] = var_value
        return par_copy
    else:
        return var_value


def joint_dict(par, var):
    # Create a new dictionary to store the updated values
    d = par.copy()
    # Iterate through keys of A that are also present in B
    for key in var.keys() & par.keys():
        if np.isscalar(var[key]):
            d[key] = fill_nans(par[key], np.array([var[key]]), key)
        else:
            d[key] = fill_nans(par[key], var[key],key)
    return d


def system(var, par):

    d=joint_dict(par,var)
    non_zero_index_pE_Pj=np.array(np.where(d["pE_Pj"] != 0)).flatten()
    non_zero_index_aYE_Bj=np.array(np.where(d["aYE_Bj"] != 0)).flatten()
    non_zero_index_aYE_Pj=np.array(np.where(d["aYE_Pj"] != 0)).flatten()
    non_zero_index_aYE_Tj=np.array(np.where(d["aYE_Tj"] != 0)).flatten()
    index_wo_E=np.delete(np.array(range(N)), E)
    index_E=np.array([E])
    index_wo_E_SE=np.delete(index_wo_E, SE)
    global equations
    equations= {
        ###
        "eqCESquantityKj":eq.eqCESquantity(Xj=d['Kj'], Zj=d['KLj'], alphaXj=d['alphaKj'], alphaYj=d['alphaLj'], pXj=d['pK'], pYj=d['pL'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL']),#e-5
        ###
        "eqCESpriceKL":eq.eqCESprice(pZj=d['pKLj'], pXj=d['pL'], pYj=d['pK'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta = d['bKL'], E_exception=True),
        ###
        "eqYij":eq.eqYij(Yij=d['Yij'], aYij=d['aYij'],Yj=d['Yj'], _index=non_zero_index_Yij),
        ###
        "eqLeontiefVolumes_KL":eq.eqLeontiefVolumes(quantity=d['KLj'],technical_coeff=d['aKLj'],output=d['Yj']),
        ###
        ### da verificare uguaglianza
        ### 
        "eqpYj_E":eq.eqpYj_E(pYj=d['pYj'], pCj=d['pCj'], aKLj=d['aKLj'], pKLj=d['pKLj'], aYij=d['aYij'], pY_Ej=d["pY_Ej"], tauYj=d['tauYj']),
        ###
        "eqCESquantityX":eq.eqCESquantity(Xj=d['Xj'], Zj=d['Yj'] , alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], pXj=d['pXj'], pYj=d['pDj'], sigmaj=d['sigmaXj'], thetaj=d['thetaj'], _index=np.intersect1d(index_wo_E,non_zero_index_X)),
        ###
        "eqCESquantityDy":eq.eqCESquantity(Xj=d['Dj'], Zj=d['Yj'], alphaXj=d['alphaDj'], alphaYj=d['alphaXj'], pXj=d['pDj'], pYj=d['pXj'], sigmaj=d['sigmaXj'],  thetaj=d['thetaj'], _index=index_wo_E),
        ###
        "eqCESquantityDy_E":eq.eqsum_scalar(d['Yj'][E], d['Dj'][E], d['Xj'][E]),
        ###
        "eqpriceYj":eq.eqRevenueCost(p1j=d['pDj'],p2j=d['pXj'],p12j=d['pYj'],V1j=d['Dj'],V2j=d['Xj'],V12j=d['Yj']),
        ###
        "eqCESquantityDs":eq.eqCESquantity(Xj=d['Dj'], Zj=d['Sj'], alphaXj=d['betaDj'], alphaYj=d['betaMj'], pXj=d['pDj'], pYj=d['pMj'], sigmaj=d['sigmaSj'], thetaj=d['csij'], _index=index_wo_E),
        ###
        "eqCESquantityDs_E":eq.eqsum_scalar(d['Sj'][E], d['Dj'][E], d['Mj'][E]),
        ###
        "eqCESquantityM":eq.eqCESquantity(Xj=d['Mj'], Zj=d['Sj'], alphaXj=d['betaMj'], alphaYj=d['betaDj'], pXj=d['pMj'], pYj=d['pDj'], sigmaj=d['sigmaSj'], thetaj=d['csij'], _index=np.intersect1d(index_wo_E,non_zero_index_M)),
        ###
        "eqpriceSj":eq.eqRevenueCost(p1j=d['pDj'],p2j=d['pMj'],p12j=d['pSj'],V1j=d['Dj'],V2j=d['Mj'],V12j=d['Sj']),
        ###
        "eqB":eq.eqB(B=d['B'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        ###
        "eqIDpX":eq.eqID(x=d['pXj'],y=d['pMj'],_index=index_wo_E),
        ###
        "eqCobbDouglasjG":eq.eqCobbDouglasj(Qj=d['Gj'],alphaQj=d['alphaGj'],pCj=d['pCj'],Q=d['Rg'], _index=non_zero_index_G),
        ###
        "eqIj":eq.eqIj(Ij=d['Ij'],alphaIj=d['alphaIj'],I=d['I'],_index=non_zero_index_I),
        ###
        "eqMultRg":eq.eqMultiplication(result=d['Rg'],mult1=d['wG'],mult2=d['GDP']),
        ###
        "eqSj":eq.eqSj(Sj=d['Sj'],Cj=d['Cj'], Gj=d['Gj'], Ij=d['Ij'], Yij=d['Yij']),
        ###
        "eqGDP":eq.eqGDP(GDP=d['GDP'],pCj=d['pCj'],Cj=d['Cj'],Gj=d['Gj'],Ij=d['Ij'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        ###
        "eqGDPPI":eq.eqGDPPI(GDPPI = d['GDPPI'], pCj=d['pCj'], pXj=d['pXj'], pCtp= d['pCtp'], pXtp=d['pXtp'], Cj= d['Cj'], Gj= d['Gj'], Ij= d['Ij'], Xj=d['Xj'], Mj=d['Mj'], Ctp= d['Ctp'], Gtp= d['Gtp'], Itp= d['Itp'], Xtp=d['Xtp'], Mtp=d['Mtp']),
        ###
        "eqGDPreal":eq.eqGDPreal(GDPreal=d['GDPreal'],GDP=d['GDP'], GDPPI=d['GDPPI']), #expected GDPPI time series
        ###
        "eqPriceTaxtauS":eq.eqPriceTax(pGross=d['pCj'], pNet=d['pSj'], tau=d['tauSj'], exclude_idx=E),
        ###
        "eqpI":eq.eqpI(pI=d['pI'],pCj=d['pCj'],alphaIj=d['alphaIj']),
        ###
        "eqMultRi":eq.eqMultiplication(result=d['Ri'],mult1=d['pI'],mult2=d['I']),
        ###


        #energy coupling
        ###
        "eqC_E":eq.eqsum_scalar(d['Cj'][E], d['C_EB'],d['C_ET']),
        ###
        "eqY_E":eq.eqsum_arr(d['Yij'][E,:], d['YE_Pj'], d['YE_Bj'],d['YE_Tj'], d['YE_Ej']  ),
        ###
        "eqpC_E":eq.eqsum_pEYE(p_CE=d["pCj"][E], pY_Ej=d['pY_Ej'], C_E=d['Cj'][E], Y_Ej=d['Yij'][E,:], 
                               pE_B=d['pE_B'], C_EB=d['C_EB'], YE_Bj=d['YE_Bj'], pE_Pj=d['pE_Pj'], 
                               YE_Pj=d['YE_Pj'], pE_TnT=d['pE_TnT'], pE_TT=d['pE_TT'], C_ET=d['C_ET'], 
                               YE_Tj=d['YE_Tj'], pE_Ej=d['pE_Ej'], YE_Ej=d['YE_Ej']),
        ###
        "eqrhoB": eq.eqrho(pEi=d['pE_B'], p_EE=d['pE_Ej'][E], rho=d["rhoB"]), #
        ###
        "eqrhoTT": eq.eqrho(pEi=d['pE_TT'], p_EE=d['pE_Ej'][E], rho=d["rhoTT"]), #
        ###
        "eqrhoTnT": eq.eqrho(pEi=d['pE_TnT'], p_EE=d['pE_Ej'][E], rho=d["rhoTnT"]), #
        ###
        "eqrhoPj": eq.eqrho(pEi=d['pE_Pj'], p_EE=d['pE_Ej'][E], rho=d["rhoPj"],_index=non_zero_index_pE_Pj ), #
        ###
        "eqLeontiefVolumes_YE_B":eq.eqLeontiefVolumes(quantity=d['YE_Bj'],technical_coeff=d['aYE_Bj'],output=d['Yj'],_index=non_zero_index_aYE_Bj),
        ###
        "eqLeontiefVolumes_YE_P":eq.eqLeontiefVolumes(quantity=d['YE_Pj'],technical_coeff=d['aYE_Pj'],output=d['Yj'],_index=non_zero_index_aYE_Pj),
        ###
        "eqLeontiefVolumes_YE_T":eq.eqLeontiefVolumes(quantity=d['YE_Tj'],technical_coeff=d['aYE_Tj'],output=d['Yj'],_index=non_zero_index_aYE_Tj),


        ### CDES
        "eqCj_CDE":eq.eqC_CDE(A_Cj=d["A_Cj_nE"],betaCj=d["betaCj_nE"],u_C=d["u_C"],gammaCj=d["gammaCj_nE"],pCj=np.delete(d["pCj"], E),Cj=np.delete(d["Cj"], E),R=d["R_nE"]),
        ###
        "eq_u_CDE":eq.eq_u_CDE(norm_factor=d["normalisation_factor"], A_Cj=d["A_Cj_nE"],betaCj=d["betaCj_nE"],u_C=d["u_C"],gammaCj=d["gammaCj_nE"],pCj=np.delete(d["pCj"], E),Cj=np.delete(d["Cj"], E),R=d["R_nE"]),
        ###
        "eq_R_E":eq.eq_R_E(R_E=d["R_E"], pC_E=d["pCj"][E], C_E=d["Cj"][E]),
        ###
        "eq_RH_nE":eq.eq_RH_nE(R=d["R"], R_E=d["R_E"], R_nE=d["R_nE"]),
                ###
        "eqpCE":eq.eqsum_pESE(p_SE=d['pSj'][E], tauSE=d['tauSj'][E], S_E=d['Sj'][E], Y_Ej=d['Yij'][E,:], C_E=d['Cj'][E], pY_Ej=d['pY_Ej'], p_CE=d['pCj'][E]),#
        ###
        "eqaKLj0":eq.eqaKLj0(aKLj0=d['aKLj0'], aKLj=d['aKLj'], lambda_KLM=d['lambda_KLM']),
        ###
        'eqaYij0':eq.eqaYij0(aYij0=d['aYij0'], aYij=d['aYij'], lambda_KLM=d['lambda_KLM']),#
        ###
        "eqWorldPrices": eq.eqSameRatio(numerator1=d['pXj'][index_wo_E_SE],numerator2=d['pYj'][index_wo_E_SE],denominator1=d['pXj'][SE],denominator2=d['pYj'][SE]),
        ###
        "eqMultwI":eq.eqMultiplication(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
        ###
        "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL']),#e-5
        ###
        "eqFL":eq.eqF(F=d['L'],Fj=d['Lj']),
        ###
        "eqFK":eq.eqF(F=d['K'],Fj=d['Kj']),
        ###
        "eqMultB":eq.eqMultiplication(result=d['B'],mult1=d['wB'],mult2=d['GDP'])
        

        #"eqAlphaX":eq.eqsum_arr(d['alphaXj'][[ST,CH]], d['alphaXj0'][[ST,CH]], d['lambda_XMj'][[ST,CH]]  ),
        
        #"eqAlphaD":eq.eqsum_arr(d['alphaDj'][[ST,CH]], d['alphaDj0'][[ST,CH]], -d['lambda_XMj'][[ST,CH]]  ),
        
        #"eqBetaD":eq.eqsum_arr(d['betaDj'][[ST,CH]], d['betaDj0'][[ST,CH]], d['lambda_XMj'][[ST,CH]]  ),
        
        #"eqBetaM":eq.eqsum_arr(d['betaMj'][[ST,CH]], d['betaMj0'][[ST,CH]], -d['lambda_XMj'][[ST,CH]]  ),

        #"eqPriceTaxtauL":eq.eqPriceTax(pGross=d['pL'], pNet=d['w'], tau=d['tauL']),

        #"eqCPI":eq.eqCPI(CPI = d['CPI'], pCj=d['pCj'], pCtp= d['pCtp'], Cj= d['Cj'], Ctp= d['Ctp']),
        
        #"eqRreal":eq.eqRreal(Rreal=d['Rreal'],R=d['R'], CPI=d['CPI']), 
        
        #"eqT":eq.eqT(T=d['T'], tauYj=d['tauYj'], pYj=d['pYj'], Yj=d['Yj'], tauSj=d['tauSj'], pSj=d['pSj'], Sj=d['Sj'], tauL=d['tauL'], w=d['w'], Lj=d['Lj']),#
        
        #"eqpYj":eq.eqpYj(pYj=d['pYj'],pCj=d['pCj'],aKLj=d['aKLj'],pKLj=d['pKLj'],aYij=d['aYij'], tauYj=d['tauYj']),
        
        #"eqCESpriceY":eq.eqCESprice(pZj=d['pYj'], pXj=d['pXj'], pYj=d['pDj'], alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], sigmaj=d['sigmaXj'],  thetaj=d['thetaj']),
        
        #"eqCESpriceS":eq.eqCESprice(pZj=d['pSj'],pXj=d['pMj'],pYj=d['pDj'],alphaXj=d['betaMj'],alphaYj=d['betaDj'],sigmaj=d['sigmaSj'], thetaj=d['csij']),
                
        #"eqCobbDouglasjC":eq.eqCobbDouglasj(Qj=d['Cj'],alphaQj=d['alphaCj'],pCj=d['pCj'],Q=d['R'], _index=non_zero_index_C),
        
        # "eqE_P":eq.eqsum_scalar(d['E_P'], d['YE_Pj']),
        
        # "eqE_B":eq.eqsum_scalar(d['E_B'], d['YE_Bj'], d['C_EB']),
        
        # "eqE_T":eq.eqsum_scalar(d['E_T'], d['YE_Tj'], d['C_ET']),
        
        #"eqCj_new":eq.eqCobbDouglasj_lambda(Cj=d['Cj'], alphaCj=d['alphaCj0'],pCj=d['pCj'], R=d['R'], lambda_E=d["lambda_E"], lambda_nE=d["lambda_nE"], _index=non_zero_index_C),#checked
        
        #"eqlambda_nE":eq.eqlambda_nE(alphaCj=d['alphaCj0'],lambda_E=d['lambda_E'], lambda_nE=d['lambda_nE']), #checked

        }
    
    solution = np.hstack(list(equations.values()))
        
        
    return solution







        
########################################################################
########################  CALIBRATION CHECK  ###########################
########################################################################


max_err_cal=max(abs(system(endo_var_calibration, exo_var_calibration_unsolved)))

if max_err_cal>1e-07:
    d=joint_dict(exo_var_calibration_unsolved,endo_var_calibration)
    raise RuntimeError("the system is not correctly calibrated")


########################################################################
##########################  CREATE BOUNDS  #############################
########################################################################


#### set the bounds in the good format for the solver ####
def multiply_bounds_len(key,this_bounds,this_variables):
    return [this_bounds[key] for i in range(len(this_variables[key].flatten()))]

def bounds_dict(this_bounds,this_variables):
    return dict((k, multiply_bounds_len(k,this_bounds,this_variables) ) for k in this_variables.keys())

def flatten_bounds_dict(this_bounds,this_variables):
    return np.vstack(list(bounds_dict(this_bounds,this_variables).values()))


#####  create a reduced dictionary for variables (without zeros) and correspondent set of bounds#####

def to_array(candidate):
    return candidate if isinstance(candidate, np.ndarray) else np.array([candidate])

variables_values = [ to_array(endo_var_calibration[keys])[to_array(endo_var_calibration[keys]) !=0 ] for keys in endo_var_calibration.keys()]

var_keys = list(endo_var_calibration.keys())

non_zero_variables = {var_keys[i]: variables_values[i] for i in range(len(var_keys))}

bounds_variables = [[row[i] for row in flatten_bounds_dict(bounds, non_zero_variables)] for i in (0,1)]

#number_modified=10, percentage_modified=0.1, modification=0.1, seed=4


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

######### HANDLE PARAMETER'S JUMP ##################

def conduct_solution(endo_exo_origin, endo_exo_target, system, bounds_variables, N, threshold= 0.07, growth_rate=0.01):
    print("finding solution progressively ...")

    parameters_gr = jump.dictionary_gr( endo_exo_target,endo_exo_origin  )
    positions= jump.find_positions_above_threshold(parameters_gr, threshold)
    
    endo_vars=timeseries_df_to_endogenous_dict(timeseries_df, years[t-1], VARIABLES_SPECS)
        
    while not jump.are_dicts_equal(endo_exo_origin, endo_exo_target) :

        endo_exo_origin = jump.smooth_par_evolution(endo_exo_origin, endo_exo_target, growth_rate, positions)

        solution = dict_least_squares( system, endo_vars, endo_exo_origin, bounds_variables, N, verb=0, check=True)
        
        endo_vars = solution.dvar
        maxerror=max(abs( system(solution.dvar, endo_exo_origin)))
        if maxerror>1e-06:
            raise RuntimeError(f"the system doesn't converge, maxerror={maxerror}")

    
    return endo_vars



########################################################################################
##################################  SYSTEM SOLUTION  ###################################
########################################################################################


for t in range(len(years)):
    
    print("year: ", years[t])
    
    if t==0:
        endo_vars=endo_var_calibration #kick(endo_var_calibration)

        endo_exo_vars=exo_var_calibration_unsolved 
    else:
        endo_vars=timeseries_df_to_endogenous_dict(timeseries_df, years[t-1], VARIABLES_SPECS) #kick(System.df_to_dict(var=True, t=years[t-1]))
    
        endo_exo_vars=timeseries_df_to_exo_endo_dict(timeseries_df, years[t], VARIABLES_SPECS)
    
    sol = dict_least_squares( system, endo_vars, endo_exo_vars, bounds_variables, N, verb=1, check=True)
        
    maxerror=max(abs( system(sol.dvar, endo_exo_vars)))

    endo_solution=sol.dvar
    d=joint_dict(endo_exo_vars, endo_solution)
    
    if maxerror>1e-06:
        endo_exo_origin=timeseries_df_to_exo_endo_dict(timeseries_df, years[t-1], VARIABLES_SPECS)
        endo_solution=conduct_solution(endo_exo_origin, endo_exo_vars, system, bounds_variables, N, threshold= 0.07, growth_rate=0.01)
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


#System.parameters_df.to_csv(output_file_name)
   

########################################################################
################## USEFUL FUNCTIONS FOR DEBUGGING  #####################
########################################################################



def compare_dictionaries(dict1, dict2):
    if dict1.keys() != dict2.keys():
        print("The dictionaries have different keys")
    else:
        equal_keys = True
        for key in dict1.keys():
            value1 = dict1[key]
            value2 = dict2[key]
            if type(value1) != type(value2):
                equal_keys = False
                print(f"The key '{key}' has different value types")
            elif hasattr(value1, "__len__"):
                if not np.array_equal(value1, value2):
                    equal_keys = False
                    if len(value1) != len(value2):
                        print(f"The key '{key}' has arrays with different lengths")
                    else:
                        unequal_indexes = np.where(value1 != value2)[0]
                        print(f"The key '{key}' has unequal values at indexes: {unequal_indexes}")
                        print(f"Value 1 at unequal indexes: {value1[unequal_indexes]}")
                        print(f"Value 2 at unequal indexes: {value2[unequal_indexes]}")
            else:
                if value1 != value2:
                    equal_keys = False
                    print(f"The key '{key}' has unequal float values")
        if equal_keys:
            print("The dictionaries are equal")


def count_elements(dictionary):
    count = 0
    for value in dictionary.values():
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                count += len(value)
            elif value.ndim == 2:
                count += value.size
        elif isinstance(value, (float, int)):
            count += 1
        else:
            print("Unsupported type:", type(value))
    return count



def filter_nan_values(original_dict):
    new_dict = {}
    
    for key, value in original_dict.items():
        if isinstance(value, np.ndarray):
            if not np.any(np.isnan(value)):
                new_dict[key] = value
        elif np.isnan(value):
            continue
        else:
            new_dict[key] = value
    
    return new_dict

def compare_dictionaries(dict1, dict2):
    unequal_keys = []

    for key in dict1.keys():
        if key in dict2.keys():
            value1 = dict1[key]
            value2 = dict2[key]

            if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
                if not np.array_equal(value1, value2):
                    unequal_keys.append(key)
            elif value1 != value2:
                unequal_keys.append(key)
        else:
            unequal_keys.append(key)

    # Check for keys that exist in dict2 but not in dict1
    for key in dict2.keys():
        if key not in dict1.keys():
            unequal_keys.append(key)

    return unequal_keys


def compare_dictionaries_keys(dictionary1, dictionary2):
    # Find keys present only in dictionary1
    keys_only_in_dictionary1 = set(dictionary1.keys()) - set(dictionary2.keys())

    # Find keys present only in dictionary2
    keys_only_in_dictionary2 = set(dictionary2.keys()) - set(dictionary1.keys())

    return keys_only_in_dictionary1, keys_only_in_dictionary2


def find_keys_with_large_elements(dictionary, threshold=10e-2):
    keys_with_large_elements = []

    for key, value in dictionary.items():
        # Check if at least one element in the array is greater than one
        if isinstance(value, (list, np.ndarray)):
            if np.any(abs(np.array(value)) > threshold):
                keys_with_large_elements.append(key)

    return keys_with_large_elements

def find_keys_with_negative_elements(dictionary, threshold=10e-2):
    keys_with_large_elements = []

    for key, value in dictionary.items():
        # Check if at least one element in the array is greater than one
        if isinstance(value, (list, np.ndarray)):
            if np.any(np.array(value) < threshold):
                keys_with_large_elements.append(key)

    return keys_with_large_elements

def column(matrix, i):
    return [row[i] for row in matrix]






    """
    if endo_Knext:
        equations.update("eqinventory", eq.eqinventory(Knext=d['Knext'], K=d['K'], delta=d['delta'], I=d['I']) )
                      

    if closure=="johansen": 
        equations.update({
                                #"eqsD":eq.eqsD(sD=d['sD'], Ij=d['Ij'], pCj=d['pCj'], Mj=d['Mj'], Xj=d['Xj'], pXj=d['pXj'], GDP=d['GDP']),
                                ###
                                "eqMultwI":eq.eqMultiplication(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
                                ###
                                "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL']),#e-5
                                ###
                                "eqFL":eq.eqF(F=d['L'],Fj=d['Lj']),
                                ###
                                "eqFK":eq.eqF(F=d['K'],Fj=d['Kj']),
                                ###
                                "eqMultB":eq.eqMultiplication(result=d['B'],mult1=d['wB'],mult2=d['GDP'])
                                }
                                )
        
        solution = np.hstack(list(equations.values()))
        
        
        return solution 

    elif closure=="neoclassic":
        equations.update({"eqRi":eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                          "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL']),#e-5
                                      
                          "eqL":eq.eqF(F=d['L'],Fj=d['Lj']),
                                      
                          "eqF":eq.eqF(F=d['K'],Fj=d['Kj']),
                                      
                          "eqMult":eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP'])
                          })
        solution = np.hstack(list(equations.values()))

        return solution 

    elif closure=="kaldorian":
        equations.update({"eqlj":eq.eqlj(l=d['l'], alphalj=d['alphalj'], KLj=d['KLj'], Lj=d['Lj']),
                                      
                           "eqRi": eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                            
                           "eqwI": eq.eqMultiplication(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
                            
                           "eqL": eq.eqF(F=d['L'],Fj=d['Lj']),
                            
                           "eqK": eq.eqF(F=d['K'],Fj=d['Kj']),
                            
                           "eqB": eq.eqMultiplication(result=d['B'],mult1=d['wB'],mult2=d['GDP'])})

        solution = np.hstack(list(equations.values()))

        return solution
    
    elif closure=="keynes-marshall":
        equations.update({"eqRi":eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                          "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL']),#e-5
                            
                          "eqwI":eq.eqMultiplication(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
                            
                          "eqK":eq.eqF(F=d['K'],Fj=d['Kj']),
                            
                          "eqB":eq.eqMultiplication(result=d['B'],mult1=d['wB'],mult2=d['GDP'])})
       
        solution = np.hstack(list(equations.values()))

        return solution


    
    elif closure=="keynes" or closure=="keynes-kaldor" :
        equations.update({"eqlj":eq.eqlj(l=d['l'], alphalj=d['alphalj'], KLj=d['KLj'], Lj=d['Lj']),
                                      
                          "eqRi":eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                        
                          "eqwI":eq.eqMultiplication(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
                            
                          "equL":eq.equ(u=d['uL'], L=d['L'], Lj=d['Lj']),
                            
                          "eqw_real":eq.eqw_real(w_real=d['w_real'], CPI=d['CPI'], w=d['w']),
                            
                          "eqw_curve":eq.eqw_curve(w_real=d['w_real'], alphaw=d['alphaw'], u=d['uL'], sigmaw=d['sigmaw'] ),
                            
                          "eqK":eq.eqF(F=d['K'],Fj=d['Kj']),
                            
                          "eqB":eq.eqMultiplication(result=d['B'],mult1=d['wB'],mult2=d['GDP']),
                          })
        solution = np.hstack(list(equations.values()))

        return solution
    elif closure=="keynes-marshall":
        equations.update({"eqRi":eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                        
                          "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL'], _index=non_zero_index_L),#e-5
                            
                          "eqwI":eq.eqMultiplication(result=d['I'],mult1=d['wI'],mult2=d['GDP']),
                            
                          "eqK":eq.eqF(F=d['K'],Fj=d['Kj']),
                            
                          "eqB":eq.eqMultiplication(result=d['B'],mult1=d['wB'],mult2=d['GDP']),
                          })
        solution = np.hstack(list(equations.values()))

        return solution
    elif closure=="neokeynesian1":
        equations.update({"eqRi":eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                          "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL'], _index=non_zero_index_L),#e-5
                            
                          "equL":eq.equ(u=d['uL'], L=d['L'], Lj=d['Lj']),
                            
                          "eqw_real":eq.eqw_real(w_real=d['w_real'], CPI=d['CPI'], w=d['w']),
                            
                          "eqw_curve":eq.eqw_curve(w_real=d['w_real'], alphaw=d['alphaw'], u=d['uL'], sigmaw=d['sigmaw'] ),
                         
                          "equK":eq.equ(u=d['uK'], L=d['K'], Lj=d['Kj']),
                            
                          "eqpK_real":eq.eqw_real(w_real=d['pK_real'], CPI=d['CPI'], w=d['pK']),
                            
                          "eqsigmapK":eq.eqw_curve(w_real=d['pK_real'], alphaw=d['alphapK'], u=d['uK'], sigmaw=d['sigmapK'] ),
        
                          "eqwB":eq.eqMultiplication(result=d['B'],mult1=d['wB'],mult2=d['GDP']),
                          })                          
        solution = np.hstack(list(equations.values()))

        return solution
    elif closure=="neokeynesian2":
        equations.update({"eqRi":eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                          "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL'], _index=non_zero_index_L),#e-5
                            
                          "equL":eq.equ(u=d['uL'], L=d['L'], Lj=d['Lj']),
                            
                          "eqw_real":eq.eqw_real(w_real=d['w_real'], CPI=d['CPI'], w=d['w']),
                            
                          "eqw_curve":eq.eqw_curve(w_real=d['w_real'], alphaw=d['alphaw'], u=d['uL'], sigmaw=d['sigmaw'] ),
                            
                          "equK":eq.equ(u=d['uK'], L=d['K'], Lj=d['Kj']),
                            
                          "eqpK_real":eq.eqw_real(w_real=d['pK_real'], CPI=d['CPI'], w=d['pK']),
                            
                          "eqsigmapK":eq.eqw_curve(w_real=d['pK_real'], alphaw=d['alphapK'], u=d['uK'], sigmaw=d['sigmapK'] ),
    
                          "eqIneok":eq.eqIneok(I=d['I'], K=d['K'], alphaIK=d['alphaIK'] )
                          })
        solution = np.hstack(list(equations.values()))

        return solution
    else:
        raise ValueError("the closure doesn't exist")
        """