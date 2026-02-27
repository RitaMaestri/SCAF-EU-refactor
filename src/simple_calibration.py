import numpy as np 
import import_GTAP_data as imp
from import_GTAP_data import N,sectors
from solvers import dict_least_squares, dict_minimize
import sys
from copy import deepcopy as cp
import csv
import pandas as pd
import scipy
from functools import partial
import os
import json

#import pandas as pd

A = sectors.index("AGRICULTURE")
M = sectors.index("MANUFACTURE")
SE = sectors.index("SERVICES")
E = sectors.index("ENERGY")
ST = sectors.index("STEEL")
CH = sectors.index("CHEMICAL")
T = sectors.index("TRANSPORTATION")

shares = pd.read_csv("/home/rita/Documents/Tesi/Code/REMIND_energy_coupling/data/shares.csv", index_col="variable")




# Cache directory for expensive calibration parameters
CACHE_DIR = "calibration_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_filename(db_name, param_name):
    """Generate cache filename based on database and parameter name."""
    return os.path.join(CACHE_DIR, f"{db_name}_{param_name}.npy")

def get_cache_metadata_filename(db_name):
    """Generate cache metadata filename to track source database."""
    return os.path.join(CACHE_DIR, f"{db_name}_metadata.json")

def save_expensive_params(db_name, params_dict):
    """
    Save expensive calibration parameters to files.
    
    Parameters
    ----------
    db_name : str
        Database name (e.g., 'GTAP') to include in filename
    params_dict : dict
        Dictionary of parameter_name: parameter_value pairs
    """
    # Save each parameter
    for param_name, param_value in params_dict.items():
        filepath = get_cache_filename(db_name, param_name)
        if isinstance(param_value, (np.ndarray, list)):
            np.save(filepath, param_value)
        elif isinstance(param_value, (int, float)):
            # Save scalar as single-element array for consistency
            np.save(filepath, np.array([param_value]))
        else:
            print(f"Warning: Cannot cache parameter {param_name} of type {type(param_value)}")
    
    # Save metadata
    metadata = {
        'database': db_name,
        'parameters': list(params_dict.keys()),
        'N': N
    }
    with open(get_cache_metadata_filename(db_name), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Cached {len(params_dict)} parameters from {db_name}")





def load_expensive_params(db_name, param_names):
    """
    Load expensive calibration parameters from cache files.
    
    Parameters
    ----------
    db_name : str
        Database name (e.g., 'GTAP')
    param_names : list
        List of parameter names to load
        
    Returns
    -------
    dict or None
        Dictionary of loaded parameters, or None if cache doesn't exist
    """
    params = {}
    metadata_file = get_cache_metadata_filename(db_name)
    
    # Check if metadata exists
    if not os.path.exists(metadata_file):
        return None
    
    # Load and verify metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    if metadata.get('N') != N:
        print(f"Cache N={metadata.get('N')} != current N={N}. Recalculating...")
        return None
    
    # Try to load each parameter
    for param_name in param_names:
        filepath = get_cache_filename(db_name, param_name)
        if not os.path.exists(filepath):
            print(f"Cache file missing for {param_name}. Recalculating all parameters...")
            return None
        
        data = np.load(filepath)
        # Convert single-element array back to scalar if needed
        if data.size == 1 and not isinstance(data, np.ndarray) or (isinstance(data, np.ndarray) and data.shape == (1,)):
            params[param_name] = float(data.flat[0])
        else:
            params[param_name] = data
    
    print(f"Loaded {len(params)} parameters from cache ({db_name})")
    return params



def create_ordered_bounds(variables, bound_dict):
    # Initialize lists for lower and upper bounds
    lower_bounds = []
    upper_bounds = []
    
    # Iterate over the variables in the order of their appearance
    for key in variables:
        # Check if the variable is a scalar or an array
        if np.isscalar(variables[key]):
            var_size = 1  # Scalar has a single bound
        else:
            var_size = len(variables[key])  # Array size
            
        # Extract the bounds from bound_dict
        lower_bound, upper_bound = bound_dict[key]
        
        # Repeat the bounds for each element in the variable (either once or for each array element)
        lower_bounds.extend([lower_bound] * var_size)
        upper_bounds.extend([upper_bound] * var_size)
    
    # Convert lists to numpy arrays and stack them into the required format
    bounds = np.array([lower_bounds, upper_bounds])
    
    return bounds






def _compute_expensive_params(alphaCj0_nE, target_ni_j_nE, target_etaCj_nE, 
                               pCjCj_nE, pCj0_nE, R_nE, N_nE):
    """
    Compute expensive calibration parameters (betaCj, etaCj, A_Cj, etc.).
    This function is separated for caching purposes.
    """
    
    #########################
    #### finding betaCj #####
    #########################
    
    def eqni_j(var,alphaCj,N):
        ni_j=var[:N]
        betaCj=var[N:]
        thetaCj =  1 - betaCj
        term1 = thetaCj * (2 * alphaCj - 1)
        term2 = - alphaCj * (np.sum(alphaCj * thetaCj))
        zero = -1 + ni_j / (term1 + term2)
        return zero
    
    constraint_with_params = partial(eqni_j, alphaCj=alphaCj0_nE, N=N_nE)
    
    ni_j_constraint = scipy.optimize.NonlinearConstraint(
        fun=constraint_with_params,
        lb=-1e-14,  
        ub=1e-14
    )

    
    def eq_difference(alphaCj0, el,target_el):
        result = np.sqrt(alphaCj0)*(el-target_el)
        return result
    
    def systemNi(var, par):
        d = {**var, **par}
        return eq_difference(d["alphaCj0"], d["ni_j"],d["target_ni_j"])
    
    
    variables = { "betaCj": np.array([float(-0.5)]*N_nE),
                 "ni_j":target_ni_j_nE*0.99}
    
    parameters={'target_ni_j':target_ni_j_nE,
                'alphaCj0':alphaCj0_nE
                }
    
    #beta must be either between 0 and 1 or less than 0 for every j
    bound_dict = {
        "betaCj": [-np.inf, -0.001],
        "ni_j": [-np.inf, np.inf],
    }
    
    bounds = create_ordered_bounds(variables, bound_dict)
    
    solNi_constrained = dict_minimize(systemNi, variables , parameters, N_nE, bounds, constraint=ni_j_constraint )
    
    systemNi(solNi_constrained.dvar, parameters)
    
    computed_ni_j_nE=solNi_constrained.dvar["ni_j"]
    betaCj_nE=solNi_constrained.dvar["betaCj"]
    
    eqni_j(solNi_constrained.x,alphaCj0_nE,N_nE)
    ni_calibration_error=(computed_ni_j_nE-target_ni_j_nE)/target_ni_j_nE

    #########################
    #### finding gammaCj #####
    #########################
    
    #### CONSTRAINT 1 ######
    
    def eq_etaCj(gammaCj, alphaCj0, betaCj):
        
        # Compute thetaCj as 1 - betaCj
        thetaCj = 1 - betaCj
    
        # Compute the summation terms
        sum_alpha_gamma = np.sum(alphaCj0 * gammaCj)
        sum_alpha_gamma_theta = np.sum(alphaCj0 * gammaCj * thetaCj)
        sum_alpha_theta = np.sum(alphaCj0 * thetaCj)
       # Compute the etaCj array using vectorized operations
        etaCj = (1 / sum_alpha_gamma) * (gammaCj * (1 - thetaCj) + sum_alpha_gamma_theta) + thetaCj - sum_alpha_theta
         
        return etaCj
    
    
    def eq_constraint_etaCj(var, alphaCj0, betaCj, N):
        gammaCj=var[:N]
        etaCj=var[N:]
        computed_etaCj=eq_etaCj(gammaCj, alphaCj0, betaCj)
        zero=1-computed_etaCj/etaCj
        return zero
        
    eq_constraint_etaCj_with_params = partial(eq_constraint_etaCj, alphaCj0=alphaCj0_nE, betaCj=betaCj_nE, N=N_nE)

    scipy_etaCj_constraint = scipy.optimize.NonlinearConstraint(
        fun=eq_constraint_etaCj_with_params,
        lb=-1e-12,
        ub=+1e-12,
        keep_feasible=False
    )
    

    
    ####### CONSTRAINT 2 ######
    
    def sign_eq(var, target_etaCj, N):
        gammaCj=var[:N]
        etaCj=var[N:]
        
        # Compute the value for the last N positions
        positive = (etaCj - 1) * (target_etaCj - 1)
        
        return positive
    

    sign_constraint_with_params = partial(sign_eq, target_etaCj=target_etaCj_nE, N=N_nE)

    scipy_sign_constraint = scipy.optimize.NonlinearConstraint(
        fun=sign_constraint_with_params,
        lb=0,  
        ub=np.inf 
    )
    

    
    #### OPTIMIZATION EQAUATION ######
    
    def systemEta_j(var, par):
        d = {**var, **par}
        return eq_difference(d["alphaCj0"],d["etaCj"],d["target_etaCj"])

    guess_gammaCj=np.array(4.+8.*1/np.array(range(1,7)))
    guess_etaCj= eq_etaCj(guess_gammaCj, alphaCj0_nE, betaCj_nE)+1e-2
    
    variables = { "gammaCj": guess_gammaCj,
                  'etaCj':guess_etaCj}
    
    parameters={'target_etaCj':target_etaCj_nE,
                'betaCj':betaCj_nE,
                'alphaCj0':alphaCj0_nE, 
                }
    
    bounds_dict={ "gammaCj": [0,np.inf],
                  'etaCj': [0, np.inf]}
    
    
    bounds = create_ordered_bounds(variables, bounds_dict)


    solEtaj_constrained = dict_minimize(systemEta_j, variables , parameters, N_nE, bounds, constraint=[scipy_etaCj_constraint, scipy_sign_constraint ] )
    
    computed_etaCj_nE=solEtaj_constrained.dvar["etaCj"]
    gammaCj_nE=solEtaj_constrained.dvar["gammaCj"]
    
    
    eq_etaCj(solEtaj_constrained.dvar["gammaCj"], alphaCj0_nE, betaCj_nE)-solEtaj_constrained.dvar["etaCj"]

    
    etai_calibration_error=(computed_etaCj_nE-target_etaCj_nE)/target_etaCj_nE
    
    ##################################
    ########## finding A_Cj ##########
    ##################################
    
    u_C=1
    
    def eq_A_Cj(pCjCj,A_Cj,betaCj,u_C,gammaCj,pCj,R0):
        Z_j = A_Cj * betaCj * u_C ** (gammaCj * betaCj) * (pCj / R0) ** betaCj
        zero=1-pCjCj/(R0*Z_j/sum(Z_j))
        return zero
    
    
    def systemA_C(var, par):
        d = {**var, **par}
        return eq_A_Cj(d["pCjCj"],d["A_Cj"],d["betaCj"],d["u_C"],d["gammaCj"],d["pCj"],d["R0"])
    
    
    variables = { 'A_Cj': np.array([0.3]*N_nE)}

    R_nE_local = R_nE  # Get from closure
    parameters={'pCjCj':pCjCj_nE,
                'betaCj':betaCj_nE, 
                'u_C':u_C,
                "gammaCj":gammaCj_nE,
                "pCj":pCj0_nE,
                "R0":R_nE_local
                }
    
    bounds_dict= { 'A_Cj': [0,np.inf],
               }
    
    bounds= create_ordered_bounds(variables, bounds_dict)
    
    solA_Cj = dict_least_squares(systemA_C, variables , parameters, bounds, N, check=False,verb=0)
    
    A_Cj_nE=solA_Cj.dvar["A_Cj"]
    
    normalisation_factor= sum(A_Cj_nE * u_C ** (gammaCj_nE * betaCj_nE) * (pCj0_nE / R_nE_local) ** betaCj_nE)
    
    return {
        'betaCj_nE': betaCj_nE,
        'computed_ni_j_nE': computed_ni_j_nE,
        'computed_etaCj_nE': computed_etaCj_nE,
        'gammaCj_nE': gammaCj_nE,
        'u_C': u_C,
        'A_Cj_nE': A_Cj_nE,
        'normalisation_factor': normalisation_factor
    }


def _load_energy_matrix_from_csv(df, variable_type, row_labels, col_map, default_fill=0.0):
    """Build an (n_rows, 4) energy matrix from a calibration CSV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full calibration DataFrame read from calibration_2020.csv.
    variable_type : str
        Value to match on the ``Variable`` column (e.g. ``"Volume"``,
        ``"Price"``, ``"Rho"``, ``"Technical_coefficient"``).
    row_labels : list of str
        Ordered row names; length determines the number of matrix rows.
    col_map : dict
        Maps ``"Energy uses"`` strings found in the CSV to 0-based column
        indices.  Entries whose energy-use label is absent from *col_map* are
        ignored.
    default_fill : float
        Value used to pre-fill the matrix before CSV values are inserted.

    Returns
    -------
    np.ndarray, shape ``(len(row_labels), 4)``
    """
    mat = np.full((len(row_labels), 4), default_fill, dtype=float)
    # Find the year-2020 column label (int or str depending on pandas version)
    year_col = next((c for c in df.columns if str(c) == "2020"), None)
    if year_col is None:
        raise ValueError("Column '2020' not found in calibration CSV.")
    subset = df[df["Variable"] == variable_type]
    for _, row in subset.iterrows():
        consumer   = row["Energy consumers"]
        energy_use = row["Energy uses"]
        if consumer in row_labels and energy_use in col_map:
            r = row_labels.index(consumer)
            c = col_map[energy_use]
            mat[r, c] = float(row[year_col])
    return mat


class calibrationVariables:
    
    def __init__(self, L0=None):
        
        #labor
        if L0 is None:   
            self.pL0 = 1
            self.Lj0= imp.pLLj / cp(self.pL0)
            self.L0=sum(cp(self.Lj0))
        else:
            self.L0=L0
            self.pL0=sum(imp.pLLj)/L0
            self.Lj0 = imp.pLLj / cp(self.pL0)
        
        
        #prezzi
        
        self.pYj0=np.array([float(1000)]*N)
        self.pSj0=np.array([float(1000)]*N)
        self.pKLj0=np.array([float(1000)]*N)
        self.pXj0=np.array([float(1000)]*N)
        self.pDj0=np.array([float(1000)]*N)
        self.pXj=np.array([float(1000)]*N)
        self.pMj0=cp(self.pXj0)
        self.pXj0=cp(self.pXj)
        
        #taxes
        
        self.tauYj0 = imp.production_taxes/( imp.pYjYj - imp.production_taxes)

        self.tauSj0 = imp.sales_taxes / (imp.pCiYij.sum(axis=1)+imp.pCjCj+imp.pCjGj+imp.pCjIj - imp.sales_taxes)
        self.pCj0 = (1+cp(self.tauSj0))*cp(self.pSj0)
        
        #quantità
        self.Ij0 = imp.pCjIj/ cp(self.pCj0)
        self.Cj0 = imp.pCjCj/ cp(self.pCj0)
        self.Gj0 = imp.pCjGj/ cp(self.pCj0)

        self.Yij0 = imp.pCiYij/ cp(self.pCj0[:,None])
        self.KLj0= imp.pKLjKLj / cp(self.pKLj0)
        self.Xj0= imp.pXjXj / cp(self.pXj0)
        self.Mj0= imp.pMjMj / cp(self.pMj0)
        self.Dj0= imp.pDjDj / cp(self.pDj0)
        self.Yj0= imp.pYjYj / cp(self.pYj0)        
        self.Sj0= imp.pSjSj / cp(self.pSj0)        
        
        
        
        
        #adjusting energy quantity and price

        self.Sj0[E]=91.9143818
        self.pSj0[E]=imp.pSjSj[E] / cp(self.Sj0[E])

        self.Dj0[E]=cp(self.Sj0[E])-cp(self.Mj0[E])
        self.pDj0[E]=imp.pDjDj[E] / cp(self.Dj0[E])
        
        self.Yj0[E]=cp(self.Xj0[E])+cp(self.Dj0[E])
        self.pYj0[E]=imp.pYjYj[E] / cp(self.Yj0[E])
        
        #scalari

        self.B0=sum(imp.pXjXj)-sum(imp.pMjMj)
        self.R0= sum(imp.pCjCj)
        self.Ri0= sum(imp.pCjIj)
        self.Rg0= sum(imp.pCjGj)
        self.GDP0= sum(imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pXjXj-imp.pMjMj)
        

        
        # elasticities 
        self.sigmaXj=imp.sigmaXj.astype(float)
        self.sigmaSj=imp.sigmaSj.astype(float)
        self.sigmaKLj=imp.sigmaKLj.astype(float)


        self.etaSj=(imp.sigmaSj-1)/imp.sigmaSj
        self.etaXj=(imp.sigmaXj-1)/imp.sigmaXj
        self.etaKLj=(imp.sigmaKLj-1)/imp.sigmaKLj

        
        self.aKLj= cp(self.KLj0)/ cp(self.Yj0)
        
        def compute_alphas_CES(Q1j,Q2j,p1j,p2j,etaj):
            alphaj = 1 / (
                1 + np.float_power( Q2j , (1 - etaj) ) * p2j / ( np.float_power( Q1j , 1-etaj ) * p1j )
                )
            return alphaj
        
        def compute_theta_CES(Zj,alpha1j,alpha2j,Q1j,Q2j,etaj):
            thetaj = Zj / ( 
                np.float_power( 
                    alpha1j * np.float_power(Q1j,etaj) + alpha2j * np.float_power( Q2j,etaj) ,
                    1/etaj )
                )
            
            return thetaj
        

        self.alphaXj= compute_alphas_CES(Q1j= cp(self.Xj0),Q2j= cp(self.Dj0),p1j= cp(self.pXj0),p2j= cp(self.pDj0),etaj= cp(self.etaXj))
        self.alphaDj= compute_alphas_CES(Q1j= cp(self.Dj0),Q2j= cp(self.Xj0),p1j= cp(self.pDj0),p2j= cp(self.pXj0),etaj= cp(self.etaXj))
        self.alphaDj0=cp(self.alphaDj)
        self.alphaXj0=cp(self.alphaXj)
        
        self.thetaj = compute_theta_CES(Zj= cp(self.Yj0), alpha1j= cp(self.alphaXj), alpha2j= cp(self.alphaDj), Q1j= cp(self.Xj0), Q2j= cp(self.Dj0),etaj= cp(self.etaXj))
        
        self.betaMj= compute_alphas_CES(Q1j= cp(self.Mj0),Q2j= cp(self.Dj0),p1j= cp(self.pMj0),p2j= cp(self.pDj0),etaj= cp(self.etaSj))
        self.betaDj= compute_alphas_CES(Q1j= cp(self.Dj0),Q2j= cp(self.Mj0),p1j= cp(self.pDj0),p2j= cp(self.pMj0),etaj= cp(self.etaSj))
        self.betaDj0=cp(self.betaDj)
        self.betaMj0=cp(self.betaMj)
        
        self.csij = compute_theta_CES(Zj= cp(self.Sj0),alpha1j= cp(self.betaMj),alpha2j= cp(self.betaDj),Q1j= cp(self.Mj0),Q2j= cp(self.Dj0),etaj= cp(self.etaSj))
        
        self.alphaCj0 = imp.pCjCj / cp(self.R0)
        self.alphaGj = imp.pCjGj/ cp(self.Rg0)
        self.wB = cp(self.B0)/ cp(self.GDP0)
        self.wG = cp(self.Rg0)/ cp(self.GDP0)
        self.wI = cp(self.Ri0)/ cp(self.GDP0)
        self.GDPreal= cp(self.GDP0)
        self.pXtp= cp(self.pXj)
        self.Gtp= cp(self.Gj0)
        self.Itp= cp(self.Ij0)
        self.pXtp= cp(self.pXj)
        self.Xtp= cp(self.Xj0)
        self.Mtp = cp(self.Mj0)
        
        #calibrate alphaIj, I and pI
        
        def eqpI(pI,pCj,alphaIj):
            zero= -1+ pI / sum(pCj*alphaIj)
            return zero
        
        def eqIj(Ij,alphaIj,I):
            zero= -1+Ij/(alphaIj*I)
            return zero
        
        def eqRi(Ri,pI,I):
            zero= - 1 + Ri / (pI*I)
            return zero
        
        def systemI(var, par):
            d = {**var, **par}
            return np.hstack([eqpI(pI=d['pI'],pCj=d['pCj'],alphaIj=d['alphaIj']),
                              eqIj(Ij=d['Ij'], alphaIj=d['alphaIj'],I=d['I']),
                              eqRi(Ri=d['Ri'],pI=d['pI'],I=d['I'])]
                              )
        
        len_alphaIj=len(imp.pCjIj[imp.pCjIj!=0])
        
        variables = { 'I': self.Ri0/sum(np.array([0.02]*N)*self.pCj0),
           'alphaIj': np.array([10]*len_alphaIj),
           'pI': np.array([sum(np.array([0.02]*N)*self.pCj0)])
           }

        
        parameters={'pCj':self.pCj0[imp.pCjIj!=0],
                    'Ij':self.Ij0[imp.pCjIj!=0], 
                    'Ri':self.Ri0
                    }
        
        bounds_dict= { 'I': [0,np.inf],
                   'alphaIj': [0,np.inf],
                   'pI': [0,np.inf]
                   }
        
        bounds= create_ordered_bounds(variables, bounds_dict)
        
        solI = dict_least_squares(systemI, variables , parameters, bounds, N, check=False,verb=0)
        
        self.I0=float(solI.dvar['I'])
        self.pI0=float(solI.dvar['pI'])
        self.alphaIj=np.zeros(N)
        self.alphaIj[imp.pCjIj!=0]=solI.dvar['alphaIj']
        self.delta=0.04
        self.g0=-0.019215761298272
        self.pK0 = (sum(imp.pKKj)*(cp(self.g0)+cp(self.delta)))/ cp(self.I0)
        self.Kj0= imp.pKKj / cp(self.pK0)
        self.K0=sum(self.Kj0)
        
        self.GDPPI=1
        self.alphaLj= compute_alphas_CES(Q1j= cp(self.Lj0),Q2j= cp(self.Kj0),p1j= cp(self.pL0),p2j= cp(self.pK0),etaj= cp(self.etaKLj))
        self.alphaKj= compute_alphas_CES(Q1j= cp(self.Kj0),Q2j= cp(self.Lj0),p1j= cp(self.pK0),p2j= cp(self.pL0),etaj= cp(self.etaKLj))
        # #this is 1 by default for the E sector so calibrate accordingly
        self.bKL=1
        self.bKLj = cp(self.KLj0)*cp(self.bKL)/np.float_power(cp(self.alphaLj)*np.float_power(cp(self.Lj0),cp(self.etaKLj)) + cp(self.alphaKj) * np.float_power(cp(self.Kj0),cp(self.etaKLj)), 1/ cp(self.etaKLj))
        self.target_ni_j=imp.ni_j
        self.target_etaCj=imp.etaCj
        
        #take off energy sector
        
        self.target_ni_j_nE=np.delete(self.target_ni_j, E)
        self.target_etaCj_nE=np.delete(self.target_etaCj, E)
        self.alphaCj0_nE=np.delete(self.alphaCj0, E)
        self.alphaCj0_nE=self.alphaCj0_nE/sum(self.alphaCj0_nE)
        
        self.pCjCj_nE=np.delete(imp.pCjCj, E)
        self.pCj0_nE=np.delete(self.pCj0, E)
        
        self.R_E = imp.pCjCj[E]
        self.R_nE = self.R0 - self.R_E
        

        
        
        N_nE=N-1
        
        #########################
        #### LOAD FROM CACHE OR COMPUTE EXPENSIVE PARAMETERS ####
        #########################
        
        # Database identifier for cache files
        db_name = "GTAP"
        expensive_params_names = ['betaCj_nE', 'computed_ni_j_nE', 'computed_etaCj_nE', 
                                   'gammaCj_nE', 'u_C', 'A_Cj_nE', 'normalisation_factor']
        
        cached_params = load_expensive_params(db_name, expensive_params_names)
        
        if cached_params is not None:
            # Load from cache
            self.betaCj_nE = cached_params['betaCj_nE']
            
            self.gammaCj_nE = cached_params['gammaCj_nE']
            self.u_C = cached_params['u_C']
            self.A_Cj_nE = cached_params['A_Cj_nE']
            self.normalisation_factor = cached_params['normalisation_factor']
        else:
            # Compute expensive parameters
            computed_params = _compute_expensive_params(
                self.alphaCj0_nE, self.target_ni_j_nE, self.target_etaCj_nE,
                self.pCjCj_nE, self.pCj0_nE, self.R_nE, N_nE
            )
            
            # Assign to self
            self.betaCj_nE = computed_params['betaCj_nE']
            
            self.gammaCj_nE = computed_params['gammaCj_nE']
            self.u_C = computed_params['u_C']
            self.A_Cj_nE = computed_params['A_Cj_nE']
            self.normalisation_factor = computed_params['normalisation_factor']
            
            # Save to cache
            save_expensive_params(db_name, computed_params)
        
        #########################
        #### ENERGY COUPLING ####
        #########################

        #i don't have to determine pC_E because it is endogenously determined.
        sE_P = float(shares.loc["sE_P"])#from excel
        sE_T = float(shares.loc["sE_T"])
        sE_B = float(shares.loc["sE_B"])
        sY_E_PE = float(shares.loc["sY_E_PE"])
        
        S_E=cp(self.Sj0[E])
        self.E_P = sE_P*S_E
        self.E_B= sE_B*S_E
        self.E_T=sE_T*S_E
        self.Y_EE = sY_E_PE * S_E
        
        #PRIMARY ENERGY
        self.YE_Ej = np.array([float(0)]*N)
        self.YE_Ej[E] = self.Y_EE
        
        #ENERGY FOR PROCESSES
        self.YE_Pj = np.array([float(0)]*N)
        self.YE_Pj[A]=float(shares.loc["YE_Pj_A"])*cp(self.E_P)
        self.YE_Pj[CH]=float(shares.loc["YE_Pj_CH"])*cp(self.E_P)
        self.YE_Pj[ST]=float(shares.loc["YE_Pj_ST"])*cp(self.E_P)
        self.YE_Pj[M]=float(shares.loc["YE_Pj_M"])*cp(self.E_P)


        #ENERGY FOR TRANSPORT
        self.YE_Tj = np.array([float(0)]*N)        
        self.s_LDV = float(shares.loc["s_LDV"])
        self.s_trucks = float(shares.loc["s_trucks"])
        self.s_other_transport = float(shares.loc["s_other_transport"])

        self.s_LDV_C = float(shares.loc["s_LDV_C"])
        self.s_LDV_T = float(shares.loc["s_LDV_T"])
        
        self.s_trucks_T = float(shares.loc["s_trucks_T"])
        
        self.YE_Tj[T]=(self.s_LDV*self.s_LDV_T+self.s_trucks*self.s_trucks_T+self.s_other_transport)*cp(self.E_T)
        self.C_ET = self.s_LDV_C * (self.s_LDV * cp(self.E_T))
        nonT = [A,M,SE,ST,CH,E]
        for i in nonT:    
            self.YE_Tj[i]=cp(self.KLj0[i]) / cp(self.KLj0[nonT].sum()) * (cp(self.E_T)-cp(self.YE_Tj[T])-cp(self.C_ET)) 
        
        #ENERGY FOR BUILDINGS
        self.YE_Bj = np.array([float(0)]*(N))
        sC_EB = float(shares.loc["sC_EB"])
        self.C_EB = sC_EB*cp(self.E_B)
        self.YE_Bj[SE]= (1-sC_EB)*cp(self.E_B)
        
        #energy volumes
        self.Cj0[E] = cp(self.C_EB)+cp(self.C_ET)
        for j in [A,M,CH,ST]:
            self.Yij0[E,j] =  cp(self.YE_Pj[j]) +  cp(self.YE_Tj[j])
        self.Yij0[E,SE] =  cp(self.YE_Bj[SE]) +  cp(self.YE_Tj[SE])
        self.Yij0[E,T] =  cp(self.YE_Tj[T])
        self.Yij0[E,E] =  cp(self.YE_Ej[E]) + cp(self.YE_Tj[E])

        #energy technical coefficients
        self.aYE_Bj = cp(self.YE_Bj) / cp(self.Yj0)
        self.aYE_Pj = cp(self.YE_Pj) / cp(self.Yj0)
        self.aYE_Tj = cp(self.YE_Tj) / cp(self.Yj0)
        self.aYE_Ej = cp(self.YE_Ej) / cp(self.Yj0)

        #non-zero indices for energy technical coefficients
        self.non_zero_index_aYE_Bj = np.array(np.where(self.aYE_Bj != 0)).flatten()
        self.non_zero_index_aYE_Pj = np.array(np.where(self.aYE_Pj != 0)).flatten()
        self.non_zero_index_aYE_Tj = np.array(np.where(self.aYE_Tj != 0)).flatten()
        
        ############ ENERGY PRICES   #############
        
        #consumption
        self.pY_Ej = np.array([float(0)]*(N))
        for i in [A,M,SE,ST,CH,E,T]:
            self.pY_Ej[i] = imp.pCiYij[E,i]/cp(self.Yij0[E,i])
        # self.pY_Ej[-1] = imp.pCjCj[E]/cp(self.Cj0[E])
        
        self.pCj0[E] = imp.pCjCj[E]/cp(self.Cj0[E])
        
        #transport
        self.pE_TT = self.pY_Ej[T]
        self.pE_TnT = (( cp(self.C_EB)*imp.pCiYij[E,SE] - imp.pCjCj[E] * cp(self.YE_Bj[SE]) ) /
                 ( cp(self.C_EB) * cp(self.YE_Tj[SE]) - cp(self.C_ET) * cp(self.YE_Bj[SE] ) ) )
        
        #buildings
        self.pE_B = 1 / cp(self.C_EB) * ( imp.pCjCj[E] - cp(self.C_ET) * cp(self.pE_TnT) )
        
        #processes
        self.pE_Pj = np.array([float(0)]*(N))
        for i in [A,M,ST,CH]:
            self.pE_Pj[i] = ( imp.pCiYij[E,i] - cp(self.pE_TnT) * cp(self.YE_Tj[i]) ) / cp(self.YE_Pj[i])
        
        #primary energy
        self.pE_Ej = np.array([float(0)]*(N))
        self.pE_Ej[E] = (imp.pCiYij[E,E] - cp(self.pE_TnT) * cp(self.YE_Tj[E]) ) / cp(self.Y_EE)
        

        #adjusted variables
        self.aYij= cp(self.Yij0) / cp(self.Yj0[None,:])
        
        self.pCjtp= cp(self.pCj0)
        self.Ctp= cp(self.Cj0)
        
        
        self.aKLj0=cp(self.aKLj)
        self.aYij0=cp(self.aYij)
        
        self.rhoB=cp(self.pE_B)/cp(self.pE_Ej[E])
        self.rhoTT=cp(self.pE_TT)/cp(self.pE_Ej[E])
        self.rhoTnT=cp(self.pE_TnT)/cp(self.pE_Ej[E])
        self.rhoPj= cp(self.pE_Pj)/cp(self.pE_Ej[E])

        #### ENERGY MATRICES ####

        # Load calibration CSV once for matrix initialisation
        _cal_csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "calibration_2020.csv")
        _df_cal = pd.read_csv(_cal_csv_path)
        _row_labels = sectors + ["HOUSEHOLDS"]

        # column order: T=0, B=1, P=2, PE=3
        _col_map = {"T": 0, "B": 1, "P": 2, "PE": 3}

        self.E_vol = _load_energy_matrix_from_csv(
            _df_cal, "Volume", _row_labels, _col_map, default_fill=0.0)

        self.pE    = _load_energy_matrix_from_csv(
            _df_cal, "Price", _row_labels, _col_map, default_fill=0.0)

        # Rho has no PE rows in CSV; PE column defaults to 1.0
        self.rhoE  = _load_energy_matrix_from_csv(
            _df_cal, "Rho", _row_labels, _col_map, default_fill=0.0)
        self.rhoE[:, 3] = 1.0

        # Technical_coefficient: computed from intermediate variables (same as aYE_*j / Yj0).
        # Columns: T=0, B=1, P=2, PE=3.  Last row (HOUSEHOLDS) = 0 by construction.
        self.a_Ej = np.zeros((len(_row_labels), 4))
        self.a_Ej[:N, 0] = cp(self.YE_Tj) / cp(self.Yj0)   # T
        self.a_Ej[:N, 1] = cp(self.YE_Bj) / cp(self.Yj0)   # B
        self.a_Ej[:N, 2] = cp(self.YE_Pj) / cp(self.Yj0)   # P
        self.a_Ej[:N, 3] = cp(self.YE_Ej) / cp(self.Yj0)   # PE

        self.Yij0[E,:] = cp(self.E_vol.sum(axis=1)[:-1])
        self.Cj0[E] = cp(self.E_vol.sum(axis=1)[-1])

        self.pCj0[E] = imp.pCjCj[E]/cp(self.Cj0[E])
        self.pY_Ej = imp.pCiYij[E,:]/cp(self.Yij0[E,:])

        self.lambda_KLM = 1

        # Remove individual energy variables now consolidated into the matrices above
        del self.YE_Pj, self.YE_Ej, self.YE_Tj, self.YE_Bj
        del self.C_ET, self.C_EB
        del self.pE_TT, self.pE_TnT, self.pE_B, self.pE_Pj, self.pE_Ej
        del self.rhoB, self.rhoTT, self.rhoTnT, self.rhoPj
        del self.aYE_Bj, self.aYE_Pj, self.aYE_Tj, self.aYE_Ej
        del self.non_zero_index_aYE_Bj, self.non_zero_index_aYE_Pj, self.non_zero_index_aYE_Tj

        # self.lambda_XMj=np.array([float(0)]*N)
        # self.Rh_E = imp.pCjCj[E]
        # self.computed_ni_j_nE = cached_params['computed_ni_j_nE']
        # self.computed_etaCj_nE = cached_params['computed_etaCj_nE']
        # self.computed_ni_j_nE = computed_params['computed_ni_j_nE']
        # self.computed_etaCj_nE = computed_params['computed_etaCj_nE']
        # self.s_trucks_nonT = float(shares.loc["s_trucks_nonT"])
        # self.s_LDV_nonT = float(shares.loc["s_LDV_nonT"])
        # self.gammaj = compute_theta_CES(Zj= cp(self.KLj0), alpha1j= cp(self.alphaKj), alpha2j= cp(self.alphaLj), Q1j= cp(self.Kj0), Q2j= cp(self.Lj0),etaj= cp(self.etaKLj))
        # self.alphapK = cp(self.pK0)/(cp(self.uK0)**cp(self.sigmapK))
        # self.alphaIK = cp(self.Ri0)/ cp(self.K0)
        # self.K0next = cp(self.K0) * (1-cp(self.delta)) + cp(self.I0)
        # self.L0u=sum(self.Lj0)/(1-cp(self.uL0))
        # self.K0u=sum(self.Kj0)/(1-cp(self.uK0))
        # self.K0u_next= cp(self.K0u) * (1-cp(self.delta)) + cp(self.I0)
        #self.betaRj= (imp.epsilonPCj+1)/(self.alphaCj-1)
        #self.epsilonRj=imp.epsilonRj
        # self.sD0=sum(imp.pCjIj+imp.pXjXj-imp.pMjMj)/ cp(self.GDP0)
        # self.alphalj = cp(self.Lj0)/(cp(self.KLj0)*cp(self.l0))
        # self.alphaw = cp(self.w)/(cp(self.uL0)**cp(self.sigmaw))
        # self.lambda_E = 1
        # self.lambda_nE = 1
        # self.lambda_KL = 1
        #self.etaXj=(imp.sigmaXj-1)/imp.sigmaXj
        ### aternative closures
        # self.uL0 = 0.105
        # self.sigmaw= 0.
        # self.uK0 = 0.105
        # self.sigmapK= -0.1
        # self.T0= sum(imp.production_taxes + imp.sales_taxes + imp.labor_taxes)
        # self.w= cp(self.pL0)/(1+cp(self.tauL0))
        # self.tauL0 = imp.labor_taxes / (imp.pLLj - imp.labor_taxes)
        # self.l0=sum(cp(self.Lj0)/ cp(self.KLj0))
        
        # self.CPI=1
        # self.Rh_nE = self.R0 - self.Rh_E
        



#a=calibrationVariables(0.003985893420850095)


# export_calib_dict={
#         "YE_T_A" : a.YE_Tj[A],
#         "YE_T_M" : a.YE_Tj[M],
#         "YE_T_SE" : a.YE_Tj[SE],
#         "YE_T_E" : a.YE_Tj[E],
#         "YE_T_ST" : a.YE_Tj[ST],
#         "YE_T_CH" : a.YE_Tj[CH],
#         "YE_T_T" : a.YE_Tj[T],
#         "C_ET": a.C_ET,
#         "E_T":a.E_T,
#         "s_LDV": a.s_LDV,
#         "s_trucks" : a.s_trucks,
#         "s_other_transport" : a.s_other_transport,
#     }

# with open('transport_calibration.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
#     w = csv.DictWriter(f, export_calib_dict.keys())
#     w.writeheader()
#     w.writerow(export_calib_dict)

