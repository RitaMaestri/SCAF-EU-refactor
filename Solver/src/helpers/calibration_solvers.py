
import numpy as np 
from helpers.solvers import dict_least_squares, dict_minimize
from Variables_specs import VARIABLES_SPECS
import scipy
from functools import partial
import os
import json

# Cache directory for CDES calibration parameters
CACHE_DIR = "Solver/preprocessed_data/calibration/calibration_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_filename(db_name, param_name):
    """Generate cache filename based on database and parameter name."""
    return os.path.join(CACHE_DIR, f"{db_name}_{param_name}.npy")

def get_cache_metadata_filename(db_name):
    """Generate cache metadata filename to track source database."""
    return os.path.join(CACHE_DIR, f"{db_name}_metadata.json")

def save_expensive_params(db_name, params_dict, N, mask_sig=None):
    """
    Save expensive calibration parameters to files.

    Parameters
    ----------
    db_name : str
        Database name (e.g., 'GTAP') to include in filename
    params_dict : dict
        Dictionary of parameter_name: parameter_value pairs
    N : int
        Number of sectors, stored in cache metadata for invalidation.
    mask_sig : array-like of bool, optional
        Boolean mask whose nonzero-index signature is stored in the metadata
        for cache-invalidation purposes (e.g. ``pCjIj != 0``).  When
        provided on load, the stored signature is validated against the
        current mask before accepting cached values.
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

    # Merge with existing metadata so multiple save calls do not wipe each
    # other's parameter lists or validation fields.
    metadata_file = get_cache_metadata_filename(db_name)
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        existing_params = set(metadata.get('parameters', []))
        existing_params.update(params_dict.keys())
        metadata['parameters'] = sorted(existing_params)
    else:
        metadata = {
            'database': db_name,
            'parameters': sorted(params_dict.keys()),
        }

    metadata['N'] = N
    if mask_sig is not None:
        metadata['pCjIj_nonzero_indices'] = sorted(
            int(i) for i in np.where(np.asarray(mask_sig))[0]
        )

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Cached {len(params_dict)} parameters from {db_name}")





def load_expensive_params(db_name, param_names, N, mask_sig=None):
    """
    Load expensive calibration parameters from cache files.

    Parameters
    ----------
    db_name : str
        Database name (e.g., 'GTAP')
    param_names : list
        List of parameter names to load
    N : int
        Number of sectors; compared against cached value for invalidation.
    mask_sig : array-like of bool, optional
        If provided, the nonzero-index signature of this mask is compared
        against the value stored in the cache metadata.  A mismatch forces
        a full recalculation (returns ``None``).

    Returns
    -------
    dict or None
        Dictionary of loaded parameters, or None if cache doesn't exist or
        is stale.
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

    # Validate mask signature when the caller supplies one
    if mask_sig is not None:
        stored_indices = metadata.get('pCjIj_nonzero_indices')
        if stored_indices is None:
            print(f"Cache has no mask signature for {db_name}. Recalculating...")
            return None
        current_indices = sorted(int(i) for i in np.where(np.asarray(mask_sig))[0])
        if stored_indices != current_indices:
            print(f"Cache mask signature mismatch for {db_name}. Recalculating...")
            return None

    # Try to load each parameter
    for param_name in param_names:
        filepath = get_cache_filename(db_name, param_name)
        if not os.path.exists(filepath):
            print(f"Cache file missing for '{param_name}'. Recalculating all parameters...")
            return None

        data = np.load(filepath)
        # Convert single-element array back to scalar if needed
        if isinstance(data, np.ndarray) and data.shape == (1,):
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






def _compute_CDES_params(alphaCj0_nE, target_ni_j_nE, target_etaCj_nE, 
                               pCjCj_nE, pCj0_nE, R_nE, N_nE):
    """
    Compute CDES calibration parameters (betaCj, etaCj, A_Cj, etc.).
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
    
    solA_Cj = dict_least_squares(systemA_C, variables , parameters, bounds, VARIABLES_SPECS, check=False,verb=0)
    
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


def _compute_solI_params(pCj0, Ij0, Ri0, pCjIj, N_val):
    """
    Compute investment calibration parameters (I0, pI0, alphaIj).
    This function is separated for caching purposes.

    Parameters
    ----------
    pCj0 : np.ndarray, shape (N,)
        Composite-good consumer prices at baseline.
    Ij0 : np.ndarray, shape (N,)
        Baseline investment demand by good j.
    Ri0 : float
        Total nominal investment expenditure.
    pCjIj : np.ndarray, shape (N,)
        Investment expenditure by good j; nonzero entries define the
        active sector mask for the reduced solve.
    N_val : int
        Total number of sectors.

    Returns
    -------
    dict with keys ``'I0'`` (float), ``'pI0'`` (float),
    ``'alphaIj'`` (np.ndarray, shape (N,))
    """
    mask = pCjIj != 0
    len_alphaIj = int(mask.sum())

    def eqpI(pI, pCj, alphaIj):
        return -1 + pI / sum(pCj * alphaIj)

    def eqIj(Ij, alphaIj, I):
        return -1 + Ij / (alphaIj * I)

    def eqRi(Ri, pI, I):
        return -1 + Ri / (pI * I)

    def systemI(var, par):
        d = {**var, **par}
        return np.hstack([
            eqpI(pI=d['pI'], pCj=d['pCj'], alphaIj=d['alphaIj']),
            eqIj(Ij=d['Ij'],  alphaIj=d['alphaIj'], I=d['I']),
            eqRi(Ri=d['Ri'],  pI=d['pI'],            I=d['I']),
        ])

    variables = {
        'I':       Ri0 / sum(np.array([0.02] * N_val) * pCj0),
        'alphaIj': np.array([10] * len_alphaIj),
        'pI':      np.array([sum(np.array([0.02] * N_val) * pCj0)]),
    }

    parameters = {
        'pCj': pCj0[mask],
        'Ij':  Ij0[mask],
        'Ri':  Ri0,
    }

    bounds_dict = {
        'I':       [0, np.inf],
        'alphaIj': [0, np.inf],
        'pI':      [0, np.inf],
    }

    bounds = create_ordered_bounds(variables, bounds_dict)

    solI = dict_least_squares(systemI, variables, parameters, bounds, VARIABLES_SPECS,
                              check=False, verb=0)

    alphaIj_full = np.zeros(N_val)
    alphaIj_full[mask] = solI.dvar['alphaIj']

    return {
        'I0':      float(solI.dvar['I']),
        'pI0':     float(solI.dvar['pI']),
        'alphaIj': alphaIj_full,
    }