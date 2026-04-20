import numpy as np 
import import_EXIOBASE as imp
from copy import deepcopy as cp
import pandas as pd
from helpers.calibration_solvers import _compute_solI_params, _compute_CDES_params, load_expensive_params, save_expensive_params

sectors         = pd.read_csv("Solver/preprocessed_data/indexes/sectors.csv")["sector"].tolist()
energy_consumers = pd.read_csv("Solver/preprocessed_data/indexes/energy_consumers.csv")["energy_consumer"].tolist()
N = len(sectors)

A = sectors.index("AGRICULTURE")
M = sectors.index("MANUFACTURE")
SE = sectors.index("SERVICES")
E = sectors.index("ENERGY")
ST = sectors.index("STEEL")
CH = sectors.index("CHEMICAL")
T = sectors.index("TRANSPORTATION")





def _load_energy_matrix_from_csv(df, variable_type, row_labels, col_map, default_fill=0.0, calibration_year=None):
    """Build an (n_rows, 4) energy matrix from a calibration CSV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full calibration DataFrame read from the calibration CSV.
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
    np.ndarray, shape ``(len(row_labels), len(col_map))``
    """
    mat = np.full((len(row_labels), len(col_map)), default_fill, dtype=float)
    # Find the calibration-year column label (int or str depending on pandas version)
    year_col = next((c for c in df.columns if str(c) == str(calibration_year)), None)
    if year_col is None:
        raise ValueError(f"Column '{calibration_year}' not found in calibration CSV.")
    subset = df[df["Variable"] == variable_type]
    for _, row in subset.iterrows():
        consumer   = row["Energy consumers"]
        energy_use = row["Energy uses"]
        if consumer in row_labels and energy_use in col_map:
            r = row_labels.index(consumer)
            c = col_map[energy_use]
            mat[r, c] = float(row[year_col])
    return mat


def _load_trade_energy_from_csv(df, calibration_year):
    """Extract scalar trade-energy values from a hybridization CSV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full calibration DataFrame (e.g. ``hybridization_df.csv``).
    calibration_year : int or str
        Year to extract; matched against column names via ``str()`` comparison.

    Returns
    -------
    dict with keys ``'Xj0_E'``, ``'Mj0_E'``, ``'pXj0_E'``, ``'pMj0_E'``
    """
    year_col = next((c for c in df.columns if str(c) == str(calibration_year)), None)
    if year_col is None:
        raise ValueError(f"Column '{calibration_year}' not found in calibration CSV.")
    _var_map = {
        "Export|Energy":       "Xj0_E",
        "Import|Energy":       "Mj0_E",
        "Export|Energy Price": "pXj0_E",
        "Import|Energy Price": "pMj0_E",
    }
    result = {}
    for var_name, key in _var_map.items():
        rows = df[df["Variable"] == var_name]
        if rows.empty:
            raise ValueError(f"Variable '{var_name}' not found in calibration CSV.")
        result[key] = float(rows.iloc[0][year_col])
    return result


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


class calibrationVariables:
    
    def __init__(self, 
                 calibration_year, 
                 energy_calibration_data, 
                 population_calibration_data, 
                 armington_elasticities_df, 
                 export_elasticities_df, 
                 kl_elasticities_df, 
                 income_elasticities_df, 
                 compensated_price_elasticities_df, 
                 assumed_variables_df,
                 cache_dir="Solver/preprocessed_data/calibration/calibration_cache"):
        
        _av = assumed_variables_df["value"]

        _pop_year_cols = sorted([c for c in population_calibration_data.columns if str(c).lstrip('-').isdigit()], key=lambda c: int(str(c)))
        calibration_year = next(c for c in _pop_year_cols if str(c) == str(calibration_year))
        next_year = next(c for c in _pop_year_cols if int(str(c)) > int(calibration_year))
        

        self.L0=population_calibration_data[calibration_year].iloc[0]
        self.pL0=sum(imp.pLLj)/self.L0

        
        
        #prezzi
        
        self.pYj0=np.full(N, float(_av["pYj0"]))
        self.pSj0=np.full(N, float(_av["pSj0"]))
        self.pKLj0=np.full(N, float(_av["pKLj0"]))
        self.pXj0=np.full(N, float(_av["pXj0"]))
        self.pDj0=np.full(N, float(_av["pDj0"]))
        self.lambda_pXj=np.full(N, float(_av["lambda_pXj"]))

        self.pMj0=cp(self.pXj0)

        #taxes
        
        self.tauYj0 = imp.production_taxes/( imp.pYjYj - imp.production_taxes)

        self.tauSj0 = imp.sales_taxes / (imp.pCiYij.sum(axis=1)+imp.pCjCj+imp.pCjGj+imp.pCjIj - imp.sales_taxes)
        self.pCj0 = (1+cp(self.tauSj0))*cp(self.pSj0)
                
        #quantità
        self.Ij0 = imp.pCjIj/ cp(self.pCj0)
        self.Cj0 = imp.pCjCj/ cp(self.pCj0)
        self.Gj0 = imp.pCjGj/ cp(self.pCj0)
        self.Lj0 = imp.pLLj / cp(self.pL0)
        self.Yij0 = imp.pCiYij/ cp(self.pCj0[:,None])
        self.KLj0= imp.pKLjKLj / cp(self.pKLj0)
        self.Xj0= imp.pXjXj / cp(self.pXj0)
        self.Mj0= imp.pMjMj / cp(self.pMj0)
        self.Dj0= imp.pDjDj / cp(self.pDj0)
        self.Yj0= imp.pYjYj / cp(self.pYj0)        
        self.Sj0= imp.pSjSj / cp(self.pSj0)        

        #########################
        #### ENERGY COUPLING ####
        #########################


        #### ENERGY MATRICES ####


        _row_labels = energy_consumers

        _col_map = {eu: i for i, eu in enumerate(dict.fromkeys(
            energy_calibration_data.loc[
                energy_calibration_data["Variable"] == "Volume", "Energy uses"
            ].dropna()
        ))}
        _rhos_col_map = {eu: i for i, eu in enumerate(dict.fromkeys(
            energy_calibration_data.loc[
                energy_calibration_data["Variable"] == "Rho", "Energy uses"
            ].dropna()
        ))}

        self.E_vol = _load_energy_matrix_from_csv(
            energy_calibration_data, "Volume", _row_labels, _col_map, default_fill=0.0, calibration_year=calibration_year)

        self.pE    = _load_energy_matrix_from_csv(
            energy_calibration_data, "Price", _row_labels, _col_map, default_fill=0.0, calibration_year=calibration_year
            )*(1 + self.tauSj0[E]) #prezzo pieno di tasse per il settore energia
 
        # Rho has no PE rows in CSV; PE column defaults to 1.0
        self.rhoE  = _load_energy_matrix_from_csv(
            energy_calibration_data, "Rho", _row_labels, _rhos_col_map, default_fill=0.0, calibration_year=calibration_year)
        

        _row_map = {label: idx for idx, label in enumerate(_row_labels)}
        _households_idx = _row_map["HOUSEHOLDS"]
        _non_household_rows = [
            idx for idx in _row_map.values()
            if idx != _households_idx
        ]

        _trade_energy = _load_trade_energy_from_csv(energy_calibration_data, calibration_year)
        self.Xj0[E]  = _trade_energy["Xj0_E"]
        self.Mj0[E]  = _trade_energy["Mj0_E"]
        self.pXj0[E] = _trade_energy["pXj0_E"]
        self.pMj0[E] = _trade_energy["pMj0_E"]

        self.rhoM =  self.pMj0[E]/self.pE[E, _col_map["PE"]]
        self.rhoX =  self.pXj0[E]/self.pE[E, _col_map["PE"]]
        
        #adjusting energy quantity and price

        self.Sj0[E]=self.E_vol.sum()
        self.pSj0[E]=imp.pSjSj[E] / cp(self.Sj0[E])

        self.Dj0[E]=cp(self.Sj0[E])-cp(self.Mj0[E])
        self.pDj0[E]=imp.pDjDj[E] / cp(self.Dj0[E])

        self.Yj0[E]=cp(self.Xj0[E])+cp(self.Dj0[E])
        self.pYj0[E]=imp.pYjYj[E] / cp(self.Yj0[E])

        self.Yij0[E,:] = cp(self.E_vol.sum(axis=1)[:-1])
        self.Cj0[E] = cp(self.E_vol.sum(axis=1)[-1])

        self.pCj0[E] = imp.pCjCj[E]/cp(self.Cj0[E])
        self.pY_Ej = imp.pCiYij[E,:]/cp(self.Yij0[E,:])

        # Technical_coefficient: computed from intermediate variables (same as aYE_*j / Yj0).
        # Last row (HOUSEHOLDS) = 0 by construction.
        self.a_Ej = np.zeros((len(_row_labels), len(_col_map)))
        self.a_Ej = np.zeros((len(_row_labels), len(_col_map)))
        for eu, c in _col_map.items():
            self.a_Ej[:N, c] = cp(self.E_vol[_non_household_rows, c]) / cp(self.Yj0)


        
        
        #####################################
        #### OTHER CALIBRATION VARIABLES ####
        #####################################
        
        #scalari

        self.B0=sum(imp.pXjXj)-sum(imp.pMjMj)
        self.R0= sum(imp.pCjCj)
        self.Ri0= sum(imp.pCjIj)
        self.Rg0= sum(imp.pCjGj)
        self.GDP0= sum(imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pXjXj-imp.pMjMj)
        

        
        # elasticities
        _sigmaSj  = armington_elasticities_df.squeeze().reindex(sectors).to_numpy().astype(float)
        _sigmaXj  = export_elasticities_df.squeeze().reindex(sectors).to_numpy().astype(float)
        _sigmaKLj = kl_elasticities_df.squeeze().reindex(sectors).to_numpy().astype(float)
        self.sigmaSj  = _sigmaSj
        self.sigmaXj  = _sigmaXj
        self.sigmaKLj = _sigmaKLj

        self.etaSj  = (_sigmaSj  - 1) / _sigmaSj
        self.etaXj  = (_sigmaXj  - 1) / _sigmaXj
        self.etaKLj = (_sigmaKLj - 1) / _sigmaKLj

        
        self.aKLj= cp(self.KLj0)/ cp(self.Yj0)
        

        

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
        self.pXtp= cp(self.pXj0)
        self.Gtp= cp(self.Gj0)
        self.Itp= cp(self.Ij0)
        self.pXtp= cp(self.pXj0)
        self.Xtp= cp(self.Xj0)
        self.Mtp = cp(self.Mj0)
        
        #calibrate alphaIj, I and pI

        db_name = "EXIOBASE"
        solI_param_names = ['I0', 'pI0', 'alphaIj']
        _pCjIj_mask = imp.pCjIj != 0

        cached_solI = load_expensive_params(cache_dir, db_name, solI_param_names, N,
                                            mask_sig=_pCjIj_mask)
        if cached_solI is not None:
            self.I0      = cached_solI['I0']
            self.pI0     = cached_solI['pI0']
            self.alphaIj = cached_solI['alphaIj']
        else:
            computed_solI = _compute_solI_params(
                self.pCj0, self.Ij0, self.Ri0, imp.pCjIj, N
            )
            self.I0      = computed_solI['I0']
            self.pI0     = computed_solI['pI0']
            self.alphaIj = computed_solI['alphaIj']
            save_expensive_params(cache_dir, db_name, computed_solI, N, mask_sig=_pCjIj_mask)



        self.delta=float(_av["delta"])
        
        self.population_growth_rate = float(population_calibration_data[next_year].iloc[0]) / float(population_calibration_data[calibration_year].iloc[0]) - 1
        
        self.pK0 = (sum(imp.pKKj)*(cp(self.population_growth_rate)+cp(self.delta)))/ cp(self.I0)
        self.Kj0= imp.pKKj / cp(self.pK0)
        self.K0=sum(self.Kj0)
        self.alphaLj_CobbDouglas= imp.pLLj / imp.pKLjKLj
        self.alphaKj_CobbDouglas= imp.pKKj / imp.pKLjKLj
        self.GDPPI=float(_av["GDPPI"])
        self.alphaLj= compute_alphas_CES(Q1j= cp(self.Lj0),Q2j= cp(self.Kj0),p1j= cp(self.pL0),p2j= cp(self.pK0),etaj= cp(self.etaKLj))
        self.alphaKj= compute_alphas_CES(Q1j= cp(self.Kj0),Q2j= cp(self.Lj0),p1j= cp(self.pK0),p2j= cp(self.pL0),etaj= cp(self.etaKLj))
        # #this is 1 by default for the E sector so calibrate accordingly
        
        self.bKL=float(_av["bKL"])
        #self.bKLj = (cp(self.KLj0)/(
        #             cp(self.bKL)*
        #             np.float_power(cp(self.Lj0),cp(self.alphaLj_CobbDouglas))*
        #             np.float_power(cp(self.Kj0),cp(self.alphaKj_CobbDouglas))
        #             ))
        
        self.bKLj = cp(self.KLj0)*cp(self.bKL)/np.float_power(cp(self.alphaLj)*np.float_power(cp(self.Lj0),cp(self.etaKLj)) + cp(self.alphaKj) * np.float_power(cp(self.Kj0),cp(self.etaKLj)), 1/ cp(self.etaKLj))
        
        self.target_ni_j  = compensated_price_elasticities_df.squeeze().reindex(sectors).to_numpy()
        self.target_etaCj = income_elasticities_df.squeeze().reindex(sectors).to_numpy()
        
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
        db_name = "EXIOBASE"
        CDES_params_names = ['betaCj_nE', 'computed_ni_j_nE', 'computed_etaCj_nE', 
                                   'gammaCj_nE', 'u_C', 'A_Cj_nE', 'normalisation_factor']
        
        cached_params = load_expensive_params(cache_dir, db_name, CDES_params_names, N)
        
        if cached_params is not None:
            # Load from cache
            self.betaCj_nE = cached_params['betaCj_nE']
            
            self.gammaCj_nE = cached_params['gammaCj_nE']
            self.u_C = cached_params['u_C']
            self.A_Cj_nE = cached_params['A_Cj_nE']
            self.normalisation_factor = cached_params['normalisation_factor']
        else:
            # Compute CDES parameters
            computed_params = _compute_CDES_params(
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
            save_expensive_params(cache_dir, db_name, computed_params, N)
        


        self.lambda_KLM = float(_av["lambdaKLM"])

        self.aYij= cp(self.Yij0) / cp(self.Yj0[None,:])
        

        self.pCjtp= cp(self.pCj0)
        self.Ctp= cp(self.Cj0)
        
        self.aKLj0=cp(self.aKLj)
        self.aYij0=cp(self.aYij)

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

