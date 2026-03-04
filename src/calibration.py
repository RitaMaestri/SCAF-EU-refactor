import numpy as np 
import import_GTAP_data as imp
from import_GTAP_data import N,sectors
from copy import deepcopy as cp
import pandas as pd
import os
from helpers.calibration_solvers import _compute_solI_params, _compute_CDES_params, load_expensive_params, save_expensive_params

#import pandas as pd

A = sectors.index("AGRICULTURE")
M = sectors.index("MANUFACTURE")
SE = sectors.index("SERVICES")
E = sectors.index("ENERGY")
ST = sectors.index("STEEL")
CH = sectors.index("CHEMICAL")
T = sectors.index("TRANSPORTATION")






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

        db_name = "GTAP"
        solI_param_names = ['I0', 'pI0', 'alphaIj']
        _pCjIj_mask = imp.pCjIj != 0

        cached_solI = load_expensive_params(db_name, solI_param_names,
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
            save_expensive_params(db_name, computed_solI, mask_sig=_pCjIj_mask)



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
        CDES_params_names = ['betaCj_nE', 'computed_ni_j_nE', 'computed_etaCj_nE', 
                                   'gammaCj_nE', 'u_C', 'A_Cj_nE', 'normalisation_factor']
        
        cached_params = load_expensive_params(db_name, CDES_params_names)
        
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
            save_expensive_params(db_name, computed_params)
        
        #########################
        #### ENERGY COUPLING ####
        #########################


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

        _row_map = {label: idx for idx, label in enumerate(_row_labels)}
        _households_idx = _row_map["HOUSEHOLDS"]
        _non_household_rows = [
            idx for idx in _row_map.values()
            if idx != _households_idx
        ]

        # Technical_coefficient: computed from intermediate variables (same as aYE_*j / Yj0).
        # Columns: T=0, B=1, P=2, PE=3.  Last row (HOUSEHOLDS) = 0 by construction.
        self.a_Ej = np.zeros((len(_row_labels), 4))
        self.a_Ej[:N, 0] = cp(self.E_vol[_non_household_rows, _col_map["T"]]) / cp(self.Yj0)   # T
        self.a_Ej[:N, 1] = cp(self.E_vol[_non_household_rows, _col_map["B"]]) / cp(self.Yj0)   # B
        self.a_Ej[:N, 2] = cp(self.E_vol[_non_household_rows, _col_map["P"]]) / cp(self.Yj0)   # P
        self.a_Ej[:N, 3] = cp(self.E_vol[_non_household_rows, _col_map["PE"]]) / cp(self.Yj0)   # PE


        self.Yij0[E,:] = cp(self.E_vol.sum(axis=1)[:-1])
        self.Cj0[E] = cp(self.E_vol.sum(axis=1)[-1])

        self.pCj0[E] = imp.pCjCj[E]/cp(self.Cj0[E])
        self.pY_Ej = imp.pCiYij[E,:]/cp(self.Yij0[E,:])

        self.lambda_KLM = 1

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

