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

#import pandas as pd

A = sectors.index("AGRICULTURE")
M = sectors.index("MANUFACTURE")
SE = sectors.index("SERVICES")
E = sectors.index("ENERGY")
ST = sectors.index("STEEL")
CH = sectors.index("CHEMICAL")
T = sectors.index("TRANSPORTATION")

shares = pd.read_csv("/home/rita/Documents/Tesi/Code/REMIND_energy_coupling/data/shares.csv", index_col="variable")


tau=imp.labor_taxes / (imp.pLLj - imp.labor_taxes)
pL=1
w=pL/(1+tau)

def division_by_zero(num,den):
    if len(num)==len(den):
        n=len(num)
    else:
        print("denominator and numerator have different len")
        sys.exit()
    result=np.zeros(n)
    result[den!=0]=num[den!=0]/den[den!=0]
    return result

def compute_intermediate_prices(idx_E, pCj, p_CEj):
        intermediate_prices=np.repeat( cp(pCj), len(pCj), axis=0 )
        intermediate_prices_matrix = intermediate_prices.reshape(len(pCj), len(pCj))

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


class calibrationVariables:
    
    def __init__(self, L_gr0, L0=None):
        
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
        self.tauL0 = imp.labor_taxes / (imp.pLLj - imp.labor_taxes)
        self.pCj0 = (1+cp(self.tauSj0))*cp(self.pSj0)
        self.w= cp(self.pL0)/(1+cp(self.tauL0))
        
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

        self.T0= sum(imp.production_taxes + imp.sales_taxes + imp.labor_taxes)
        self.B0=sum(imp.pXjXj)-sum(imp.pMjMj)
        self.R0= sum(imp.pCjCj)
        self.Ri0= sum(imp.pCjIj)
        self.Rg0= sum(imp.pCjGj)
        self.l0=sum(cp(self.Lj0)/ cp(self.KLj0))
        self.uL0 = 0.105
        self.sigmaw= 0.
        self.uK0 = 0.105
        self.sigmapK= -0.1
        self.GDP0= sum(imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pXjXj-imp.pMjMj)
        
        # parametri
        self.sigmaXj=imp.sigmaXj.astype(float)
        self.sigmaSj=imp.sigmaSj.astype(float)
        self.sigmaKLj=imp.sigmaKLj.astype(float)

        #self.etaXj=(imp.sigmaXj-1)/imp.sigmaXj
        
        self.etaSj=(imp.sigmaSj-1)/imp.sigmaSj
        self.etaXj=(imp.sigmaXj-1)/imp.sigmaXj
        self.etaKLj=(imp.sigmaKLj-1)/imp.sigmaKLj

        
        self.aKLj= cp(self.KLj0)/ cp(self.Yj0)
        self.lambda_KL = 1
        
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
        self.lambda_E = 1
        self.lambda_nE = 1
        self.alphaGj = imp.pCjGj/ cp(self.Rg0)
        self.alphalj = cp(self.Lj0)/(cp(self.KLj0)*cp(self.l0))
        self.alphaw = cp(self.w)/(cp(self.uL0)**cp(self.sigmaw))
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
        #self.betaRj= (imp.epsilonPCj+1)/(self.alphaCj-1)
        #self.epsilonRj=imp.epsilonRj
        self.sD0=sum(imp.pCjIj+imp.pXjXj-imp.pMjMj)/ cp(self.GDP0)
        
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
        self.g0=L_gr0
        self.pK0 = (sum(imp.pKKj)*(cp(self.g0)+cp(self.delta)))/ cp(self.I0)
        self.Kj0= imp.pKKj / cp(self.pK0)
        self.K0=sum(self.Kj0)
        self.alphapK = cp(self.pK0)/(cp(self.uK0)**cp(self.sigmapK))
        self.alphaIK = cp(self.Ri0)/ cp(self.K0)
        self.K0next = cp(self.K0) * (1-cp(self.delta)) + cp(self.I0)
        self.L0u=sum(self.Lj0)/(1-cp(self.uL0))
        self.K0u=sum(self.Kj0)/(1-cp(self.uK0))
        self.K0u_next= cp(self.K0u) * (1-cp(self.delta)) + cp(self.I0)
        self.GDPPI=1
        self.CPI=1
        self.alphaLj= compute_alphas_CES(Q1j= cp(self.Lj0),Q2j= cp(self.Kj0),p1j= cp(self.pL0),p2j= cp(self.pK0),etaj= cp(self.etaKLj))
        self.alphaKj= compute_alphas_CES(Q1j= cp(self.Kj0),Q2j= cp(self.Lj0),p1j= cp(self.pK0),p2j= cp(self.pL0),etaj= cp(self.etaKLj))
        self.gammaj = compute_theta_CES(Zj= cp(self.KLj0), alpha1j= cp(self.alphaKj), alpha2j= cp(self.alphaLj), Q1j= cp(self.Kj0), Q2j= cp(self.Lj0),etaj= cp(self.etaKLj))
        #this is 1 by default for the E sector so calibrate accordingly
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
        
        constraint_with_params = partial(eqni_j, alphaCj=self.alphaCj0_nE, N=N_nE)
        
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
                     "ni_j":self.target_ni_j_nE*0.99}
        
        parameters={'target_ni_j':self.target_ni_j_nE,
                    'alphaCj0':self.alphaCj0_nE
                    }
        
        #beta must be either between 0 and 1 or less than 0 for every j
        bound_dict = {
            "betaCj": [-np.inf, -0.001],
            "ni_j": [-np.inf, np.inf],
        }
        
        bounds = create_ordered_bounds(variables, bound_dict)
        
        solNi_constrained = dict_minimize(systemNi, variables , parameters, N, bounds, constraint=ni_j_constraint )
        
        systemNi(solNi_constrained.dvar, parameters)
        
        self.computed_ni_j_nE=solNi_constrained.dvar["ni_j"]
        self.betaCj_nE=solNi_constrained.dvar["betaCj"]
        
        eqni_j(solNi_constrained.x,self.alphaCj0_nE,N_nE)
        ni_calibration_error=(self.computed_ni_j_nE-self.target_ni_j_nE)/self.target_ni_j_nE

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
            
        eq_constraint_etaCj_with_params = partial(eq_constraint_etaCj, alphaCj0=self.alphaCj0_nE, betaCj=self.betaCj_nE, N=N_nE)

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
        

        sign_constraint_with_params = partial(sign_eq, target_etaCj=self.target_etaCj_nE, N=N_nE)

        scipy_sign_constraint = scipy.optimize.NonlinearConstraint(
            fun=sign_constraint_with_params,
            lb=0,  
            ub=np.inf 
        )
        
        
                ######  CONSTRAINT 3  #######
        
        # def Engel_eq(var, alphaCj0,N):
        #     gammaCj=var[:N]
        #     etaCj=var[N:]
            
        #     one = sum(alphaCj0*etaCj)
        #     return one
        
        # Engel_constraint_with_params = partial(Engel_eq, alphaCj0=self.alphaCj0,N=N)
        
        # scipy_Engel_constraint = scipy.optimize.NonlinearConstraint(
        #     fun=Engel_constraint_with_params,
        #     lb=1-1e-8,  
        #     ub=1+1e-8,
        #     keep_feasible=False
        # )
        

        
        # def eq_ratio(el,target_el):
        #     result = 1-el/target_el
        #     return result
        
        
        
        #### OPTIMIZATION EQAUATION ######
        
        def systemEta_j(var, par):
            d = {**var, **par}
            return eq_difference(d["alphaCj0"],d["etaCj"],d["target_etaCj"])

        guess_gammaCj=np.array(4.+8.*1/np.array(range(1,7)))
        guess_etaCj= eq_etaCj(guess_gammaCj, self.alphaCj0_nE, self.betaCj_nE)+1e-2
        
        variables = { "gammaCj": guess_gammaCj,
                      'etaCj':guess_etaCj}
        
        parameters={'target_etaCj':self.target_etaCj_nE,
                    'betaCj':self.betaCj_nE,
                    'alphaCj0':self.alphaCj0_nE, 
                    }
        
        bounds_dict={ "gammaCj": [0,np.inf],
                      'etaCj': [0, np.inf]}
        
        
        bounds = create_ordered_bounds(variables, bounds_dict)


        solEtaj_constrained = dict_minimize(systemEta_j, variables , parameters, N_nE, bounds, constraint=[scipy_etaCj_constraint, scipy_sign_constraint ] )
        
        self.computed_etaCj_nE=solEtaj_constrained.dvar["etaCj"]
        self.gammaCj_nE=solEtaj_constrained.dvar["gammaCj"]
        
        
        eq_etaCj(solEtaj_constrained.dvar["gammaCj"], self.alphaCj0_nE, self.betaCj_nE)-solEtaj_constrained.dvar["etaCj"]

        #Engel_constraint_with_params(solEtaj_constrained.x)
        
        
        
        etai_calibration_error=(self.computed_etaCj_nE-self.target_etaCj_nE)/self.target_etaCj_nE
        
        ##################################
        ########## finding A_Cj ##########
        ##################################
        
        self.u_C=1
        
        def eq_A_Cj(pCjCj,A_Cj,betaCj,u_C,gammaCj,pCj,R0):
            Z_j = A_Cj * betaCj * u_C ** (gammaCj * betaCj) * (pCj / R0) ** betaCj
            zero=1-pCjCj/(R0*Z_j/sum(Z_j))
            return zero
        
        
        def systemA_C(var, par):
            d = {**var, **par}
            return eq_A_Cj(d["pCjCj"],d["A_Cj"],d["betaCj"],d["u_C"],d["gammaCj"],d["pCj"],d["R0"])
        
        
        variables = { 'A_Cj': np.array([0.3]*N_nE)}

        
        parameters={'pCjCj':self.pCjCj_nE,
                    'betaCj':self.betaCj_nE, 
                    'u_C':self.u_C,
                    "gammaCj":self.gammaCj_nE,
                    "pCj":self.pCj0_nE,
                    "R0":self.R_nE
                    }
        
        bounds_dict= { 'A_Cj': [0,np.inf],
                   }
        
        bounds= create_ordered_bounds(variables, bounds_dict)
        
        solA_Cj = dict_least_squares(systemA_C, variables , parameters, bounds, N, check=False,verb=0)
        
        self.A_Cj_nE=solA_Cj.dvar["A_Cj"]
        
        self.normalisation_factor= sum(self.A_Cj_nE * self.u_C ** (self.gammaCj_nE * self.betaCj_nE) * (self.pCj0_nE / self.R_nE) ** self.betaCj_nE)
        
        #eq_A_Cj(imp.pCjCj,self.A_Cj,self.betaCj,self.u_C,self.gammaCj,self.pCj0,self.R0)
        
        
        
        
        
        
        

# _____ _   _ _____ ____   ______   __   ____ ___  _   _ ____  _     ___ _   _  ____ 
#| ____| \ | | ____|  _ \ / ___\ \ / /  / ___/ _ \| | | |  _ \| |   |_ _| \ | |/ ___|
#|  _| |  \| |  _| | |_) | |  _ \ V /  | |  | | | | | | | |_) | |    | ||  \| | |  _ 
#| |___| |\  | |___|  _ <| |_| | | |   | |__| |_| | |_| |  __/| |___ | || |\  | |_| |
#|_____|_| \_|_____|_| \_\\____| |_|    \____\___/ \___/|_|   |_____|___|_| \_|\____|
#                                                                                    
        
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
        self.s_LDV_nonT = float(shares.loc["s_LDV_nonT"])

        self.s_trucks_T = float(shares.loc["s_trucks_T"])
        self.s_trucks_nonT = float(shares.loc["s_trucks_nonT"])
        
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
        self.Rh_E = imp.pCjCj[E]
        self.Rh_nE = self.R0 - self.Rh_E
        self.pCjtp= cp(self.pCj0)
        self.Ctp= cp(self.Cj0)
        
        self.lambda_KLM = 1
        self.lambda_XMj=np.array([float(0)]*N)
        
        self.aKLj0=cp(self.aKLj)
        self.aYij0=cp(self.aYij)
        
        self.rhoB=cp(self.pE_B)/cp(self.pE_Ej[E])
        self.rhoTT=cp(self.pE_TT)/cp(self.pE_Ej[E])
        self.rhoTnT=cp(self.pE_TnT)/cp(self.pE_Ej[E])
        self.rhoPj= cp(self.pE_Pj)/cp(self.pE_Ej[E])





a=calibrationVariables(0.003985893420850095)


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




